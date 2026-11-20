"""C 表合并：c_eval 的 <scene>__<arm>.jsonl + .done.json → 驾驶指标、沿自车轨迹的感知保真（按接管后时间分段）、臂间配对差、效率（md + json）。

    venv/bin/python c_table/c_merge.py <eval_dir> [--boot 2000] [--ref copy] [--out-dir DIR]

口径
  * 场景集合：表 1 / 2 = 全部完成 main 任务的场景（log / copy / 各预测器）；表 3 = 有 3dgs 的场景子集上的同一批指标（各臂在同一批场景上配对）。
  * 驾驶（场景级）：有责碰撞率（closed_loop 的 NAVSIM 式分类；另报 agent 组、任意碰撞）、可行驶区域合规（全程自车框都在 GT drivable 的场景占比，
    与逐步合规率）、行驶进度（沿日志路线的弧长 / 日志同期弧长）、最大横向偏离。
  * 感知保真（与 B 表同一套 GT）：IoU 池化（Σ交 / Σ并）、recall@2m = Σtp / Σgt、lag 中位数（正 = 框落在车后面）、lag_recall = Σ匹配 / Σ折线完整的 GT 车辆；
    按接管后时间 τ ∈ (0,1] / (1,3] / (3,6] / (6,∞) s 分段（τ = (j − W + 1) × 0.5 s，第一步 τ = 0.5）。
  * 统计：逐场景配对 bootstrap 95% CI（场景内先求和再重采样）+ Wilcoxon 符号秩（场景级值；比例类用场景内池化值）。
  * 自检：同一场景各臂的 warmup 摘要必须相同（c_eval 的 warm_digest），否则报错。
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

RENDER = ('3dgs', '3dgs_difix')     # 渲染臂（只在已重建场景上跑）
BINS = ((0.0, 1.0, '0–1 s'), (1.0, 3.0, '1–3 s'), (3.0, 6.0, '3–6 s'), (6.0, 1e9, '>6 s'), (0.0, 1e9, '全程'))
SUMS = ('lane_i', 'lane_u', 'drv_i', 'drv_u', 'n_gt', 'n_tp', 'lag_gt', 'lag_m')


def load(dirs):
    """多个运行目录的并集（第一轮 + 追加臂的后续作业）。同一 (场景, 臂) 出现在多个目录时核对 warmup 摘要与行数，保留第一个目录的。"""
    summ, rows, preds, fails = {}, defaultdict(dict), {}, []
    n_dup = n_bad = 0
    for d in dirs:
        done = sorted(glob.glob(os.path.join(d, 'scene*__*.done.json')))
        n_new = 0
        for p in done:
            D = json.load(open(p))
            if 'arm' not in D:
                continue                                    # 旧格式（任务级）跳过
            arm, si = D['arm'], D['si']
            if si in summ.get(arm, {}):
                n_dup += 1
                n_bad += int(summ[arm][si]['warm_digest'] != D['summary']['warm_digest'] or len(rows[arm][si]) != D['rows'])
                continue
            path = p.replace('.done.json', '.jsonl')
            rs = [json.loads(l) for l in open(path)]
            if len(rs) != D['rows']:
                raise ValueError(f'{path}: 行数 {len(rs)} ≠ done.json 的 {D["rows"]}')
            summ.setdefault(arm, {})[si] = D['summary']
            rows[arm][si] = rs
            if D.get('pred'):
                if arm in preds and preds[arm]['path'] != D['pred']['path']:
                    raise ValueError(f'臂 {arm} 在不同运行里指向不同 ckpt：{preds[arm]["path"]} vs {D["pred"]["path"]}')
                preds[arm] = D['pred']
            n_new += 1
        fl = glob.glob(os.path.join(d, 'failed_*'))
        fails += fl
        print(f'[load] {d}: 并入 {n_new} 个（场景, 臂），失败任务 {len(fl)}')
    if n_dup:
        print(f'[load] 多个目录重复的（场景, 臂）{n_dup} 个，其中 warmup 摘要 / 行数不一致 {n_bad} 个' + ('  ⚠️' if n_bad else ''))
    print(f'[load] 臂 {sorted(summ)}')
    return summ, rows, preds, fails


def check_warm(summ):
    by_si = defaultdict(dict)
    for arm, d in summ.items():
        for si, s in d.items():
            by_si[si][arm] = s['warm_digest']
    bad = [si for si, d in by_si.items() if len(set(d.values())) > 1]
    if bad:
        raise ValueError(f'warmup 摘要在臂间不一致的场景：{bad[:10]}（流程有 bug，结果不可用）')
    return len(by_si)


def bin_sums(rs, lo, hi):
    S = np.zeros(len(SUMS))
    lags = []
    for r in rs:
        if not (lo < r['tau'] <= hi):
            continue
        S += [r['lane_i'], r['lane_u'], r['drv_i'], r['drv_u'], r['n_gt'], r['n_tp'],
              max(r['lag_n_gt'], 0), r['lag_n_match']]
        lags += r['lag_s']
    return S, lags


def ratio(S, a, b):
    return S[..., SUMS.index(a)] / np.maximum(S[..., SUMS.index(b)], 1e-9)


def perc(S, name):
    return {'lane_iou': ratio(S, 'lane_i', 'lane_u'), 'drv_iou': ratio(S, 'drv_i', 'drv_u'), 'recall': ratio(S, 'n_tp', 'n_gt'),
            'lag_recall': ratio(S, 'lag_m', 'lag_gt')}[name]


def boot(vals_a, vals_b, n_boot, rng, stat=np.mean):
    """场景级配对 bootstrap（vals_*：[n_scene] 或 [n_scene, k]，stat 作用于重采样后的第 0 维）。返回 (Δ, lo, hi)。"""
    a, b = np.asarray(vals_a, np.float64), np.asarray(vals_b, np.float64)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = stat(a, b, None)
    bs = np.array([stat(a[i], b[i], None) for i in idx])
    return float(d), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def mean_diff(a, b, _):
    return np.nanmean(a) - np.nanmean(b)


def pooled_diff_factory(name):
    def f(a, b, _):
        return float(perc(a.sum(0), name) - perc(b.sum(0), name))
    return f


def wilcoxon(a, b):
    try:
        from scipy.stats import wilcoxon as w
        d = np.asarray(a, float) - np.asarray(b, float)
        d = d[np.isfinite(d)]
        if len(d) < 5 or np.all(d == 0):
            return float('nan')
        return float(w(d, zero_method='wilcox').pvalue)
    except Exception:
        return float('nan')


def fmt(v, nd=3):
    return '—' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{v:.{nd}f}'


def fci(t, nd=3):
    d, lo, hi = t
    star = '★' if (lo > 0 or hi < 0) else ''
    return f'{d:+.{nd}f} [{lo:+.{nd}f},{hi:+.{nd}f}]{star}'


def scene_drive(summ, arm, sis):
    s = [summ[arm][si] for si in sis]
    return dict(at_fault=np.array([x['at_fault'] for x in s], float), at_fault_agent=np.array([x['at_fault_agent'] for x in s], float),
                any_coll=np.array([x['any_coll'] for x in s], float), dac=np.array([x['dac'] for x in s], float),
                dac_frac=np.array([x['dac_frac'] for x in s], float), progress=np.array([x['progress'] for x in s], float),
                max_abs_lat=np.array([x['max_abs_lat'] for x in s], float))


def section(lines, title, summ, rows, arms, sis, ref, n_boot, rng, out_json, extra_pairs=()):
    lines += [f'\n## {title}（{len(sis)} 个场景）\n']
    dr = {arm: scene_drive(summ, arm, sis) for arm in arms}
    lines += ['### 驾驶（场景级）\n', '| 臂 | 有责碰撞率 | 其中 agent | 任意碰撞 | 全程在路面 | 逐步在路面 | 进度（中位） | 最大横偏 中位 m |',
              '|---|---|---|---|---|---|---|---|']
    for arm in arms:
        d = dr[arm]
        lines.append(f'| {arm} | {fmt(d["at_fault"].mean())} | {fmt(d["at_fault_agent"].mean())} | {fmt(d["any_coll"].mean())} | {fmt(d["dac"].mean())} | '
                     f'{fmt(np.nanmean(d["dac_frac"]))} | {fmt(np.nanmedian(d["progress"]))} | {fmt(np.median(d["max_abs_lat"]), 2)} |')
        out_json.setdefault(title, {}).setdefault(arm, {})['drive'] = {k: float(np.nanmean(v)) for k, v in d.items()}
    # 感知（分段）
    lines += ['\n### 沿自车轨迹的感知保真（与仿真位姿下的 GT 比；IoU 池化、recall@2m、lag 中位 s、lag_recall）\n',
              '| 臂 | 段 | lane IoU | drivable IoU | recall@2m | lag 中位 | lag_recall | GT 车辆数 |', '|---|---|---|---|---|---|---|---|']
    S = {}
    for arm in arms:
        for lo, hi, bn in BINS:
            per = [bin_sums(rows[arm][si], lo, hi) for si in sis]
            Sm = np.stack([p[0] for p in per])
            lags = sum((p[1] for p in per), [])
            S[(arm, bn)] = Sm
            tot = Sm.sum(0)
            lines.append(f'| {arm} | {bn} | {fmt(perc(tot, "lane_iou"))} | {fmt(perc(tot, "drv_iou"))} | {fmt(perc(tot, "recall"))} | '
                         f'{fmt(float(np.median(lags)) if lags else None, 3)} | {fmt(perc(tot, "lag_recall"))} | {int(tot[SUMS.index("n_gt")])} |')
            out_json[title][arm].setdefault('perc', {})[bn] = {m: float(perc(tot, m)) for m in ('lane_iou', 'drv_iou', 'recall', 'lag_recall')}
    # 配对差：每个臂 vs ref
    pairs = ([(arm, ref) for arm in arms if arm != ref] if ref in arms else []) + [(x, y) for x, y in extra_pairs if x in arms and y in arms and y != ref]
    lines += [f'\n### 配对差（a − b；逐场景 bootstrap 95% CI，★ = 不含 0；p = Wilcoxon 符号秩）\n',
              '| a − b | 有责碰撞率 | 全程在路面 | 进度 | recall@2m 1–3 s | recall@2m 全程 | lane IoU 全程 | drivable IoU 全程 |', '|---|---|---|---|---|---|---|---|']
    for arm, ref_b in pairs:
        a, b = dr[arm], dr[ref_b]
        cells = []
        for k in ('at_fault', 'dac', 'progress'):
            t = boot(a[k], b[k], n_boot, rng, mean_diff)
            cells.append(f'{fci(t)} p={fmt(wilcoxon(a[k], b[k]), 3)}')
        for bn, m in (('1–3 s', 'recall'), ('全程', 'recall'), ('全程', 'lane_iou'), ('全程', 'drv_iou')):
            t = boot(S[(arm, bn)], S[(ref_b, bn)], n_boot, rng, pooled_diff_factory(m))
            va = [perc(x, m) for x in S[(arm, bn)]]
            vb = [perc(x, m) for x in S[(ref_b, bn)]]
            cells.append(f'{fci(t)} p={fmt(wilcoxon(va, vb), 3)}')
        lines.append(f'| {arm} − {ref_b} | ' + ' | '.join(cells) + ' |')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('eval_dirs', nargs='+', help='一个或多个 c_eval 输出目录（第一轮 + 追加臂的作业），取并集')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--ref', default='copy', help='配对差的基线臂（另外总会给对 log 的差）')
    ap.add_argument('--out-dir', default='')
    ap.add_argument('--pairs', default='rof16_3000:rof8cont_3000,rofull6000:ronomap6000,tffull6000:rofull6000,assoc3000:rof8cont_3000,rofull6000@det:rofull6000,ronomap6000@det:ronomap6000,ronomap6000@det:copy,nolayout6000:ronomap6000,nolayout6000:copy',
                    help='额外配对差 a:b（两臂都在时才出）；默认含 O4a−O4b（F=16 vs F=8 续训）、地图、rollout 训练、关联通道')
    a = ap.parse_args()
    out = a.out_dir or a.eval_dirs[0]
    os.makedirs(out, exist_ok=True)
    summ, rows, preds, fails = load(a.eval_dirs)
    n_sc = check_warm(summ)
    rng = np.random.default_rng(0)
    main_arms = [x for x in ('log', 'copy') if x in summ] + sorted(x for x in summ if x not in ('log', 'copy') + RENDER)
    rend_arms = [x for x in RENDER if x in summ]
    sis_main = sorted(set.intersection(*[set(summ[x]) for x in main_arms])) if main_arms else []
    cover = '，'.join(f'{x} {len(summ[x])}' for x in main_arms + rend_arms)
    lines = [f'# C 表：整段闭环（{", ".join(a.eval_dirs)}）\n',
             f'- 场景：全部 main 臂都有的 {len(sis_main)} 个（各臂覆盖：{cover}）；失败任务 {len(fails)}；warmup 摘要臂间一致（{n_sc} 个场景）。',
             f'- 预测器：' + '；'.join(f'`{n}` = {p["path"]}（step {p["step"]}，map_layout={p["map_layout"]}，assoc={p.get("assoc", False)}，ctx={p["ctx"]}）' for n, p in sorted(preds.items())),
             '- 自车更新：运动学自行车模型跟踪 UniAD 规划的第一个路点（0.5 s 一步）；其他 agent 按日志回放（非反应式）；导航命令 = 日志路线 + 仿真位姿。'
             '碰撞 = NAVSIM 式有责分类（closed_loop.py 文档）；`log` 行是"位姿强制沿日志 + 真图"的参照（驾驶指标 = 日志自车本身）。']
    J = {}
    if sis_main:
        xp = [tuple(p.split(':')) for p in a.pairs.split(',') if p]
        section(lines, '全部场景（main 臂）', summ, rows, main_arms, sis_main, a.ref, a.boot, rng, J, xp)
        if a.ref != 'log' and 'log' in main_arms:
            section(lines, '全部场景（对 log 的差）', summ, rows, main_arms, sis_main, 'log', a.boot, rng, J)
    if rend_arms:
        # 已重建场景子集：所有渲染臂都有的场景（3dgs 与 3dgs_difix 同一批 30 个），配对差以 rend_arms[0]（3dgs）为基线，
        # 另给 3dgs_difix 为基线的差（审稿 W2：LatentLoop 对最强渲染基线）
        sis3 = sorted(set.intersection(set(sis_main), *[set(summ[x]) for x in rend_arms]))
        if sis3:
            section(lines, '3DGS 子集（已重建场景，各臂同一批场景）', summ, rows, main_arms + rend_arms, sis3, rend_arms[0], a.boot, rng, J)
            for rb in rend_arms[1:]:
                section(lines, f'3DGS 子集（对 {rb} 的差）', summ, rows, main_arms + rend_arms, sis3, rb, a.boot, rng, J)
    # 效率
    lines += ['\n## 效率（逐步中位数，s；仿真每步 = 产生 UniAD 输入特征的代价，栈每步 = 仿真 + UniAD 从 FPN 起跑）\n',
              '| 臂 | 条件准备 | 预测器 / 渲染 | JPEG 存 + 读 | 主干 | 仿真每步合计 | UniAD | 栈每步合计 | 步数 |', '|---|---|---|---|---|---|---|---|---|']
    for arm in main_arms + rend_arms:
        rs = [r for d in rows[arm].values() for r in d]

        def m(k):                           # 中位数：每个场景第一步含 CUDA / 渲染预热（3DGS 首个请求约 6 s），均值会被它拉高
            v = [r[k] for r in rs if k in r]
            return float(np.median(v)) if v else 0.0
        cond, pred, rend, save, io, bb, un = m('t_cond'), m('t_pred'), m('t_render'), m('t_save'), m('t_io'), m('t_backbone'), m('t_uniad')
        dfx = m('t_difix')
        isr = arm in RENDER
        sim = cond + pred + rend + dfx + save + (io if isr else 0.0) + bb
        lines.append(f'| {arm} | {fmt(cond)} | {fmt(pred if not isr else rend + dfx)} | {fmt(save + io if isr else 0.0)} | {fmt(bb)} | '
                     f'{fmt(sim)} | {fmt(un)} | {fmt(sim + un)} | {len(rs)} |')
        J.setdefault('efficiency', {})[arm] = dict(cond=cond, pred=pred, render=rend, difix=dfx, save=save, io=io, backbone=bb, uniad=un, sim=sim, n=len(rs))
    lines.append('\n注：`log` 的"主干"是真图主干，仿真每步不含读原图；`3dgs` 的"预测器 / 渲染"列是服务端渲染（6 相机；`3dgs_difix` 另加 Difix 6 张），JPEG 存 + 读是为了与真图走同一条加载路径（同 B 表），不是渲染器固有开销。')
    with open(os.path.join(out, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(J, f, indent=1, ensure_ascii=False)
    print('\n'.join(lines[:4]))
    print(f'[merge] → {os.path.join(out, "summary.md")}')


if __name__ == '__main__':
    main()
