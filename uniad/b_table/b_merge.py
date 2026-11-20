"""B 表合并：汇总 b_eval 的 scene*.jsonl → "指标 vs δ / s" 表（md + json）和图。

    venv/bin/python b_table/b_merge.py <run_dir> [<run_dir2> ...] [--kmin 1 --kmax 8] [--boot 1000] [--pairs a:b,c:d] [--extra DIR] [--out-dir DIR]

口径（继承 offlog/g4_merge.py，11-03 用户拍板）
  * IoU 池化（Σ交 / Σ并），recall@2m = Σtp/Σgt，感知类用全部锚点；规划类（plan∈drivable）只统计"该扰动下 [kmin,kmax] 每一步自车都在 GT 可行驶区域"的锚点。
  * Δ vs real = 该臂 − 同一批锚点上 real 臂（δ=0 真图）。理想仿真器的曲线是贴近 real 水平的一条平线，所以同时给**水平**（Δ vs real）和**斜率**（|δ| 相对同臂 δ=0）。
  * CI：按锚点做配对 bootstrap（每个锚点先聚合成可加和的统计量，再重采样，向量化）。中位数类（中心误差、lag）用锚点内样本拼接后取中位数，bootstrap 次数取 min(boot, 500)。
  * lag：以 GT 世界状态为参照（offlog/det_lag_gt.py），lag_med 正值 = 框落在车后面；lag_recall = Σ匹配 / Σ折线完整的 GT 车辆；n_lag = 进入 lag 统计的样本数（相对速度 ≥ vmin）。
  * 臂间配对差（--pairs，默认自动）：回应 W2（pred vs pred@zero / @log / @shuf：action 通路）、地图贡献（完整条件 vs 不画地图）、预测器 vs copy。
  * identity 臂（若有）必须与 real 逐行相同，否则直接报错。
  * 多个位置参数 = 主目录的并集（如 11-06 主作业 + 11-07 补跑的 9 个场景）：同一行键 (scene, anchor, fam, val, arm, k) 出现在多个目录时逐行核对（数值字段），
    不同只告警（计数），保留第一个目录的行。
  * `--extra DIR`（可重复）：并入另一个运行目录（如 3DGS 臂单独一个作业、BEV-Warp / 航向扰动的补充作业）。吸收规则（11-07 起）：行键不在主目录里，且
    （该臂是主目录没有的新臂，或该行的 (scene, anchor) 是主目录已有的锚点）——后者让"已有臂的新扰动族"（如 yaw）能并进来，同时不把补充作业里多出来的锚点
    混进已有臂（各臂锚点集合保持一致）。两边都有的行键逐行核对，不同只告警（计数），不并入。11-06 的用法（只有新臂）行为不变。
    表 1 的水平 / Δ 只在与 real 有共同锚点的范围内算，所以 3DGS（只覆盖已重建的场景）与预测器的比较看第 3 张表（配对差，取共同锚点）。
  * 扰动族：lat（δ，m）/ speed（倍率）/ yaw（航向偏置 ψ，deg）。斜率表对 lat 给 |δ|=0.5/1/2，对 yaw 给 |ψ|（两侧合并）。
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

COLS = ('lane_i', 'lane_u', 'drv_i', 'drv_u', 'n_gt', 'n_tp', 'pin_s', 'pin_n', 'lag_gt', 'lag_m')
METRICS = ('lane_iou', 'drv_iou', 'recall', 'plan_in_drv', 'lag_recall')
MEDIANS = ('cerr_med', 'lag_med')
SHOW = ('lane_iou', 'drv_iou', 'recall', 'plan_in_drv', 'lag_med', 'lag_recall')


def load(run_dir):
    paths = sorted(glob.glob(os.path.join(run_dir, 'scene*.jsonl')))
    tmp = glob.glob(os.path.join(run_dir, 'scene*.jsonl.tmp'))
    done = sorted(glob.glob(os.path.join(run_dir, 'scene*.done.json')))
    if tmp or len(done) != len(paths):
        print(f'[warn] 分片不完整：完成 {len(paths)} 个场景（done.json {len(done)} 个），未完成 {len(tmp)} 个')
    rows = []
    for d in done:
        with open(d) as f:
            n_exp = json.load(f)['rows']
        with open(d.replace('.done.json', '.jsonl')) as f:
            rs = [json.loads(l) for l in f]
        if len(rs) != n_exp:
            raise ValueError(f'{d}: 行数 {len(rs)} 与 done.json 记录的 {n_exp} 不符')
        rows += rs
    print(f'[load] {run_dir}：{len(done)} 个场景、{len(rows)} 行')
    return rows


def check_identity(rows):
    real = {(r['si'], r['anchor'], r['k']): r for r in rows if r['arm'] == 'real'}
    ident = [r for r in rows if r['arm'] == 'identity']
    for r in ident:
        a = real[(r['si'], r['anchor'], r['k'])]
        for f in r:
            if f != 'arm' and r[f] != a[f]:
                raise ValueError(f'identity 臂与 real 不同：scene {r["si"]} anchor {r["anchor"]} k={r["k"]} 字段 {f}（协议 / 状态恢复有 bug，结果不可用）')
    if ident:
        print(f'[check] identity 臂 {len(ident)} 行与 real 逐位相同')


NUM_FIELDS = ('lane_i', 'lane_u', 'drv_i', 'drv_u', 'n_gt', 'n_tp', 'n_pred', 'err_sum', 'plan_in_drv', 'ego_drv', 'plan_end', 'lag_n_gt', 'lag_n_match')


def row_key(r):
    return (r['si'], r['anchor'], r['fam'], r['val'], r['arm'], r['k'])


def union_dirs(dirs):
    """多个主目录取并集：同一行键只留第一个目录的行，重叠行逐行核对（数值字段），不同只告警。"""
    rows = load(dirs[0])
    idx = {row_key(r): r for r in rows}
    for d in dirs[1:]:
        n_new = n_cmp = n_bad = 0
        for r in load(d):
            b = idx.get(row_key(r))
            if b is None:
                idx[row_key(r)] = r
                rows.append(r)
                n_new += 1
            else:
                n_cmp += 1
                n_bad += any(b.get(f) != r.get(f) for f in NUM_FIELDS)
        print(f'[union] {d}：新增 {n_new} 行；与前面目录重叠的 {n_cmp} 行逐行核对，不同 {n_bad} 行' + ('  ⚠️ 两次运行结果不逐位一致' if n_bad else ''))
    return rows


def merge_extra(base, extra, name):
    """并入 extra 中行键不在 base 里、且（新臂 或 锚点已在 base）的行；两边都有的行键逐行核对（数值字段），不同只告警。"""
    base_arms = {r['arm'] for r in base}
    base_anchors = {(r['si'], r['anchor']) for r in base}
    idx = {row_key(r): r for r in base}
    n_cmp = n_bad = 0
    new = []
    for r in extra:
        b = idx.get(row_key(r))
        if b is not None:
            n_cmp += 1
            n_bad += any(b.get(f) != r.get(f) for f in NUM_FIELDS)
        elif r['arm'] not in base_arms or (r['si'], r['anchor']) in base_anchors:
            new.append(r)
    got = sorted({(r['arm'], r['fam']) for r in new})
    print(f'[extra] {name}：并入 {len(new)} 行（臂×族 {got}）；与主目录重叠的 {n_cmp} 行逐行核对，不同 {n_bad} 行' + ('  ⚠️ 两次运行结果不逐位一致' if n_bad else ''))
    return base + new


def onroad_table(rows):
    """(si, anchor, fam, val) → 窗口内每一步自车都在 GT 可行驶区域（各臂同一位姿，取任一臂的 ego_drv 即可）。"""
    ok = {}
    for r in rows:
        if r['arm'] == 'identity':
            continue
        key = (r['si'], r['anchor'], r['fam'], r['val'])
        ok[key] = ok.get(key, True) and bool(r['ego_drv'])
    return ok


def anchor_stats(rs):
    """一组行（同臂同扰动）→ {(si,anchor): (sums[len(COLS)], errs, lags)}。"""
    out = {}
    for r in rs:
        key = (r['si'], r['anchor'])
        if key not in out:
            out[key] = [np.zeros(len(COLS)), [], []]
        s = out[key][0]
        s[0] += r['lane_i']; s[1] += r['lane_u']; s[2] += r['drv_i']; s[3] += r['drv_u']
        s[4] += r['n_gt']; s[5] += r['n_tp']
        if r['plan_in_drv'] == r['plan_in_drv']:
            s[6] += r['plan_in_drv']; s[7] += 1
        if r['lag_n_gt'] >= 0:
            s[8] += r['lag_n_gt']; s[9] += r['lag_n_match']
            out[key][2] += r['lag_s']
        out[key][1] += r['err']
    return out


def metric_from(S, name):
    """S：[n_anchor 或 n_boot, len(COLS)] 的和（最后一维是列）。"""
    d = lambda a, b: np.where(b > 0, a / np.maximum(b, 1e-12), np.nan)
    c = lambda n: S[..., COLS.index(n)]
    return {'lane_iou': lambda: d(c('lane_i'), c('lane_u')), 'drv_iou': lambda: d(c('drv_i'), c('drv_u')),
            'recall': lambda: d(c('n_tp'), c('n_gt')), 'plan_in_drv': lambda: d(c('pin_s'), c('pin_n')),
            'lag_recall': lambda: d(c('lag_m'), c('lag_gt'))}[name]()


def stat(stats, keys, name, mask=None):
    """keys 里锚点聚合出的指标值。mask：只用其中的锚点（规划类的在路面锚点）。"""
    ks = [k for k in keys if k in stats and (mask is None or mask.get(k, False))]
    if not ks:
        return float('nan')
    if name in MEDIANS:
        v = np.concatenate([np.asarray(stats[k][1] if name == 'cerr_med' else stats[k][2], float) for k in ks]) if ks else np.array([])
        return float(np.median(v)) if len(v) else float('nan')
    return float(metric_from(np.sum([stats[k][0] for k in ks], 0), name))


def boot_ci(sa, sb, name, n_boot, rng, mask=None):
    """sa − sb 的 95% 区间；按共同锚点配对重采样。"""
    keys = sorted(set(sa) & set(sb))
    if mask is not None:
        keys = [k for k in keys if mask.get(k, False)]
    if len(keys) < 2 or n_boot <= 0:
        return (float('nan'), float('nan'))
    idx = rng.integers(0, len(keys), (n_boot, len(keys)))
    if name in MEDIANS:
        n_boot = min(n_boot, 500)
        sel = 1 if name == 'cerr_med' else 2
        la = [np.asarray(sa[k][sel], float) for k in keys]
        lb = [np.asarray(sb[k][sel], float) for k in keys]
        vals = []
        for b in range(n_boot):
            x = np.concatenate([la[i] for i in idx[b]])
            y = np.concatenate([lb[i] for i in idx[b]])
            if len(x) and len(y):
                vals.append(np.median(x) - np.median(y))
        vals = np.array(vals)
    else:
        A = np.stack([sa[k][0] for k in keys])
        B = np.stack([sb[k][0] for k in keys])
        vals = metric_from(A[idx].sum(1), name) - metric_from(B[idx].sum(1), name)
    vals = vals[np.isfinite(vals)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) else (float('nan'), float('nan'))


def fmt_ci(c):
    return f'[{c[0]:+.3f},{c[1]:+.3f}]' if np.isfinite(c[0]) else ''


FAM_ORDER = ('lat', 'speed', 'yaw')


def pert_name(fam, val):
    return {'lat': f'δ={val:+.1f} m', 'speed': f's={val:.1f}', 'yaw': f'ψ={val:+.1f}°'}[fam]


def is_3dgs(a):
    return a == '3dgs' or a.startswith('3dgs_')


def default_pairs(arms):
    pairs = []
    for a in arms:
        if '@' in a and a.split('@')[0] in arms:
            pairs.append((a.split('@')[0], a))
    preds = [a for a in arms if a not in ('real', 'copy', 'identity', 'warp') and '@' not in a and not is_3dgs(a)]
    pairs += [(p, 'copy') for p in preds if 'copy' in arms]
    full = [p for p in preds if 'nomap' not in p and any(q.replace('nomap', 'full') == p or q == p.replace('full', 'nomap') for q in preds if q != p)]
    for p in full:
        q = p.replace('full', 'nomap')
        if q in preds:
            pairs.append((p, q))
    for g in [a for a in arms if is_3dgs(a)]:
        pairs += [(p, g) for p in preds] + ([(g, 'copy')] if 'copy' in arms else [])
        if g != '3dgs' and '3dgs' in arms:
            pairs.append((g, '3dgs'))
    if 'warp' in arms:
        pairs += [(p, 'warp') for p in preds] + ([('warp', 'copy')] if 'copy' in arms else [])
        pairs += [(g, 'warp') for g in arms if is_3dgs(g)]
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', nargs='+', help='一个或多个主目录（取并集）')
    ap.add_argument('--kmin', type=int, default=1)
    ap.add_argument('--kmax', type=int, default=8)
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--pairs', default='', help='a:b,c:d；留空 = 自动（X vs X@mode、预测器 vs copy、full vs nomap）')
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--extra', action='append', default=[], help='另一个运行目录，只并入主目录没有的臂（可重复）')
    ap.add_argument('--out-dir', default='')
    args = ap.parse_args()
    out_dir = args.out_dir or args.run_dir[0]
    os.makedirs(out_dir, exist_ok=True)
    rows_all = union_dirs(args.run_dir)
    check_identity(rows_all)
    for ex in args.extra:
        ex_rows = load(ex)
        check_identity(ex_rows)                       # identity 只与**同一次运行**的 real 比（跨作业不保证逐位相同），比完不并入
        rows_all = merge_extra(rows_all, [r for r in ex_rows if r['arm'] != 'identity'], ex)
    rows = [r for r in rows_all if args.kmin <= r['k'] <= args.kmax and r['arm'] != 'identity']
    rng = np.random.default_rng(0)
    onroad = onroad_table(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r['arm'], r['fam'], r['val'])].append(r)
    stats = {g: anchor_stats(rs) for g, rs in groups.items()}
    arms = sorted({g[0] for g in groups}, key=lambda a: (a != 'real', a != 'copy', a))
    perts = sorted({(g[1], g[2]) for g in groups if g[0] != 'real'}, key=lambda p: (FAM_ORDER.index(p[0]), p[1]))
    real0 = stats[('real', 'lat', 0.0)]
    n_anchor = len(real0)
    tag = f'k{args.kmin}-{args.kmax}'

    def mask_of(fam, val):
        return {k: onroad[k + (fam, val)] for k in real0 if k + (fam, val) in onroad}

    # ---- 表 1：每臂每扰动的水平（Δ vs real δ=0）----
    table = []
    for arm in arms:
        if arm == 'real':
            continue
        for fam, val in perts:
            if (arm, fam, val) not in stats:
                continue
            sa, m = stats[(arm, fam, val)], mask_of(fam, val)
            keys = sorted(k for k in sa if k in real0)         # 与 real 的共同锚点
            ent = dict(arm=arm, fam=fam, val=val, n_anchor=len(keys), ego_on=float(np.mean([m.get(k, False) for k in keys])))
            for name in SHOW + ('cerr_med',):
                msk = m if name == 'plan_in_drv' else None
                ent[name] = stat(sa, keys, name, msk)
                ent['d_' + name] = ent[name] - stat(real0, keys, name, msk)
                if name in SHOW:
                    ent['ci_' + name] = boot_ci(sa, real0, name, args.boot, rng, msk)
            ent['n_lag'] = int(sum(len(sa[k][2]) for k in keys))
            table.append(ent)
    real_base = {name: stat(real0, sorted(real0), name, mask_of('lat', 0.0) if name == 'plan_in_drv' else None) for name in SHOW + ('cerr_med',)}

    # ---- 表 2：斜率（同臂 |δ| 相对 δ=0，两侧合并）----
    slope = []
    for arm in arms:
        if arm == 'real' or (arm, 'lat', 0.0) not in stats:
            continue
        ref = stats[(arm, 'lat', 0.0)]
        yaw_abs = sorted({abs(g[2]) for g in groups if g[0] == arm and g[1] == 'yaw' and g[2] != 0.0})
        for fam, ad in [('lat', x) for x in (0.5, 1.0, 2.0)] + [('yaw', x) for x in yaw_abs]:
            cur = {}
            for v in (-ad, ad):
                for k, s in stats.get((arm, fam, v), {}).items():
                    if k in cur:                                           # 两侧合并：可加和统计量相加，样本拼接
                        cur[k] = [cur[k][0] + s[0], cur[k][1] + s[1], cur[k][2] + s[2]]
                    else:
                        cur[k] = [s[0].copy(), list(s[1]), list(s[2])]
            if not cur:
                continue
            keys = sorted(set(cur) & set(ref))
            ent = dict(arm=arm, fam=fam, abs_d=ad, n_anchor=len(keys))
            for name in ('lane_iou', 'drv_iou', 'recall', 'lag_med', 'lag_recall'):
                ent[name] = dict(ref=stat(ref, keys, name), val=stat(cur, keys, name))
                ent[name]['delta'] = ent[name]['val'] - ent[name]['ref']
                ent[name]['ci'] = boot_ci(cur, ref, name, args.boot, rng)
            slope.append(ent)

    # ---- 表 3：臂间配对差 ----
    pairs = [tuple(p.split(':')) for p in args.pairs.split(',') if p] or default_pairs(arms)
    pair_tab = []
    for a, b in pairs:
        for fam, val in perts:
            if (a, fam, val) not in stats or (b, fam, val) not in stats:
                continue
            sa, sb, m = stats[(a, fam, val)], stats[(b, fam, val)], mask_of(fam, val)
            keys = sorted(set(sa) & set(sb))
            ent = dict(a=a, b=b, fam=fam, val=val, n_anchor=len(keys))
            for name in SHOW:
                msk = m if name == 'plan_in_drv' else None
                ent[name] = stat(sa, keys, name, msk) - stat(sb, keys, name, msk)
                ent['ci_' + name] = boot_ci(sa, sb, name, args.boot, rng, msk)
            pair_tab.append(ent)

    out = dict(run_dir=args.run_dir, extra=args.extra, k_range=[args.kmin, args.kmax], n_anchors=n_anchor, real=real_base, rows=table, slope=slope, pairs=pair_tab)
    with open(os.path.join(out_dir, f'summary_{tag}.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)

    L = [f'# B 表汇总（{" + ".join(args.run_dir)}{"；extra: " + " + ".join(args.extra) if args.extra else ""}，k∈[{args.kmin},{args.kmax}]，{n_anchor} 个锚点）', '',
         f"real（δ=0 真图）：lane IoU {real_base['lane_iou']:.3f} / drivable IoU {real_base['drv_iou']:.3f} / recall@2m {real_base['recall']:.3f} / "
         f"中心误差 {real_base['cerr_med']:.2f} m / plan∈drv {real_base['plan_in_drv']:.3f} / lag 中位 {real_base['lag_med']:+.3f} s / lag_recall {real_base['lag_recall']:.3f}", '',
         '规划类（plan∈drv）只统计窗口内每一步自车都在 GT 可行驶区域的锚点；Δ 对比同一批锚点上的 real（δ=0）。lag 正值 = 框落在车后面。', '',
         '## 1. 水平：各臂各扰动（Δ = 该臂 − real δ=0，括号为配对 bootstrap 95% CI）', '',
         '| 臂 | 扰动 | lane IoU | Δ | drivable IoU | Δ | recall@2m | Δ | plan∈drv | Δ | lag 中位 s | Δ | lag_recall | n_lag |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for e in table:
        L.append(f"| {e['arm']} | {pert_name(e['fam'], e['val'])} | {e['lane_iou']:.3f} | {e['d_lane_iou']:+.3f} {fmt_ci(e['ci_lane_iou'])} | "
                 f"{e['drv_iou']:.3f} | {e['d_drv_iou']:+.3f} {fmt_ci(e['ci_drv_iou'])} | {e['recall']:.3f} | {e['d_recall']:+.3f} {fmt_ci(e['ci_recall'])} | "
                 f"{e['plan_in_drv']:.3f} | {e['d_plan_in_drv']:+.3f} {fmt_ci(e['ci_plan_in_drv'])} | {e['lag_med']:+.3f} | "
                 f"{e['d_lag_med']:+.3f} {fmt_ci(e['ci_lag_med'])} | {e['lag_recall']:.3f} | {e['n_lag']} |")
    L += ['', '## 2. 斜率：同臂 |δ|（或 |ψ|）相对 δ=0（两侧合并；理想仿真器 = 0）', '',
          '| 臂 | \\|δ\\| / \\|ψ\\| | lane IoU：δ=0 → \\|δ\\|（Δ [CI]） | drivable IoU | recall@2m | lag 中位 s |', '|---|---|---|---|---|---|']

    def cell(e):
        return f"{e['ref']:.3f} → {e['val']:.3f}（{e['delta']:+.3f} {fmt_ci(e['ci'])}）"
    for e in slope:
        L.append(f"| {e['arm']} | {e['abs_d']}{' m' if e['fam'] == 'lat' else '°'} | {cell(e['lane_iou'])} | {cell(e['drv_iou'])} | {cell(e['recall'])} | {cell(e['lag_med'])} |")
    L += ['', '## 3. 臂间配对差（a − b，同扰动同锚点）', '',
          '| a | b | 扰动 | Δ lane IoU | Δ drivable IoU | Δ recall@2m | Δ plan∈drv | Δ lag 中位 s |', '|---|---|---|---|---|---|---|---|']
    for e in pair_tab:
        L.append(f"| {e['a']} | {e['b']} | {pert_name(e['fam'], e['val'])} | " + ' | '.join(
            f"{e[n]:+.3f} {fmt_ci(e['ci_' + n])}" for n in ('lane_iou', 'drv_iou', 'recall', 'plan_in_drv', 'lag_med')) + ' |')
    md = '\n'.join(L) + '\n'
    with open(os.path.join(out_dir, f'summary_{tag}.md'), 'w') as f:
        f.write(md)
    print(md)
    if not args.no_plot:
        plot(table, real_base, os.path.join(out_dir, f'b_vs_delta_{tag}.png'), n_anchor, args)


def plot(table, base, path, n_anchor, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    arms = sorted({e['arm'] for e in table}, key=lambda a: (a != 'copy', a))
    cmap = plt.get_cmap('tab10')
    fams = [f for f in FAM_ORDER if any(e['fam'] == f for e in table)] or ['lat']
    fig, axes = plt.subplots(len(fams), 4, figsize=(20, 4 * len(fams)), squeeze=False)
    for col, (m, lab) in enumerate((('lane_iou', 'lane IoU'), ('drv_iou', 'drivable IoU'), ('recall', 'recall@2m'), ('lag_med', 'lag 中位 (s)'))):
        for row, fam in enumerate(fams):
            ax = axes[row, col]
            ax.axhline(base[m], color='k', ls='--', lw=1, label='真图 δ=0')
            for i, arm in enumerate(arms):
                es = [e for e in table if e['arm'] == arm and e['fam'] == fam]
                if fam in ('speed', 'yaw'):
                    es += [e for e in table if e['arm'] == arm and e['fam'] == 'lat' and e['val'] == 0.0]
                x0 = 1.0 if fam == 'speed' else 0.0                       # lat δ=0 在速度图上是 s=1、在航向图上是 ψ=0
                xv = lambda e: x0 if (fam != 'lat' and e['fam'] == 'lat') else e['val']
                es = sorted(es, key=xv)
                xs = [xv(e) for e in es]
                if es:
                    ax.plot(xs, [e[m] for e in es], 'o-' if '@' not in arm else 'o:', color=cmap(i % 10), label=arm, ms=4)
            ax.set_xlabel({'lat': '横向偏移 δ (m)', 'speed': '速度倍率 s', 'yaw': '航向偏置 ψ (deg)'}[fam])
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f'B 表：偏离日志下的感知保真（{n_anchor} 个锚点，k∈[{args.kmin},{args.kmax}]）')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print('图：', path)


if __name__ == '__main__':
    main()
