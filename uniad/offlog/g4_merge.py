"""汇总 g4_probe 的分片 jsonl → "指标 vs δ / s" 表（markdown + json）和图。

    venv/bin/python -m offlog.g4_merge offlog/runs/g4_smoke_hf [--kmin 1 --kmax 8] [--boot 1000]

聚合口径
  * IoU：池化（Σ交 / Σ并，同 UniAD 官方 evaluate）；recall@2m = Σtp / Σgt（车辆类、≤40 m、num_lidar_pts≥1、预测 score≥0.4）；
    中心误差 = 配上的框的中位数；plan_in_drv = 规划路点落在 GT 可行驶区域的比例；dx_mean = 配上框的平均横向偏差（LiDAR x，右正）。
  * Δ 列 = 该臂 − 同一批锚帧、同一 k 上 real 臂（δ=0 真图）的值。
  * CI：按锚帧做配对 bootstrap（每个锚帧先在 k 上池化），给 Δ 的 95% 区间。
  * 默认 k ∈ [1, 8]；`--kmin 5` 只看偏移已经到位（ramp=4）之后的步。
  * **感知类**（lane/drivable IoU、recall、中心误差、dx）用全部锚帧。
  * **规划类**（plan∈drivable）主表只统计"自车在路面"的锚帧：该扰动下 [kmin,kmax] 每一步自车足迹都在 GT 可行驶区域
    （g4_probe 的 `ego_drv`）。自车位姿只取决于扰动，各臂用同一批锚帧；Δ 对比同一批锚帧上的 real 臂。
    每个扰动另报"自车在路面"的锚帧比例。全部锚帧的旧口径放在附表。
  * 产物缺 `ego_drv` 字段（11-03 第一轮冒烟）时退回全部锚帧，并在表头注明。
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

PERC_METRICS = ('lane_iou', 'drv_iou', 'recall', 'cerr_med', 'dx_mean')


def load(run_dir):
    """Load completed shards from an evaluation directory."""
    paths = sorted(glob.glob(os.path.join(run_dir, 'shard*.jsonl')))
    tmp = sorted(glob.glob(os.path.join(run_dir, 'shard*.jsonl.tmp')))
    done = sorted(glob.glob(os.path.join(run_dir, 'shard*.done.json')))
    if tmp or len(done) != len(paths):
        print(f'[warn] 分片不完整：完成 {len(paths)} 个（done.json {len(done)} 个），未完成 {len(tmp)} 个 {[os.path.basename(t) for t in tmp]}')
    nshard = {os.path.basename(p).split('of')[1].split('.')[0] for p in paths + tmp}
    if len(nshard) > 1:
        raise ValueError(f'同一目录混有不同分片数的产物：{sorted(nshard)}')
    for d in done:
        with open(d) as f:
            n_exp = json.load(f)['rows']
        p = d.replace('.done.json', '.jsonl')
        with open(p) as f:
            n = sum(1 for _ in f)
        if n != n_exp:
            raise ValueError(f'{p} 行数 {n} 与 done.json 记录的 {n_exp} 不符')
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(l) for l in f]
    print(f'[load] {run_dir}：{len(paths)} 个分片、{len(rows)} 行')
    return rows


def pool(rs):
    s = defaultdict(float)
    errs = []
    pin = []
    for r in rs:
        for k in ('lane_i', 'lane_u', 'drv_i', 'drv_u', 'n_gt', 'n_tp', 'n_pred', 'dx_sum'):
            s[k] += r[k]
        errs += r['err']
        if r['plan_in_drv'] == r['plan_in_drv']:
            pin.append(r['plan_in_drv'])
    nan = float('nan')
    return dict(lane_iou=s['lane_i'] / s['lane_u'] if s['lane_u'] else nan,
                drv_iou=s['drv_i'] / s['drv_u'] if s['drv_u'] else nan,
                recall=s['n_tp'] / s['n_gt'] if s['n_gt'] else nan,
                cerr_med=float(np.median(errs)) if errs else nan,
                plan_in_drv=float(np.mean(pin)) if pin else nan,
                dx_mean=s['dx_sum'] / s['n_tp'] if s['n_tp'] else nan,
                n_gt=int(s['n_gt']), n_rows=len(rs))


def boot_delta(by_anchor_arm, by_anchor_real, metric, n_boot, rng):
    keys = sorted(set(by_anchor_arm) & set(by_anchor_real))
    if len(keys) < 2 or n_boot <= 0:
        return (float('nan'), float('nan'))
    arm = [by_anchor_arm[k] for k in keys]
    real = [by_anchor_real[k] for k in keys]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        a = pool([r for i in idx for r in arm[i]])[metric]
        b = pool([r for i in idx for r in real[i]])[metric]
        vals.append(a - b)
    vals = np.array(vals)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return (float('nan'), float('nan'))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def onroad_table(rows):
    """(si, anchor, fam, val) -> 窗口内每一步自车都在 GT 可行驶区域。real 臂的键与 lat δ=0 相同（同一位姿）。"""
    ok = {}
    for r in rows:
        key = (r['si'], r['anchor'], r['fam'], r['val'])
        ok[key] = ok.get(key, True) and bool(r['ego_drv'])
    return ok


def by_anchor(rs):
    d = defaultdict(list)
    for r in rs:
        d[(r['si'], r['anchor'])].append(r)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--kmin', type=int, default=1)
    ap.add_argument('--kmax', type=int, default=8)
    ap.add_argument('--boot', type=int, default=500)
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--out-dir', default='', help='汇总写到哪（默认 = run_dir，会覆盖同名 summary_*）')
    args = ap.parse_args()
    out_dir = args.out_dir or args.run_dir
    os.makedirs(out_dir, exist_ok=True)
    rows = [r for r in load(args.run_dir) if args.kmin <= r['k'] <= args.kmax]
    rng = np.random.default_rng(0)
    has_ego = bool(rows) and all('ego_drv' in r for r in rows)
    if not has_ego:
        print('[warn] 产物缺 ego_drv 字段：规划类指标退回全部锚帧口径')
    onroad = onroad_table(rows) if has_ego else defaultdict(lambda: True)
    real = [r for r in rows if r['arm'] == 'real']
    real_by_anchor = by_anchor(real)
    base = pool(real)
    real_on_keys = {k for k in real_by_anchor if onroad[k + ('lat', 0.0)]}
    base_plan = dict(plan_in_drv=pool([r for k in real_on_keys for r in real_by_anchor[k]])['plan_in_drv'],
                     ego_on_anchor_frac=len(real_on_keys) / max(len(real_by_anchor), 1), n_anchor=len(real_on_keys))
    groups = defaultdict(list)
    for r in rows:
        if r['arm'] != 'real':
            groups[(r['arm'], r['fam'], r['val'])].append(r)
    table = []
    for (arm, fam, val), rs in sorted(groups.items()):
        p = pool(rs)
        ba = by_anchor(rs)
        ent = dict(arm=arm, fam=fam, val=val, **{m: p[m] for m in PERC_METRICS}, n_gt=p['n_gt'], n_rows=p['n_rows'],
                   n_anchor=len(ba))
        for m in PERC_METRICS:
            ent[f'd_{m}'] = p[m] - base[m]
        for m in ('lane_iou', 'drv_iou', 'recall'):
            ent[f'ci_{m}'] = boot_delta(ba, real_by_anchor, m, args.boot, rng)
        # 规划类：主口径 = 自车在路面的锚帧（与 real 臂同一批锚帧配对）
        on = {k: v for k, v in ba.items() if onroad[k + (fam, val)]}
        real_on = {k: real_by_anchor[k] for k in on if k in real_by_anchor}
        ent['ego_on_anchor_frac'] = len(on) / max(len(ba), 1) if has_ego else float('nan')
        ent['ego_on_frame_frac'] = float(np.mean([r['ego_drv'] for r in rs])) if has_ego else float('nan')
        ent['n_anchor_plan'] = len(on)
        ent['plan_in_drv'] = pool([r for v in on.values() for r in v])['plan_in_drv']
        ent['d_plan_in_drv'] = ent['plan_in_drv'] - pool([r for v in real_on.values() for r in v])['plan_in_drv']
        ent['ci_plan_in_drv'] = boot_delta(on, real_on, 'plan_in_drv', args.boot, rng)
        # 附表：全部锚帧（旧口径）
        ent['all_anchors'] = dict(plan_in_drv=p['plan_in_drv'], d_plan_in_drv=p['plan_in_drv'] - base['plan_in_drv'],
                                  ci_plan_in_drv=boot_delta(ba, real_by_anchor, 'plan_in_drv', args.boot, rng))
        table.append(ent)
    n_anchor = len(real_by_anchor)
    # G4 判据：同一臂 δ 相对 δ=0 的配对差（不是相对真图）；|δ|=2 的两侧合并
    g4 = {}
    for arm in ('copy', 'copy_logpose', 'zero'):
        ref_arm = 'copy' if arm == 'copy_logpose' else arm   # copy_logpose 在 δ=0 时与 copy 相同，没有单独记录
        ref = defaultdict(list)
        for r in groups.get((ref_arm, 'lat', 0.0), []):
            ref[(r['si'], r['anchor'])].append(r)
        for abs_d in (0.5, 1.0, 2.0):
            cur = defaultdict(list)
            for v in (-abs_d, abs_d):
                for r in groups.get((arm, 'lat', v), []):
                    cur[(r['si'], r['anchor'])].append(r)
            if not cur or not ref:
                continue
            # ref 每个锚帧的行数只有 cur 的一半，池化指标（比例）不受影响
            ent = {}
            for m in ('lane_iou', 'drv_iou', 'recall'):
                pc = pool([r for rs in cur.values() for r in rs])[m]
                pr = pool([r for rs in ref.values() for r in rs])[m]
                ent[m] = dict(val=pc, ref=pr, delta=pc - pr, ci=boot_delta(cur, ref, m, args.boot, rng))
            g4[f'{arm}|abs_d={abs_d}'] = ent
    # 按 k 展开（真图、copy δ=0、copy |δ|=2）
    by_k = {}
    for name, sel in (('real', lambda r: r['arm'] == 'real'),
                      ('copy δ=0', lambda r: r['arm'] == 'copy' and r['fam'] == 'lat' and r['val'] == 0.0),
                      ('copy |δ|=2', lambda r: r['arm'] == 'copy' and r['fam'] == 'lat' and abs(r['val']) == 2.0)):
        by_k[name] = {k: pool([r for r in rows if r['k'] == k and sel(r)]) for k in range(args.kmin, args.kmax + 1)}
    out = dict(run_dir=args.run_dir, k_range=[args.kmin, args.kmax], n_anchors=n_anchor, real=base, real_plan=base_plan,
               plan_scope=('anchors whose ego footprint is on GT drivable at every k in the window' if has_ego
                           else 'all anchors (ego_drv missing)'),
               rows=table, g4=g4, by_k=by_k)
    tag = f'k{args.kmin}-{args.kmax}'
    with open(os.path.join(out_dir, f'summary_{tag}.json'), 'w') as f:
        json.dump(out, f, indent=1)

    lines = [f'# G4 探针汇总（{args.run_dir}，k∈[{args.kmin},{args.kmax}]，{n_anchor} 个锚帧）', '',
             f"real（δ=0 真图）：lane IoU {base['lane_iou']:.3f} / drivable IoU {base['drv_iou']:.3f} / "
             f"recall@2m {base['recall']:.3f} / 中心误差 {base['cerr_med']:.2f} m（n_gt={base['n_gt']}）；"
             + (f"plan∈drivable {base_plan['plan_in_drv']:.3f}（自车在路面的 {base_plan['n_anchor']} 个锚帧，"
              f"占 {base_plan['ego_on_anchor_frac']:.2f}）" if has_ego else
              f"plan∈drivable {base['plan_in_drv']:.3f}（全部 {n_anchor} 个锚帧）"), '',
             ('感知类指标用全部锚帧；规划类（plan∈drv）只统计自车在 GT 可行驶区域内的锚帧（窗口内每一步），Δ 对比同一批锚帧上的真图；'
              '全部锚帧的规划指标见附表。' if has_ego else
              '⚠️ 产物缺 ego_drv 字段，规划类指标是全部锚帧口径，"自车在路面"列为空。'), '',
             '| 臂 | 扰动 | lane IoU | Δ [95% CI] | drivable IoU | Δ [95% CI] | recall@2m | Δ [95% CI] | 中心误差 m | dx 均值 m '
             '| 自车在路面（锚帧比例） | plan∈drv（在路面锚帧） | Δ [95% CI] |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']

    def ci(c):
        return f'[{c[0]:+.3f},{c[1]:+.3f}]' if np.isfinite(c[0]) else ''
    for e in table:
        pert = f"δ={e['val']:+.1f} m" if e['fam'] == 'lat' else f"s={e['val']:.1f}"
        lines.append(f"| {e['arm']} | {pert} | {e['lane_iou']:.3f} | {e['d_lane_iou']:+.3f} {ci(e['ci_lane_iou'])} | "
                     f"{e['drv_iou']:.3f} | {e['d_drv_iou']:+.3f} {ci(e['ci_drv_iou'])} | {e['recall']:.3f} | "
                     f"{e['d_recall']:+.3f} {ci(e['ci_recall'])} | {e['cerr_med']:.2f} | {e['dx_mean']:+.2f} | "
                     f"{e['ego_on_anchor_frac']:.2f}（{e['n_anchor_plan']}/{e['n_anchor']}） | {e['plan_in_drv']:.3f} | "
                     f"{e['d_plan_in_drv']:+.3f} {ci(e['ci_plan_in_drv'])} |")
    lines += ['', '## G4 判据：同臂 |δ| 相对 δ=0 的配对差（按锚帧 bootstrap）', '',
              '| 臂 | \\|δ\\| | lane IoU：δ=0 → \\|δ\\|（Δ [95% CI]） | drivable IoU | recall@2m |', '|---|---|---|---|---|']

    def cell(e):
        return f"{e['ref']:.3f} → {e['val']:.3f}（{e['delta']:+.3f} {ci(e['ci'])}）"
    for key, ent in g4.items():
        arm, d = key.split('|abs_d=')
        lines.append(f"| {arm} | {d} | {cell(ent['lane_iou'])} | {cell(ent['drv_iou'])} | {cell(ent['recall'])} |")
    lines += ['', '## 附表：规划类指标，全部锚帧（含自车出路面的锚帧，旧口径）', '',
              f"real：plan∈drv {base['plan_in_drv']:.3f}（{n_anchor} 个锚帧）", '',
              '| 臂 | 扰动 | plan∈drv（全部锚帧） | Δ vs 真图 [95% CI] | 自车在路面（逐帧比例） |', '|---|---|---|---|---|']
    for e in table:
        pert = f"δ={e['val']:+.1f} m" if e['fam'] == 'lat' else f"s={e['val']:.1f}"
        a = e['all_anchors']
        lines.append(f"| {e['arm']} | {pert} | {a['plan_in_drv']:.3f} | {a['d_plan_in_drv']:+.3f} {ci(a['ci_plan_in_drv'])} | "
                     f"{e['ego_on_frame_frac']:.2f} |")
    lines += ['', '## 按 k 展开', '', '| k（0.5 s/步） | ' + ' | '.join(f'{n} lane / drv / recall' for n in by_k) + ' |',
              '|---|' + '---|' * len(by_k)]
    for k in range(args.kmin, args.kmax + 1):
        lines.append(f'| {k} | ' + ' | '.join(
            f"{v[k]['lane_iou']:.3f} / {v[k]['drv_iou']:.3f} / {v[k]['recall']:.3f}" for v in by_k.values()) + ' |')
    md = '\n'.join(lines) + '\n'
    with open(os.path.join(out_dir, f'summary_{tag}.md'), 'w') as f:
        f.write(md)
    print(md)

    if not args.no_plot:
        plot(table, base, os.path.join(out_dir, f'g4_vs_delta_{tag}.png'), n_anchor, args)


def plot(table, base, path, n_anchor, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    colors = {'copy': '#1f77b4', 'copy_logpose': '#ff7f0e', 'zero': '#7f7f7f'}
    names = {'copy': 'copy-last（仿真位姿 metas）', 'copy_logpose': 'copy-last（日志位姿 metas）', 'zero': '置零特征'}
    try:
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, (m, lab) in enumerate((('lane_iou', 'lane IoU'), ('drv_iou', 'drivable IoU'), ('recall', 'recall@2m'))):
        for row, fam in enumerate(('lat', 'speed')):
            ax = axes[row, col]
            ax.axhline(base[m], color='k', ls='--', lw=1, label='真图 δ=0')
            for arm in ('copy', 'copy_logpose', 'zero'):
                es = sorted([e for e in table if e['arm'] == arm and e['fam'] == fam], key=lambda e: e['val'])
                if fam == 'speed':
                    # s=1 等价于 lat δ=0
                    es0 = [e for e in table if e['arm'] == arm and e['fam'] == 'lat' and e['val'] == 0.0]
                    es = sorted(es + es0, key=lambda e: (1.0 if e['fam'] == 'lat' else e['val']))
                    xs = [1.0 if e['fam'] == 'lat' else e['val'] for e in es]
                else:
                    xs = [e['val'] for e in es]
                if es:
                    ax.plot(xs, [e[m] for e in es], 'o-', color=colors[arm], label=names[arm])
            ax.set_xlabel('横向偏移 δ (m)' if fam == 'lat' else '速度倍率 s')
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f'G4：冻结特征在偏离日志下的保真度（{n_anchor} 个锚帧，k∈[{args.kmin},{args.kmax}]）')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print('图：', path)


if __name__ == '__main__':
    main()
