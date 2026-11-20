"""C 表闭环轨迹相对日志的偏离（论文表 3 / 表 S2 口径），traj_analysis/c_traj.py 的参数化版：任意运行目录、任意臂、任意配对。

ADE = 闭环开始后 (0, H] 内逐步位置偏差的均值；FDE = 最后一步的偏差；纵 / 横 = 投到日志航向的切向 / 法向分量。
配对差按场景 bootstrap（2000 次，种子 0）+ Wilcoxon 符号秩检验。同一 (场景, 臂) 出现在多个目录时取第一个目录的。
只读各目录 eval/ 下逐步的 jsonl，不改任何结果。

    venv/bin/python -I rev/c_traj_rev.py --dirs c_table/runs/full_20261107/eval <新作业>/eval \\
        --arms copy,rofull6000,3dgs,3dgs_difix --pairs rofull6000:3dgs,rofull6000:3dgs_difix [--scope 30=3dgs_difix] [--h 5,99]
--scope 名字=臂：在该臂覆盖的场景子集上再算一遍（缺省：全部有 log 的场景，加上 3dgs 覆盖的子集）。
"""
import argparse
import collections
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

NAMES = ['ADE', 'FDE', 'lon', 'lat', 'yaw_deg']


def load(dirs):
    rows = collections.defaultdict(dict)
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, 'scene*__*.jsonl'))):
            sc, arm = os.path.basename(f)[:-6].split('__')
            if (sc, arm) in rows:
                continue
            for line in open(f):
                r = json.loads(line)
                rows[(sc, arm)][r['j']] = r
    return rows


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def metr(rows, sc, arm, H):
    lg, ar = rows.get((sc, 'log')), rows.get((sc, arm))
    if not ar or not lg:
        return None
    js = sorted(j for j in ar if 0 < ar[j]['tau'] <= H + 1e-6 and j in lg)
    if not js:
        return None
    lon, lat, yw = [], [], []
    for j in js:
        dx, dy = ar[j]['x'] - lg[j]['x'], ar[j]['y'] - lg[j]['y']
        c, s = np.cos(lg[j]['yaw']), np.sin(lg[j]['yaw'])
        lon.append(c * dx + s * dy)
        lat.append(-s * dx + c * dy)
        yw.append(abs(wrap(ar[j]['yaw'] - lg[j]['yaw'])) * 180 / np.pi)
    lon, lat = np.array(lon), np.array(lat)
    d = np.hypot(lon, lat)
    return np.array([d.mean(), d[-1], np.abs(lon).mean(), np.abs(lat).mean(), np.mean(yw)])


def boot(x, n=2000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='+', required=True)
    ap.add_argument('--arms', required=True)
    ap.add_argument('--pairs', default='')
    ap.add_argument('--scope', action='append', default=[], help='名字=臂：再在该臂覆盖的场景子集上算')
    ap.add_argument('--h', default='5,99', help='时间窗（s），99 = 整段')
    ap.add_argument('--json', default='', help='另存逐场景值与配对差')
    a = ap.parse_args()
    rows = load(a.dirs)
    arms = [x for x in a.arms.split(',') if x]
    pairs = [tuple(p.split(':')) for p in a.pairs.split(',') if p]
    allsc = sorted({s for s, x in rows if x == 'log'})
    scopes = [(f'{len(allsc)}', allsc)]
    for sp in a.scope or (['3dgs=3dgs'] if any(x == '3dgs' for _, x in rows) else []):
        nm, arm = sp.split('=')
        sub = sorted(s for s, x in rows if x == arm and s in set(allsc))
        scopes.append((f'{nm} ({len(sub)} scenes covered by {arm})', sub))
    out = {}
    for H in (float(x) for x in a.h.split(',')):
        hn = f'{H:g} s' if H < 90 else 'full scene'
        for scope, scs in scopes:
            res = {}
            for arm in arms:
                m = {s: metr(rows, s, arm, H) for s in scs}
                m = {k: v for k, v in m.items() if v is not None}
                if len(m) == len(scs) and scs:
                    res[arm] = m
            if not res:
                continue
            print(f'\n## horizon {hn} | {scope}   mean (median) of ' + ' / '.join(NAMES))
            for arm, m in res.items():
                A = np.array([m[s] for s in scs])
                print(f'{arm:26s} ' + '  '.join(f'{x:6.2f} ({y:5.2f})' for x, y in zip(A.mean(0), np.median(A, 0))))
                out.setdefault(hn, {}).setdefault(scope, {})[arm] = {s: m[s].tolist() for s in scs}
            for pa, pb in pairs:
                if pa not in res or pb not in res:
                    continue
                for k in (0, 1, 2, 3):
                    dd = [res[pa][s][k] - res[pb][s][k] for s in scs]
                    lo, hi = boot(dd)
                    p = wilcoxon(dd).pvalue if np.any(dd) else 1.0
                    star = '*' if (lo > 0 or hi < 0) else ' '
                    print(f'   {pa}-{pb} {NAMES[k]:4s} {np.mean(dd):+6.2f} [{lo:+.2f},{hi:+.2f}]{star} p={p:.3g}')
                    out.setdefault('pairs', []).append(dict(h=hn, scope=scope, a=pa, b=pb, metric=NAMES[k], diff=float(np.mean(dd)),
                                                           lo=float(lo), hi=float(hi), p=float(p), n=len(scs)))
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
