"""A 表跨目录配对差（refdet 帧集合，按场景成簇 bootstrap，口径与 a_pairs_preds.py / a_merge 相同）。
a_pairs_preds.py 只能比同一个 eval 目录里的臂；O4 / O5 / TF-full 各自一个目录，要比较就得先把行按（目标帧）对齐。
各目录都是 150 个分片、一个场景一片，行序相同；这里仍然逐 Δt 核对 target_token 完全相等，不等就报错。
用法：venv/bin/python -I a_table/a_pairs_dirs.py <label_a>=<dir>:<pred_a> <label_b>=<dir>:<pred_b> [更多对，成对比较 (1,2)(3,4)…]
例：  a_pairs_dirs.py f16=<dir>:pred:rof16_3000 f8=<dir>:pred:rof8cont_3000
"""
import featsim_paths as _release_paths
import sys
import numpy as np
B = str(_release_paths.CODE_ROOT)
sys.path.insert(0, B); sys.path.insert(0, B + '/a_table')
import a_merge as am
import aggregation as gm

specs = sys.argv[1:]
assert len(specs) >= 2 and len(specs) % 2 == 0, '参数必须成对：<label>=<dir>:<pred>'
cache = {}


def get(spec):
    label, rest = spec.split('=', 1)
    d, pred = rest.split(':', 1)
    if d not in cache:
        cache[d] = am.load_eval(d)
    arms, rec, meta, ref, info = cache[d]
    assert pred in arms, f'{pred} 不在 {d} 的臂里：{arms}'
    return label, rec[pred], meta, rec[arms[0]]


cols = (('plan_ade', -1), ('plan_fde', -1), ('det_iou', +1), ('drive_iou', +1))
rng = np.random.default_rng(0)
for i in range(0, len(specs), 2):
    la, ra, ma, ca = get(specs[i])
    lb, rb, mb, cb = get(specs[i + 1])
    print(f'\n== {la} − {lb}（refdet；差 = 前者 − 后者；★ = 95% CI 不含 0；✓/✗ = 均值是否对前者更好）')
    for k in am.KS:
        ta, tb = np.asarray(ma[k]['target_token']), np.asarray(mb[k]['target_token'])
        assert len(ta) == len(tb) and (ta == tb).all(), f'k={k}：两个目录的目标帧不对齐'
        sa, sb = np.asarray(ma[k]['scene']), np.asarray(mb[k]['scene'])
        assert (sa == sb).all()
        mask = ca[k]['det_n_ref'] > 0
        assert (mask == (cb[k]['det_n_ref'] > 0)).all()
        cells = []
        for m, sign in cols:
            d = np.where(mask, ra[k][m].astype(np.float64) - rb[k][m].astype(np.float64), np.nan)
            mu, lo, hi = gm.cluster_boot(d, sa, 1000, rng)
            star = '★' if (lo > 0 or hi < 0) else ' '
            cells.append(f'{m} {mu:+.3f} [{lo:+.3f},{hi:+.3f}]{star}{"✓" if sign * mu > 0 else "✗"}')
        print(f'  Δt={am.DT[k]}s n={int(mask.sum())}  ' + ' | '.join(cells))
