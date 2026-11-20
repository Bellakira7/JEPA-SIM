"""检测器框 layout（审稿 W4：layout 用目标帧 GT 框属于特权信息）：把一个**独立**相机检测器在日志真图上的输出当成"其他 agent"的来源，
替换 nusc 索引里的 GT 框，其余（投影、filled 编码、配色、地图、画布几何）全部原样走 ``nusc_data.render_layout``。

检测器 = UniAD GitHub v1.0 发布权重（``downloads/uniad_base_e2e.pth``，'gh'），与被测栈（HF 2.0，'hf'）不是同一份权重；
沿日志真图流式跑一遍 val（与 A 表参照分支同一条加载 / 流式路径），每个关键帧取跟踪输出里 score ≥ 阈值的框。
这对应"无标注日志 → 离线相机检测 → 非反应式回放"的部署方式：没有激光标注、有漏检 / 误检 / 定位误差、只覆盖 BEV 范围（±51.2 m）。

缓存（``build`` 写，``load_det_index`` 读）：{'meta': {...}, 'frames': {sample_token: dict(boxes=[N,9] canonical LiDAR_TOP, scores, labels,
track_ids, l2g_r(3×3 float64), l2g_t(3,))}}。canonical box = runner._to_canonical 的 [x,y,z_bottom,l,w,h,yaw,vx,vy]。

几何（与 ``nusc_data.build_index`` 的 GT 框同一约定）：center_global = l2g_r.T @ [x, y, z_bottom + h/2] + l2g_t（行向量约定 p_g = p_l @ l2g_r + l2g_t），
R_global = l2g_r.T @ Rz(yaw)，wlh = [w, l, h]（nuScenes size 顺序）。l2g 由 info 的 float64 四元数现算（``UniADValData._transforms``），不用 float32 的 frame['l2g_*']。
类别：UniAD 10 类 → layout 配色下标（与 ``nusc_data._CAT2CLS`` 同一归并：truck / trailer / construction → 4，motorcycle / bicycle → 2，barrier → 6）。

    venv/bin/python feat/det_layout.py build --out <dir> --shard i --nshards n    # 每片一个 pkl
    venv/bin/python feat/det_layout.py merge --out <dir>                            # → <dir>/det_boxes.pkl + summary.json（含对 GT 的匹配统计）
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np

UNIAD_CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
LABEL2CLS = np.array([0, 4, 4, 3, 4, 6, 2, 2, 1, 5], np.int64)     # UniAD label → layout 配色下标
DEFAULT_THR = 0.3          # 与 agreement.AgreementConfig.score_thr 相同（全文"检出"的口径）
DET_CKPT = 'gh'


def rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def to_nd_boxes(rec, thr=DEFAULT_THR):
    """缓存里一帧的记录 → ``nusc_data`` 帧格式的框列表（global 系，center = 几何中心）。"""
    out = []
    R_l2g, t_l2g = np.asarray(rec['l2g_r'], np.float64), np.asarray(rec['l2g_t'], np.float64)
    for b, s, lab, tid in zip(rec['boxes'], rec['scores'], rec['labels'], rec['track_ids']):
        if s < thr:
            continue
        x, y, zb, l, w, h, yaw = (float(v) for v in b[:7])
        c = np.array([x, y, zb + h / 2.0]) @ R_l2g + t_l2g
        out.append(dict(center=c, wlh=np.array([w, l, h], np.float64), R=R_l2g.T @ rz(yaw), cls=int(LABEL2CLS[int(lab)]),
                        inst=f'det{int(tid)}', score=float(s)))
    return out


def load_det_index(path, base_idx, tokens=None, thr=DEFAULT_THR):
    """base_idx = ``nusc_data.build_index()``。返回新索引：frames[token] = dict(原帧, boxes=检测器框, box_src='det')；
    tokens 给了就要求每个 token 都在缓存里（漏帧直接报错，不静默回退到 GT）。scenes 也指向新帧对象。"""
    det = pickle.load(open(path, 'rb'))
    frames = det['frames']
    if tokens is not None:
        miss = [t for t in tokens if t not in frames]
        assert not miss, f'{path} 缺 {len(miss)} 个关键帧的检测（如 {miss[:3]}）'
    new = {}
    for tok, fr in base_idx['frames'].items():
        if tok in frames:
            assert not any('boxes' in c for c in fr['cams'].values()), f'{tok}: 关键帧不应有逐相机插值框'
            new[tok] = dict(fr, boxes=to_nd_boxes(frames[tok], thr), box_src='det')
    scenes = {name: [new.get(f['token'], f) for f in seq] for name, seq in base_idx['scenes'].items()}
    return dict(scenes=scenes, frames={**base_idx['frames'], **new}, det_meta=dict(det['meta'], thr=thr, path=path))


def match_stats(det_boxes, gt_boxes, ego2global, max_d=2.0, rng_m=50.0):
    """一帧的贪心匹配（global 中心距 < max_d，只算自车 rng_m 内的框）：返回 (n_gt, n_det, n_match, 匹配中心误差列表)。只作缓存健全性检查。"""
    g2e = np.linalg.inv(ego2global)
    near = lambda bs: [b for b in bs if np.hypot(*(g2e @ np.append(b['center'], 1.0))[:2]) <= rng_m]
    G, D = near(gt_boxes), near(det_boxes)
    if not G or not D:
        return len(G), len(D), 0, []
    d = np.linalg.norm(np.stack([g['center'][:2] for g in G])[:, None] - np.stack([x['center'][:2] for x in D])[None], axis=-1)
    errs, used_g, used_d = [], set(), set()
    for i, j in sorted(np.ndindex(d.shape), key=lambda ij: d[ij]):
        if d[i, j] >= max_d:
            break
        if i in used_g or j in used_d:
            continue
        used_g.add(i); used_d.add(j); errs.append(float(d[i, j]))
    return len(G), len(D), len(errs), errs


def build(a):
    root = os.environ.get('G3_CODE_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (os.path.join(root, 'uniad_port'), root, os.path.join(root, 'a_table')):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    from uniad_port.data import UniADValData
    from uniad_port.runner import UniADRunner
    from a_eval import CKPTS
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    os.makedirs(a.out, exist_ok=True)
    data = UniADValData(num_prefetch=4)
    scenes = list(range(len(data.scenes)))[a.shard::a.nshards]
    todo = [si for si in scenes if not os.path.exists(os.path.join(a.out, f'scene_{si:03d}.pkl'))]
    print(f'DET_BUILD shard {a.shard}/{a.nshards}: scenes {len(scenes)} todo {len(todo)} ckpt={a.ckpt}', flush=True)
    if not todo:
        return
    runner = UniADRunner(CKPTS.get(a.ckpt, a.ckpt), device='cuda:0', verbose=(a.shard == 0))
    for si in todo:
        t0 = time.time()
        runner.reset_scene()
        recs = {}
        for fr in data.iter_scene(si):
            out = runner.step_frame(runner.extract_feats(fr['img']), fr)
            info = data.infos[fr['info_idx']]
            l2e_r, l2e_t, e2g_r, e2g_t = data._transforms(info)
            recs[fr['sample_token']] = dict(boxes=out['boxes'].astype(np.float32), scores=out['scores'].astype(np.float32),
                                            labels=out['labels'].astype(np.int64), track_ids=out['track_ids'].astype(np.int64),
                                            l2g_r=l2e_r.T @ e2g_r.T, l2g_t=l2e_t @ e2g_r.T + e2g_t,
                                            scene=fr['scene_name'], frame_idx=int(fr['frame_idx']))
        path = os.path.join(a.out, f'scene_{si:03d}.pkl')
        with open(path + '.tmp', 'wb') as f:
            pickle.dump(dict(si=si, ckpt=a.ckpt, convention=runner.convention, frames=recs), f)
        os.replace(path + '.tmp', path)
        print(f'DET_SCENE_DONE {si} {len(recs)} frames {time.time() - t0:.0f}s', flush=True)


def merge(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import nusc_data as nd
    parts = sorted(glob.glob(os.path.join(a.out, 'scene_*.pkl')))
    assert len(parts) == a.n_scenes, f'只有 {len(parts)}/{a.n_scenes} 个场景'
    frames, conv, ck = {}, set(), set()
    for p in parts:
        r = pickle.load(open(p, 'rb'))
        frames.update(r['frames'])
        conv.add(r['convention']); ck.add(r['ckpt'])
    assert len(conv) == 1 and len(ck) == 1, (conv, ck)
    meta = dict(ckpt=ck.pop(), convention=conv.pop(), n_frames=len(frames), classes=UNIAD_CLASSES, label2cls=LABEL2CLS.tolist(),
                created=time.strftime('%F %T'))
    out = os.path.join(a.out, 'det_boxes.pkl')
    with open(out + '.tmp', 'wb') as f:
        pickle.dump(dict(meta=meta, frames=frames), f)
    os.replace(out + '.tmp', out)
    # 健全性：对 GT（nusc 索引里的框，与 layout 训练同一来源）贪心匹配，50 m 内
    idx = nd.build_index()
    ng = ndt = nm = 0
    errs, nbox = [], []
    for tok, rec in frames.items():
        db = to_nd_boxes(rec, a.thr)
        nbox.append(len(db))
        g, d, m, e = match_stats(db, idx['frames'][tok]['boxes'], idx['frames'][tok]['ego2global'])
        ng += g; ndt += d; nm += m; errs += e
    s = dict(meta, thr=a.thr, recall_50m_2m=nm / max(ng, 1), precision_50m_2m=nm / max(ndt, 1),
             center_err_median_m=float(np.median(errs)) if errs else None, boxes_per_frame_mean=float(np.mean(nbox)),
             n_gt_50m=ng, n_det_50m=ndt, n_match=nm)
    json.dump(s, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    print('DET_MERGE ' + json.dumps(s), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'merge'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', default=DET_CKPT)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--n-scenes', type=int, default=150)
    ap.add_argument('--thr', type=float, default=DEFAULT_THR)
    a = ap.parse_args()
    build(a) if a.cmd == 'build' else merge(a)


if __name__ == '__main__':
    main()
