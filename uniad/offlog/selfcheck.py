"""offlog 的 δ=0 自检（CPU 部分 + 可选 GPU 部分）。

    venv/bin/python -m offlog.selfcheck --scenes 0,1 [--gpu cuda:0 --ckpt hf]

CPU 检查
  A. 地图栅格：直接 import UniAD 官方 `vector_map.py` / `rasterize.py` / `data_utils.obtain_map_info` / 官方 map_api 副本
     （只读，打 shapely 2 必需的最小补丁），在日志位姿上与 offlog.map_gt 逐像素对比。
  B. GT 框：日志位姿下 offlog.gt_world 与 uniad_port 的 `_gt_boxes_cache.pkl`（gt_boxes.py 的产物）对比中心 / 尺寸 / yaw。
  C. img_metas：恒等扰动的 sim_frame 与 data.get_frame 逐项相同；另把同一日志位姿**强制走合成路径**（identity=False），
     检查 can_bus / l2g 的合成公式本身逐位相同。
GPU 检查（--gpu）
  D. 真图整段流式：data.get_frame 的 metas 与强制合成路径的 metas 各跑一遍，输出逐位相同。
  E. 真图 UniAD 输出对 offlog GT 的误差水平（lane / drivable IoU、recall@2m、中心误差），并把 lane/drivable 的 8 种
     行列翻转都算一遍，确认约定（IoU 应在 arr[row=y,col=x] 处最大）。
"""
import argparse
import importlib.util
import json
import os
import pickle
import sys
import types

import numpy as np

from . import common
from .gt_world import GTWorld
from .map_gt import MapGT
from .metas import OffLogEnv
from .sim_pose import LogTrack, SimPose

UNIAD_DS = os.path.join(common.UNIAD_BENCH, 'downloads', 'UniAD', 'projects', 'mmdet3d_plugin', 'datasets')  # 只读


# ---------------------------------------------------------------------------- 官方代码（只读加载 + shapely 2 补丁）
def _load_module(name, path, patch=None):
    src = open(path).read()
    if patch:
        for a, b in patch:
            assert a in src, (path, a)
            src = src.replace(a, b)
    mod = types.ModuleType(name)
    mod.__file__ = path
    exec(compile(src, path, 'exec'), mod.__dict__)
    sys.modules[name] = mod
    return mod


class OfficialMapGT:
    """官方 NuScenesE2EDataset.get_data_info 里地图 GT 的那几行，函数体逐字来自官方文件，仅替换 shapely 2 不兼容的语句。"""

    def __init__(self):
        from shapely.geometry import LinearRing
        # 官方 map_api 副本：mask_for_polygons 里 `for poly in polygons` -> `.geoms`
        self.map_api = _load_module('uniad_official_map_api', os.path.join(UNIAD_DS, 'eval_utils', 'map_api.py'), patch=[
            ('exteriors = [int_coords(poly.exterior.coords) for poly in polygons]',
             'exteriors = [int_coords(poly.exterior.coords) for poly in polygons.geoms]'),
            ('interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]',
             'interiors = [int_coords(pi.coords) for poly in polygons.geoms for pi in poly.interiors]'),
            # matplotlib 3.9 改名了这个样式（只影响绘图）
            ("plt.style.use('seaborn-whitegrid')", "pass"),
            # mask_for_lines（divider 栅格，GT 里不用但 obtain_map_info 会算）：迭代 MultiLineString -> .geoms
            ("        if lines.geom_type == 'MultiLineString':\n            for line in lines:",
             "        if lines.geom_type == 'MultiLineString':\n            for line in lines.geoms:")])
        # vector_map：LinearRing.coords 原地赋值 -> 新建（shapely 2 几何不可变）
        self.vm = _load_module('uniad_official_vector_map', os.path.join(UNIAD_DS, 'data_utils', 'vector_map.py'), patch=[
            ('                ext.coords = list(ext.coords)[::-1]',
             '                ext = LinearRing(list(ext.coords)[::-1])'),
            ('                inter.coords = list(inter.coords)[::-1]',
             '                inter = LinearRing(list(inter.coords)[::-1])'),
            ('from shapely.geometry import LineString, box, MultiPolygon, MultiLineString',
             'from shapely.geometry import LineString, box, MultiPolygon, MultiLineString, LinearRing')])
        # rasterize：迭代 MultiLineString -> .geoms
        self.rz = _load_module('uniad_official_rasterize', os.path.join(UNIAD_DS, 'data_utils', 'rasterize.py'), patch=[
            ('for new_single_line in new_line:', 'for new_single_line in new_line.geoms:')])
        # data_utils.obtain_map_info：整个文件 import 很多东西，只取这个函数的源码
        src = open(os.path.join(UNIAD_DS, 'data_utils', 'data_utils.py')).read()
        i0 = src.index('def obtain_map_info(')
        i1 = src.find('\ndef ', i0 + 10)
        i1 = len(src) if i1 < 0 else i1
        ns = {}
        exec('import math\nimport numpy as np\nfrom pyquaternion import Quaternion\n' + src[i0:i1], ns)
        self.obtain_map_info = ns['obtain_map_info']
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth', 'singapore-queenstown']
        self.nusc_maps = {m: self.map_api.NuScenesMap(dataroot=common.NUSC_MAP_ROOT, map_name=m) for m in self.MAPS}
        self.vector_map = self.vm.VectorizedLocalMap(common.NUSC_MAP_ROOT, patch_size=[102.4, 102.4], canvas_size=(200, 200))

    def raster(self, location, scene_token, info):
        import torch
        vectors = self.vector_map.gen_vectorized_samples(location, info['ego2global_translation'],
                                                         info['ego2global_rotation'])
        semantic_masks, instance_masks, forward_masks, backward_masks = self.rz.preprocess_map(
            vectors, [102.4, 102.4], (200, 200), 3, 2, 36)
        instance_masks = np.rot90(instance_masks, k=-1, axes=(1, 2))
        instance_masks = torch.tensor(instance_masks.copy())
        gt_labels, gt_masks = [], []
        for cls in range(3):
            for i in np.unique(instance_masks[cls]):
                if i == 0:
                    continue
                gt_labels.append(cls)
                gt_masks.append((instance_masks[cls] == i).to(torch.uint8))

        class _Nusc:   # obtain_map_info 只用 nusc.get('scene'/'log') 取 location
            def get(self, table, tok):
                return dict(log_token='L') if table == 'scene' else dict(location=location)
        map_mask = self.obtain_map_info(_Nusc(), self.nusc_maps, info, patch_size=[102.4, 102.4], canvas_size=(200, 200),
                                        layer_names=['lane_divider', 'road_divider'])
        map_mask = np.flip(map_mask, axis=1)
        map_mask = np.rot90(map_mask, k=-1, axes=(1, 2))
        map_mask = torch.tensor(map_mask.copy())
        for i, gt_mask in enumerate(map_mask[:-1]):
            gt_labels.append(i + 3)
            gt_masks.append(gt_mask)
        gt_labels = torch.tensor(gt_labels)
        gt_masks = torch.stack(gt_masks)
        # panseg_head.forward_test 的评测口径
        lanes_gt = (gt_masks[:-1].sum(0) > 0).int()
        drivable_gt = gt_masks[-1]
        out = dict(lane=lanes_gt.numpy() != 0, drivable=drivable_gt.numpy() != 0)
        for c, k in enumerate(('divider', 'crossing', 'contour')):
            sel = gt_masks[gt_labels == c]
            out[k] = (sel.sum(0) > 0).numpy() if len(sel) else np.zeros((200, 200), bool)
        return out


# ---------------------------------------------------------------------------- 指标
def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float('nan'), int((a & b).sum()), int(u)


def det_match(pred_boxes, pred_scores, pred_labels, gt, score_thr=0.4, max_range=40.0, gate=2.0):
    """车辆类 GT（≤max_range，num_lidar_pts≥1）与预测（车辆类、score≥thr）在 2 m 门限内做 Hungarian 配对。"""
    from scipy.optimize import linear_sum_assignment
    veh = set(common.VEHICLE_CLASSES)
    gsel = [i for i, n in enumerate(gt['det_name']) if n in veh and gt['num_lidar_pts'][i] >= 1
            and np.hypot(*gt['boxes'][i, :2]) <= max_range]
    gxy = gt['boxes'][gsel, :2].reshape(-1, 2)
    psel = [i for i in range(len(pred_scores)) if pred_scores[i] >= score_thr
            and common.CLASS_NAMES[int(pred_labels[i])] in veh]
    pxy = pred_boxes[psel, :2].reshape(-1, 2)
    res = dict(n_gt=len(gxy), n_pred=len(pxy), n_tp=0, err=[], dxy=[])
    if len(gxy) == 0 or len(pxy) == 0:
        return res
    d = np.linalg.norm(gxy[:, None] - pxy[None], axis=-1)
    r, c = linear_sum_assignment(np.where(d < gate, d, 1e6))
    ok = d[r, c] < gate
    res['n_tp'] = int(ok.sum())
    res['err'] = d[r, c][ok].tolist()
    res['dxy'] = (pxy[c[ok]] - gxy[r[ok]]).tolist()
    return res


# ---------------------------------------------------------------------------- 主程序
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='0,1')
    ap.add_argument('--nframes', type=int, default=0, help='每个场景最多检查多少帧（0=全部）')
    ap.add_argument('--gpu', default='')
    ap.add_argument('--ckpt', default='hf', help='hf | gh | 路径')
    ap.add_argument('--out', default=os.path.join(common.UNIAD_BENCH, 'offlog', 'logs', 'selfcheck.json'))
    args = ap.parse_args()
    from uniad_port.data import UniADValData
    data = UniADValData()
    scenes = [data.scenes[int(s)] for s in args.scenes.split(',')]
    gtw = GTWorld([i['token'] for i in data.infos], cache_path=os.path.join(common.CACHE_DIR, 'gt_world_val.pkl'))
    mg = MapGT()
    env = OffLogEnv(data, gtw, mg)
    off = OfficialMapGT()
    rep = dict(scenes=[s['scene_name'] for s in scenes])

    # ---------------- A. 地图栅格对拍
    n_px_diff, n_frames = {k: 0 for k in ('lane', 'drivable', 'divider', 'crossing', 'contour')}, 0
    for sc in scenes:
        ids = sc['frame_ids'][:args.nframes or None]
        for k in ids:
            info = data.infos[k]
            loc = env.loc[info['scene_token']]
            a = off.raster(loc, info['scene_token'], info)
            b = mg.raster(loc, info['ego2global_translation'], info['ego2global_rotation'],
                          info['lidar2ego_translation'], info['lidar2ego_rotation'])
            for key in n_px_diff:
                n_px_diff[key] += int((a[key] != b[key]).sum())
            n_frames += 1
    rep['A_map_px_diff'] = dict(n_frames=n_frames, **n_px_diff, n_geomcollection=mg.n_geomcollection)
    print('[A] 地图栅格 vs UniAD 官方代码：', rep['A_map_px_diff'], flush=True)

    # ---------------- B. GT 框对拍
    cache = os.path.join(common.UNIAD_BENCH, 'uniad_port', 'sanity', '_gt_boxes_cache.pkl')
    with open(cache, 'rb') as f:   # uniad_port 自己写的缓存（只读）
        old = pickle.load(f)
    maxd = dict(center=0.0, wlh=0.0, yaw=0.0)
    nb = 0
    for st, lst in old.items():
        if st.endswith('_pose') or st not in gtw.data:
            continue
        info = data.infos[data.token2idx[st]]
        g = gtw.boxes_in(st, info['ego2global_translation'], info['ego2global_rotation'],
                         info['lidar2ego_translation'], info['lidar2ego_rotation'])
        by_tok = {t: i for i, t in enumerate(gtw.data[st]['ann_token'])}
        for o in lst:
            i = by_tok[o['token']]
            bx = g['boxes'][i]
            c = np.array([bx[0], bx[1], bx[2] + bx[5] / 2])
            maxd['center'] = max(maxd['center'], float(np.abs(c - o['center']).max()))
            maxd['wlh'] = max(maxd['wlh'], float(np.abs(np.array([bx[4], bx[3], bx[5]]) - o['wlh']).max()))
            dy = (bx[6] - o['yaw'] + np.pi) % (2 * np.pi) - np.pi
            maxd['yaw'] = max(maxd['yaw'], float(abs(dy)))
            nb += 1
    rep['B_box_maxdiff'] = dict(n_boxes=nb, **maxd)
    print('[B] GT 框 vs gt_boxes.py 缓存：', rep['B_box_maxdiff'], flush=True)

    # ---------------- C. img_metas 对拍
    def same(a, b, path=''):
        if isinstance(a, dict):
            assert set(a) == set(b), (path, set(a) ^ set(b))
            return all(same(a[k], b[k], f'{path}.{k}') for k in a)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(same(x, y, f'{path}[{i}]') for i, (x, y) in enumerate(zip(a, b)))
        if isinstance(a, type):
            return a is b
        if isinstance(a, np.ndarray) or hasattr(a, 'numpy'):
            a2 = a.numpy() if hasattr(a, 'numpy') else a
            b2 = b.numpy() if hasattr(b, 'numpy') else b
            ok = a2.dtype == b2.dtype and a2.shape == b2.shape and np.array_equal(a2, b2)
            if not ok:
                print('  DIFF', path, np.abs(a2.astype(np.float64) - b2.astype(np.float64)).max())
            return ok
        ok = (a == b) or (a is b)
        if not ok:
            print('  DIFF', path, a, b)
        return ok

    n_id_ok = n_forced_ok = n_c = 0
    for sc in scenes:
        tr = LogTrack(data, sc)
        for a in tr.anchors(stride=4)[:3]:
            for p in tr.rollout(a, 'lat', 0.0) + tr.rollout(a, 'speed', 1.0):
                ref = data.get_frame(p.tmpl_idx, load_images=False)
                fr = env.sim_frame(p)
                fr.pop('pose')
                n_id_ok += same(ref, fr)
                p2 = SimPose(**{**p.__dict__, 'identity': False})
                fr2 = env.sim_frame(p2)
                for k in ('pose', 'sdc_planning', 'sdc_planning_mask'):
                    fr2.pop(k)
                    ref.pop(k, None)
                n_forced_ok += same({k: ref[k] for k in fr2}, fr2)
                n_c += 1
    rep['C_metas'] = dict(n=n_c, identity_equal=n_id_ok, forced_synth_equal=n_forced_ok)
    print('[C] img_metas：', rep['C_metas'], flush=True)

    # ---------------- F. 扰动的方向与幅度（符号错了 δ 曲线也会"看起来合理"，单独验）
    rep['F_perturb'] = check_perturb(data, env, scenes)
    print('[F] 扰动方向与幅度：', json.dumps(rep['F_perturb'], ensure_ascii=False), flush=True)

    if args.gpu:
        run_gpu_checks(args, data, env, scenes, rep)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print('写入', args.out)


def check_perturb(data, env, scenes):
    """δ>0 = 自车向左。ramp 结束后航向回到日志值，所以 LiDAR 系（x 右）里世界物体应整体 +δ 平移，栅格应向 +col 平移 δ/0.512 格。"""
    from pyquaternion import Quaternion
    from .sim_pose import actions
    res = dict(n=0, box_dx_med=[], box_dy_med=[], lane_best_col=[], lane_best_row=[], drv_best_col=[], drv_best_row=[],
               lat_err=[], act_dy_sum=[], speed_path_err=[])
    delta = 2.0
    for sc in scenes:
        tr = LogTrack(data, sc)
        for a in tr.anchors(stride=8):
            lg = tr.rollout(a, 'lat', 0.0)
            sm = tr.rollout(a, 'lat', delta)
            acts = actions(sm)
            res['act_dy_sum'].append(float(acts[:4, 1].sum()))
            for k in (4, 8):                      # ramp=4：k≥4 时偏移到位、Δψ=0
                g0, g1 = env.gt(lg[k]), env.gt(sm[k])
                d = g1['boxes'][:, :2] - g0['boxes'][:, :2]
                res['box_dx_med'].append(float(np.median(d[:, 0])))
                res['box_dy_med'].append(float(np.median(d[:, 1])))
                for key, kc, kr in (('lane', 'lane_best_col', 'lane_best_row'), ('drivable', 'drv_best_col', 'drv_best_row')):
                    m0, m1 = g0['map'][key], g1['map'][key]
                    best, arg = -1, (0, 0)
                    for dr in range(-3, 4):
                        for dc in range(-8, 9):
                            ov = (np.roll(np.roll(m0, dr, 0), dc, 1) & m1).sum()
                            if ov > best:
                                best, arg = ov, (dr, dc)
                    res[kr].append(arg[0])
                    res[kc].append(arg[1])
                # 与日志位姿的距离应为 |δ|，且方向在航向左侧
                yaw = tr.yaw[a + k]
                off = np.asarray(sm[k].e2g_t)[:2] - tr.T[a + k, :2]
                res['lat_err'].append(float(off @ np.array([-np.sin(yaw), np.cos(yaw)]) - delta))
                res['n'] += 1
            # 速度缩放：s=1.5 第 8 步应落在日志第 a+12 帧（整数下标逐位返回日志位姿）
            sp = tr.rollout(a, 'speed', 1.5)
            res['speed_path_err'].append(float(np.linalg.norm(np.asarray(sp[8].e2g_t)[:2] - tr.T[a + 12, :2])))
    out = dict(n=res['n'], delta=delta, expect_col=delta / 0.512)
    for k, v in res.items():
        if k != 'n':
            out[k] = dict(med=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v)))
    return out


def run_gpu_checks(args, data, env, scenes, rep):
    import torch
    from uniad_port.runner import UniADRunner
    ckpt = {'hf': common.CKPT_HF, 'gh': common.CKPT_GH}.get(args.ckpt, args.ckpt)
    runner = UniADRunner(ckpt, device=args.gpu, verbose=False)
    flips = {'row=y,col=x': lambda m: m, 'transpose': lambda m: m.T, 'flipud': lambda m: m[::-1], 'fliplr': lambda m: m[:, ::-1],
             'rot180': lambda m: m[::-1, ::-1], 'T_flipud': lambda m: m.T[::-1], 'T_fliplr': lambda m: m.T[:, ::-1],
             'T_rot180': lambda m: m.T[::-1, ::-1]}
    acc = {f'{k}_{h}': [0, 0] for k in ('lane', 'drivable') for h in flips}
    det = dict(n_gt=0, n_tp=0, n_pred=0, err=[])
    n_bit_ok = n_frames = 0
    plan_in = []
    for sc in scenes:
        runner.reset_scene()
        st_ref = None
        frames = list(data.iter_scene(sc['scene_name']))[:args.nframes or None]
        outs_ref = []
        for fr in frames:
            outs_ref.append(runner.step_frame_images(fr))
        runner.reset_scene()
        tr = LogTrack(data, sc)
        for j, fr in enumerate(frames):
            p = tr.rollout(j, 'lat', 0.0, horizon=0)[0]
            from .sim_pose import SimPose as _SP
            fr2 = env.sim_frame(_SP(**{**p.__dict__, 'identity': False}))
            fr2['img'] = fr['img']
            o = runner.step_frame_images(fr2)
            r = outs_ref[j]
            ok = all(np.array_equal(r[k], o[k]) for k in ('boxes', 'scores', 'plan')) and \
                np.array_equal(r['seg']['lane'], o['seg']['lane']) and np.array_equal(r['seg']['drivable'], o['seg']['drivable'])
            n_bit_ok += ok
            n_frames += 1
            g = env.gt(p)
            for key in ('lane', 'drivable'):
                for h, fn in flips.items():
                    _, i_, u_ = iou(fn(r['seg'][key] > 0.5), g['map'][key])
                    acc[f'{key}_{h}'][0] += i_
                    acc[f'{key}_{h}'][1] += u_
            m = det_match(r['boxes'], r['scores'], r['labels'], g)
            for k in ('n_gt', 'n_tp', 'n_pred'):
                det[k] += m[k]
            det['err'] += m['err']
            plan_in.append(plan_in_drivable(r['plan'], g['map']['drivable']))
    rep['D_real_bitwise'] = dict(n_frames=n_frames, equal=n_bit_ok)
    print('[D] 真图流式，原 metas vs 合成 metas 逐位相同：', rep['D_real_bitwise'], flush=True)
    rep['E_real_vs_gt'] = dict(
        lane_iou_pooled={h: acc[f'lane_{h}'][0] / max(acc[f'lane_{h}'][1], 1) for h in flips},
        drivable_iou_pooled={h: acc[f'drivable_{h}'][0] / max(acc[f'drivable_{h}'][1], 1) for h in flips},
        recall_2m=det['n_tp'] / max(det['n_gt'], 1), precision_2m=det['n_tp'] / max(det['n_pred'], 1),
        center_err_med=float(np.median(det['err'])) if det['err'] else float('nan'), n_gt=det['n_gt'],
        plan_in_drivable=float(np.nanmean(plan_in)))
    print('[E] 真图 UniAD 对 offlog GT：', json.dumps(rep['E_real_vs_gt'], indent=1, ensure_ascii=False), flush=True)


def plan_in_drivable(plan, drivable, half=51.2, res=0.512):
    """规划路点落在 GT drivable 栅格内的比例（plan: [6,2] LiDAR x 右 y 前）。"""
    col = np.floor((plan[:, 0] + half) / res).astype(int)
    row = np.floor((plan[:, 1] + half) / res).astype(int)
    ok = (row >= 0) & (row < drivable.shape[0]) & (col >= 0) & (col < drivable.shape[1])
    if not ok.any():
        return float('nan')
    return float(drivable[row[ok], col[ok]].mean())


if __name__ == '__main__':
    main()
