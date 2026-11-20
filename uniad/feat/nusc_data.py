"""nuScenes 关键帧表解析（不依赖 nuscenes-devkit / pyquaternion）+ GT 框 layout 渲染。


  - 6 路相机：图像路径、内参 K、cam→ego、ego(相机时刻)→global；
  - LIDAR_TOP 时刻的 ego→global（动作/UniAD 参考系用）；
  - 目标框（global 系：center / size(w,l,h) / rotation / 类别）。

layout 约定同主线：layout[t] 画的是**目标帧 t+1**（调用方负责错位），filled 编码，按深度远→近，类别上色。
"""
import featsim_paths as _release_paths
import json
import os
import pickle
import sys

import cv2
import numpy as np

ROOT = _release_paths.NUSC_ROOT
CAMS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
CACHE = str(_release_paths.Path(_release_paths.BENCH) / 'feat/nusc_index.pkl')

# nuScenes category → 主线 layout 配色表下标（0 car,1 ped,2 cyc,3 bus,4 truck,5 cone,6 tricycle(白),8 dontcare）
_CAT2CLS = {
    'vehicle.car': 0, 'vehicle.emergency.police': 0, 'vehicle.emergency.ambulance': 4,
    'vehicle.bus.bendy': 3, 'vehicle.bus.rigid': 3,
    'vehicle.truck': 4, 'vehicle.trailer': 4, 'vehicle.construction': 4,
    'vehicle.bicycle': 2, 'vehicle.motorcycle': 2,
    'movable_object.trafficcone': 5, 'movable_object.barrier': 6,
}
_BOX_COLOR_BGR = {0: (0, 140, 255), 1: (0, 225, 0), 2: (255, 0, 0), 3: (0, 255, 255), 4: (255, 0, 255),
                  5: (255, 255, 0), 6: (255, 255, 255), 8: (0, 69, 255)}


def cat_to_cls(name):
    if name.startswith('human.pedestrian'):
        return 1
    return _CAT2CLS.get(name)  # 其它（动物、debris、rack 等）返回 None → 丢弃


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def build_index(force=False):
    """返回 {'scenes': {name: [frame,...]}, 'frames': {sample_token: frame}}；缓存到 pkl。"""
    if os.path.exists(CACHE) and not force:
        return pickle.load(open(CACHE, 'rb'))
    d = f'{ROOT}/v1.0-trainval'
    ld = lambda n: json.load(open(f'{d}/{n}.json'))
    scene, sample, sd = ld('scene'), ld('sample'), ld('sample_data')
    ego_pose = {x['token']: x for x in ld('ego_pose')}
    calib = {x['token']: x for x in ld('calibrated_sensor')}
    sensor = {x['token']: x['channel'] for x in ld('sensor')}
    cat = {x['token']: x['name'] for x in ld('category')}
    inst = {x['token']: x for x in ld('instance')}
    ann = ld('sample_annotation')
    ann_by_sample = {}
    for a in ann:
        c = cat_to_cls(cat[inst[a['instance_token']]['category_token']])
        if c is None or (a['num_lidar_pts'] + a['num_radar_pts']) <= 0:
            continue
        ann_by_sample.setdefault(a['sample_token'], []).append(dict(
            center=np.array(a['translation'], np.float64), wlh=np.array(a['size'], np.float64),
            R=quat_to_R(a['rotation']), cls=c, inst=a['instance_token']))
    sd_by_sample = {}
    for x in sd:
        if x['is_key_frame']:
            sd_by_sample.setdefault(x['sample_token'], {})[sensor[calib[x['calibrated_sensor_token']]['sensor_token']]] = x
    smp = {x['token']: x for x in sample}
    frames, scenes = {}, {}
    for sc in scene:
        seq, tok = [], sc['first_sample_token']
        while tok:
            s = smp[tok]
            cams = {}
            for ch in CAMS:
                x = sd_by_sample[tok][ch]
                cs, ep = calib[x['calibrated_sensor_token']], ego_pose[x['ego_pose_token']]
                cams[ch] = dict(path=f"{ROOT}/{x['filename']}", K=np.array(cs['camera_intrinsic'], np.float64),
                                cam2ego=make_T(quat_to_R(cs['rotation']), cs['translation']),
                                ego2global=make_T(quat_to_R(ep['rotation']), ep['translation']))
            lx = sd_by_sample[tok]['LIDAR_TOP']
            lep = ego_pose[lx['ego_pose_token']]
            fr = dict(token=tok, scene=sc['name'], idx=len(seq), timestamp=s['timestamp'], cams=cams,
                      ego2global=make_T(quat_to_R(lep['rotation']), lep['translation']),
                      boxes=ann_by_sample.get(tok, []))
            frames[tok] = fr
            seq.append(fr)
            tok = s['next']
        scenes[sc['name']] = seq
    out = dict(scenes=scenes, frames=frames)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    pickle.dump(out, open(CACHE, 'wb'))
    return out


def splits():
    from nuscenes.utils.splits import create_splits_scenes
    return create_splits_scenes()


def boxes_in_ego(fr, ego2global):
    """帧 fr 的 GT 框变到给定 ego2global（某相机时刻的自车系）→ ([N,7] x,y,z,l,w,h,yaw, [N] cls)"""
    if not fr['boxes']:
        return np.zeros((0, 7)), np.zeros((0,), np.int64)
    g2e = np.linalg.inv(ego2global)
    out, cls = [], []
    for b in fr['boxes']:
        c = (g2e @ np.append(b['center'], 1.0))[:3]
        R = g2e[:3, :3] @ b['R']
        yaw = np.arctan2(R[1, 0], R[0, 0])
        w, l, h = b['wlh']
        out.append([c[0], c[1], c[2], l, w, h, yaw])
        cls.append(b['cls'])
    return np.array(out), np.array(cls)


# layout 画布与 PyramidPredictor 的 token 网格对齐：1.0× 原图 900×1600 → 补到 token 整数倍 960×1600（15×25 token，每 token 64×64 px），
# 再按 P3 的 stride 8 缩到 120×200 = 15×25 token × pixel_unshuffle 因子 8（CamLayoutAdapter 的 cam_grid，进 adapter 时不再池化）。
LAYOUT_STRIDE = 8
LAYOUT_HW = (120, 200)
IMG_HW = (900, 1600)


MAP_LINE_THICKNESS = 1  # 画布分辨率下的线宽（画布 1 格 = 原图 8 px = P3 的 1 格；LINE_AA 的灰度携带格内位置）
# 地图折线类型 → 主线 render_lane_wireframe 的配色码：divider(0) → lane 码 0，ped_crossing(1) → lane 码 1，contour(2) → border 码 2
_MAP_LANE_CODE = {0: 0, 1: 1}
_MAP_BORDER_CODE = 2


def _draw_map(canvas, c, map_vec, K, imu2cam, hw):
    """把矢量地图画进 canvas（原地）。map_vec = (E_k, [(type, pts[n,2] 关键帧 LIDAR 自车系), ...])，c = 该相机的标定/位姿 dict。
    折线点（z=0）→ global（E_k）→ 该相机时刻的自车系（inv(ego2global_cam)），再走与框相同的 imu2cam / K 投影。"""
    from jepa_layout_renderer import render_lane_wireframe
    E_k, polys = map_vec
    if not polys:
        return
    M = np.linalg.inv(c['ego2global']) @ E_k
    for typ, p in polys:
        q = np.empty((len(p), 3))
        q[:, :2] = p
        q[:, 2] = 0.0
        q = q @ M[:3, :3].T + M[:3, 3]
        if typ == 2:
            render_lane_wireframe(None, None, q[None], [_MAP_BORDER_CODE], K, imu2cam, hw, thickness=MAP_LINE_THICKNESS, canvas=canvas)
        else:
            render_lane_wireframe(q[None], [_MAP_LANE_CODE[typ]], None, None, K, imu2cam, hw, thickness=MAP_LINE_THICKNESS, canvas=canvas)


def render_layout(fr, out_hw=LAYOUT_HW, stride=LAYOUT_STRIDE, map_vec=None):
    """目标帧 fr 的 6 路 filled layout，uint8 [6,3,H,W]（BGR 配色表同主线）。
    像素 (u,v)（1.0× 原图）→ 画布 (u/stride, v/stride)，画布左上角对齐原图左上角，与 FPN P3 栅格（stride 8）同一几何；
    原图以外的补边行（v ≥ 900，原图本身是 0）置黑，不画看不见的东西。

    ``map_vec`` 给了就先画地图折线、再叠框（与主线 ``_render_layout`` 的顺序一致：车道线先画，框后叠在上面），这是"完整条件"变体；
    ``None`` = 只画框（"不画地图"变体；输出与加这个参数之前逐位相同）。框取 ``fr['cams'][ch]['boxes']``（相位增广帧：按该相机 sweep 时刻插值），
    没有就用 ``fr['boxes']``（关键帧）。"""
    _deps = os.environ.get('JEPA_DEPS_DIR') or _release_paths.DEPS
    if _deps not in sys.path:  # 每个样本都会调用：只插一次，否则训练几万步后 sys.path 里有几十万个重复项
        sys.path.insert(0, _deps)
    from jepa_layout_renderer import render_filled
    H, W = out_hw
    valid_rows = int(np.ceil(IMG_HW[0] / stride))
    imgs = []
    for ch in CAMS:
        c = fr['cams'][ch]
        boxes, cls = boxes_in_ego(fr if 'boxes' not in c else dict(fr, boxes=c['boxes']), c['ego2global'])
        K = c['K'].copy()
        K[:2] /= stride
        colors = np.array([_BOX_COLOR_BGR[int(k)] for k in cls], np.uint8).reshape(-1, 3)
        imu2cam = np.linalg.inv(c['cam2ego'])
        canvas = None
        if map_vec is not None:
            canvas = np.zeros((H, W, 3), np.uint8)
            _draw_map(canvas, c, map_vec, K, imu2cam, (H, W))
        im = render_filled(boxes, K, imu2cam, (H, W), color=colors if len(colors) else (255, 255, 255), canvas=canvas)
        im[valid_rows:] = 0
        imgs.append(im)
    return np.stack(imgs).transpose(0, 3, 1, 2)  # [6,3,H,W]


def _cam_boxes(fr, ch):
    """该相机时刻的框：相位增广帧用 ``cams[ch]['boxes']``（按该相机 sweep 时刻插值），关键帧用 ``fr['boxes']``（与 render_layout 同口径）。"""
    c = fr['cams'][ch]
    return c['boxes'] if 'boxes' in c else fr['boxes']


def assoc_frame(fr_prev, fr_tgt):
    """关联通道的合成帧：相机几何取**目标帧** fr_tgt，框取**上下文末帧** fr_prev 的、且同一实例（``inst``）在 fr_tgt 同一相机时刻也在场的那些。
    对应 FlowSim 逐 agent 描述 [s_t, s_{t-1}, Δs, valid] 在相机空间的版本：与主通道（目标帧框）同一相机、同一画布，两组框的位移即关联 + 速度；
    valid = 两帧都在场（只在一帧出现的实例不画：消失的不给假运动线索，新出现的只在主通道里）。"""
    cams = {}
    for ch in CAMS:
        keep = {b['inst'] for b in _cam_boxes(fr_tgt, ch)}
        cams[ch] = dict(fr_tgt['cams'][ch], boxes=[b for b in _cam_boxes(fr_prev, ch) if b['inst'] in keep])
    return dict(fr_tgt, cams=cams)


def render_assoc_layout(fr_prev, fr_tgt, out_hw=LAYOUT_HW, stride=LAYOUT_STRIDE):
    """关联通道 uint8 [6,3,H,W]：fr_prev 的框（只取两帧都在场的实例）画进 fr_tgt 的相机，filled 编码、类别上色、不画地图。
    直接走 ``render_layout``（同一份投影 / 填充 / 补边代码）；fr_prev 与 fr_tgt 是同一帧时与 ``render_layout(fr_tgt)`` 逐字节相同。"""
    return render_layout(assoc_frame(fr_prev, fr_tgt), out_hw, stride, map_vec=None)


def render_layout_cond(fr_prev, fr_tgt, map_vec=None, assoc=False):
    """预测器 layout 条件的唯一入口（训练 / 评测共用）：预测目标帧 fr_tgt 时的 layout，uint8 [6,C,H,W]。
    assoc=False：C=3，**就是** ``render_layout(fr_tgt, map_vec=map_vec)``（fr_prev 不用）；
    assoc=True：C=6 = [主通道（目标帧框 + 可选地图）, 关联通道 ``render_assoc_layout(fr_prev, fr_tgt)``]（ckpt 的 ``model_kwargs['assoc_layout']``）。"""
    lay = render_layout(fr_tgt, map_vec=map_vec)
    if not assoc:
        return lay
    return np.concatenate([lay, render_assoc_layout(fr_prev, fr_tgt)], axis=1)


def relative_action(fr0, fr1):
    """帧0→帧1 的自车相对位姿，在帧0 自车系：[dx, dy, dyaw]（AdaLN 3 维口径）。"""
    T = np.linalg.inv(fr0['ego2global']) @ fr1['ego2global']
    return np.array([T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0])], np.float32)


if __name__ == '__main__':
    import time
    t = time.time()
    idx = build_index(force=True)
    print('index built', len(idx['scenes']), 'scenes', len(idx['frames']), 'frames', f'{time.time() - t:.1f}s')
    sp = splits()
    n_tr = sum(len(idx['scenes'][s]) for s in sp['train'] if s in idx['scenes'])
    n_va = sum(len(idx['scenes'][s]) for s in sp['val'] if s in idx['scenes'])
    print('train frames', n_tr, 'val frames', n_va)
    acts = np.array([relative_action(a, b) for s in sp['train'][:100] for a, b in zip(idx['scenes'][s][:-1], idx['scenes'][s][1:])])
    print('action mean', acts.mean(0), 'std', acts.std(0))
    dts = np.array([(b['timestamp'] - a['timestamp']) / 1e6 for s in sp['train'][:100] for a, b in zip(idx['scenes'][s][:-1], idx['scenes'][s][1:])])
    print('dt mean %.3f std %.3f min %.3f max %.3f' % (dts.mean(), dts.std(), dts.min(), dts.max()))
