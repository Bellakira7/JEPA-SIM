"""nuScenes 相机 sweeps（12 Hz）→ 2 Hz 相位增广链（设计稿 §3.4）。

关键帧 2 Hz（间隔 499±27 ms），相机 sweeps 12 Hz（间隔按 100/100/50 ms 循环），所以每个关键帧之后 {0,100,200,250,350,450} ms 处各有一帧
真实图像 → 把"整套采样时刻"整体平移 δ_p = p·500/6 ms（p=0..5）就得到 6 条相位错开的 2 Hz 链，样本量约 ×6。
链上相邻帧的时间间隔 = 对应两个关键帧的间隔（与部署口径一致，不是 6 个 sweep 步）。

**第 i 帧、相位 p 的定义**（p=0 就是原关键帧，直接返回 ``nusc_index`` 里的原 dict，逐位不变）：
  - 每个相机：目标时刻 = 该相机第 i 个关键帧时间戳 + δ_p，取时间上最近的相机 sweep（p=0 恰好是关键帧自己）。
  - 帧级 ``ego2global``（动作用）：LIDAR_TOP 里最近 τ_i + δ_p 的那个 sweep 的自车位姿（p=0 即关键帧的 lidar 位姿，与原口径一致）。
  - GT 框：标注只在关键帧（lidar 时刻 τ）上。每个相机各自按它选中 sweep 的时间戳 t_c，在相邻关键帧 i、i+1 的标注之间线性插值
    （中心线性，yaw 沿最短弧；α=(t_c-τ_i)/(τ_{i+1}-τ_i)，限制在 [-0.25, 1.25]，略微外推）。**只在 i、i+1 都有标注的实例上插值**
    （只在一边出现的实例直接丢弃，不凭空外推）。尺寸取 i 帧的。
  - p≥1 的链需要关键帧 i+1，所以第 p≥1 条链比 p=0 少最后一帧。

返回的帧 dict 与 ``nusc_data`` 的关键帧 dict 同结构（``cams[ch]`` 多一个 ``boxes`` = 该相机时刻的框，``render_layout`` 优先用它）。


索引 ``feat/nusc_sweep_index.pkl`` 由 ``python nusc_sweeps.py`` 从上游表（``v1.0-trainval/sample_data.json`` 等）建一次。
"""
import featsim_paths as _release_paths
import collections
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nusc_data as nd

UP_ROOT = _release_paths.NUSC_MAP_ROOT
SWEEP_CACHE = str(_release_paths.Path(_release_paths.BENCH) / 'feat/nusc_sweep_index.pkl')
N_PHASES = 6
PHASE_US = 500_000 // N_PHASES  # 83333 µs
ALPHA_RANGE = (-0.25, 1.25)
CHANNELS = list(nd.CAMS) + ['LIDAR_TOP']


def build_sweep_index(force=False):
    """返回 {'scenes': {scene_name: {channel: dict(ts int64[N], rel list[str], pose float64[N,7], key bool[N], kpos int64[n_key])}}}。
    pose = [tx,ty,tz, qw,qx,qy,qz]（自车 → global，该 sample_data 自己的 ego_pose）；kpos = 关键帧在该通道时间序里的下标（按关键帧顺序）。"""
    if os.path.exists(SWEEP_CACHE) and not force:
        with open(SWEEP_CACHE, 'rb') as f:
            return pickle.load(f)
    d = f'{UP_ROOT}/v1.0-trainval'
    ld = lambda n: json.load(open(f'{d}/{n}.json'))
    t0 = time.time()
    scene_of_sample = {x['token']: x['scene_token'] for x in ld('sample')}
    scene_name = {x['token']: x['name'] for x in ld('scene')}
    ego = {x['token']: x['translation'] + x['rotation'] for x in ld('ego_pose')}
    rows = collections.defaultdict(list)
    calib = collections.defaultdict(set)
    for x in ld('sample_data'):
        parts = x['filename'].split('/')
        ch = parts[1]
        if ch not in CHANNELS:
            continue
        sc = scene_name[scene_of_sample[x['sample_token']]]
        rows[(sc, ch)].append((x['timestamp'], x['filename'], ego[x['ego_pose_token']], x['is_key_frame']))
        calib[(sc, ch)].add(x['calibrated_sensor_token'])
    print(f'[sweep index] 读表 {time.time() - t0:.0f}s，{len(rows)} 个 (scene, channel)', flush=True)
    idx = nd.build_index()
    scenes = {}
    for (sc, ch), r in rows.items():
        assert len(calib[(sc, ch)]) == 1, (sc, ch, 'sweeps 与关键帧的 calibrated_sensor 不一致', calib[(sc, ch)])
        r.sort(key=lambda z: z[0])
        key = np.array([z[3] for z in r], bool)
        out = dict(ts=np.array([z[0] for z in r], np.int64), rel=[z[1] for z in r],
                   pose=np.array([z[2] for z in r], np.float64), key=key, kpos=np.flatnonzero(key))
        kf = idx['scenes'].get(sc)
        if kf is not None:  # 关键帧数、顺序、路径都要与 nusc_index 对上
            assert len(out['kpos']) == len(kf), (sc, ch, len(out['kpos']), len(kf))
            if ch != 'LIDAR_TOP':
                for k, fr in zip(out['kpos'], kf):
                    assert fr['cams'][ch]['path'].endswith(out['rel'][k]), (sc, ch, fr['cams'][ch]['path'], out['rel'][k])
        scenes.setdefault(sc, {})[ch] = out
    res = dict(scenes=scenes, n_phases=N_PHASES)
    os.makedirs(os.path.dirname(SWEEP_CACHE), exist_ok=True)
    tmp = SWEEP_CACHE + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(res, f, protocol=4)
    os.replace(tmp, SWEEP_CACHE)
    print(f'[sweep index] {len(scenes)} 个场景，{time.time() - t0:.0f}s，写入 {SWEEP_CACHE}', flush=True)
    return res


def pose_to_T(p):
    return nd.make_T(nd.quat_to_R(p[3:7]), p[:3])


def _yaw(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _rz(y):
    c, s = np.cos(y), np.sin(y)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def interp_boxes(boxes_a, boxes_b, alpha):
    """相邻关键帧标注之间的线性插值。只保留两帧都有的实例（按 ``inst``）；中心线性、yaw 沿最短弧、尺寸/类别取 a。
    ``R`` 返回纯 yaw 旋转（``boxes_in_ego`` 只取 yaw，layout 不受影响）。"""
    alpha = float(np.clip(alpha, *ALPHA_RANGE))
    mb = {b['inst']: b for b in boxes_b}
    out = []
    for a in boxes_a:
        b = mb.get(a['inst'])
        if b is None:
            continue
        ya, yb = _yaw(a['R']), _yaw(b['R'])
        dy = (yb - ya + np.pi) % (2 * np.pi) - np.pi
        out.append(dict(center=(1 - alpha) * a['center'] + alpha * b['center'], wlh=a['wlh'], R=_rz(ya + alpha * dy),
                        cls=a['cls'], inst=a['inst']))
    return out


def _nearest(ts, t):
    j = int(np.searchsorted(ts, t))
    if j <= 0:
        return 0
    if j >= len(ts):
        return len(ts) - 1
    return j if abs(ts[j] - t) < abs(ts[j - 1] - t) else j - 1


class SweepFrames:
    """按 (scene, 关键帧下标 i, 相位 p) 现算帧 dict（轻量：只做查表 + 插值，在 DataLoader worker 里调用）。"""

    def __init__(self, idx=None, sweeps=None):
        self.idx = idx if idx is not None else nd.build_index()
        self.sw = sweeps if sweeps is not None else build_sweep_index()

    def chain_len(self, scene, p):
        n = len(self.idx['scenes'][scene])
        return n if p == 0 else n - 1

    def frame(self, scene, i, p):
        kf = self.idx['scenes'][scene]
        fr = kf[i]
        if p == 0:
            return fr
        assert 0 < p < N_PHASES and i + 1 < len(kf), (scene, i, p)
        nxt = kf[i + 1]
        sw = self.sw['scenes'][scene]
        delta = p * PHASE_US
        lid = sw['LIDAR_TOP']
        tau_i, tau_n = int(lid['ts'][lid['kpos'][i]]), int(lid['ts'][lid['kpos'][i + 1]])
        ego2global = pose_to_T(lid['pose'][_nearest(lid['ts'], tau_i + delta)])
        cams = {}
        for ch in nd.CAMS:
            c = sw[ch]
            j = _nearest(c['ts'], int(c['ts'][c['kpos'][i]]) + delta)
            tc = int(c['ts'][j])
            base = fr['cams'][ch]
            cams[ch] = dict(path=f"{UP_ROOT}/{c['rel'][j]}", K=base['K'], cam2ego=base['cam2ego'], ego2global=pose_to_T(c['pose'][j]), t=tc,
                            boxes=interp_boxes(fr['boxes'], nxt['boxes'], (tc - tau_i) / (tau_n - tau_i)))
        # 地图是静态的，取离该帧最近的关键帧的矢量地图（±51.2 m 范围内更完整）
        map_idx = i if 2 * delta <= tau_n - tau_i else i + 1
        return dict(token=f"{fr['token']}@p{p}", scene=scene, idx=i, map_idx=map_idx, phase=p, timestamp=tau_i + delta, cams=cams,
                    ego2global=ego2global, boxes=interp_boxes(fr['boxes'], nxt['boxes'], delta / (tau_n - tau_i)))


if __name__ == '__main__':
    force = '--force' in sys.argv
    sw = build_sweep_index(force=force)
    n = sum(len(v['CAM_FRONT']['ts']) for v in sw['scenes'].values())
    print('CAM_FRONT 帧总数（关键帧+sweeps）', n)
