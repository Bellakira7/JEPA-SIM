"""offlog/layout_reproject.py 的自检（纯 CPU）。用法（在 uniad_bench 下）：

    venv/bin/python -m offlog.test_layout_reproject [--viz]

1. 恒等位姿 → 与训练口径 `nd.render_layout` **逐字节相同**（有/无地图）；另强制走矩阵合成路径，应同样逐字节相同（合成公式在恒等点还原）。
2. 位姿合成 vs 独立路径：横向 / 航向 / 速度扰动下，合成帧里的框中心（ego 系）与 `GTWorld.boxes_in`（offlog 已逐位验证过的世界→仿真位姿变换，LiDAR 系）
   经 lidar→ego 外参换算后按 instance 对齐，应在 1e-6 m 内。
3. 旋转：框在 ego 系的 yaw 变化 = −pose.dpsi（ego 绕全局 z 转 dψ；ego 有俯仰/侧倾 tilt 时偏差 ≈ tilt²·|dψ|/2，判据用这个理论界而不是拍定容差）。
4. 图像方向：k≥4（偏移到位、航向回到日志）时，横向 δ=+2 m（自车向左）CAM_FRONT 里的 layout 质心应**向右**（列号增大），δ=−2 应向左；航向 +5°（向左转）也向右。
5. 地图关键帧：速度 ×1.5 时 fr_map = 路径位置最近的关键帧（≠ tmpl）；横向扰动时 fr_map = tmpl。
6. action 约定：恒等位姿的 `sim_relative_actions` 与 `nd.relative_action`（训练/A 表用的）一致；6b 偏移到位后仿真与日志的相对运动之差 = 解析式 δ·(−sinΔψ, cosΔψ−1, 0)。
"""
import argparse
import math
import os
import sys
import time

import numpy as np
from pyquaternion import Quaternion

from . import common
from .gt_world import GTWorld
from .layout_reproject import SimLayoutRenderer, sim_relative_actions
from .sim_pose import LogTrack, actions
from uniad_port.data import UniADValData  # noqa: E402  (common 已把 uniad_port 加进 sys.path)
import nusc_data as nd  # noqa: E402  (layout_reproject 已把 feat 加进 sys.path)

fails = []


def check(name, cond, msg=''):
    print(f'[{"PASS" if cond else "FAIL"}] {name} {msg}', flush=True)
    if not cond:
        fails.append(name)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--viz', action='store_true', help='另存 CAM_FRONT 三联图 offlog/viz/layout_reproject_front.png')
    a = ap.parse_args()

    data = UniADValData(num_prefetch=1)
    rend = SimLayoutRenderer(data.infos)
    gtw = GTWorld([i['token'] for i in data.infos], cache_path=os.path.join(common.CACHE_DIR, 'gt_world_val.pkl'))
    scene_ids = [i for i in (1, 12, 33, 58, 77, 101, 124, 147) if i < len(data.scenes)]
    tracks = {si: LogTrack(data, data.scenes[si]) for si in scene_ids}

    # ---- 1. 恒等位姿逐字节相同 ----
    n_cmp = n_nonempty = 0
    bad = []
    for si in scene_ids[:4]:
        tr = tracks[si]
        for aa in tr.anchors(stride=8)[:2]:
            for pose in tr.rollout(aa, 'lat', 0.0)[:3]:
                assert pose.identity
                fr = rend.nd_frame(pose.tmpl_idx)
                for with_map in (False, True):
                    mv = rend.map_vecs.for_frame(fr) if with_map else None
                    ref = nd.render_layout(fr, map_vec=mv)
                    for force in (False, True):
                        got = rend.render(pose, with_map, force_compose=force)
                        n_cmp += 1
                        if not np.array_equal(got, ref):
                            bad.append((si, aa, pose.k, with_map, force, int((got != ref).sum())))
                    n_nonempty += int(ref.any())
    check('1 identity pose == training render_layout (byte-exact, with/without map, shortcut and forced-compose paths)',
          not bad and n_cmp > 0 and n_nonempty > n_cmp // 8, f'{n_cmp} comparisons, nonempty layouts {n_nonempty}, mismatches {bad[:3]}')

    # ---- 2/3. 位姿合成 vs GTWorld；yaw 变化 = −dpsi ----
    worst_c, worst_yaw, worst_ratio, n_big, n_box, n_poses = 0.0, 0.0, 0.0, 0, 0, 0
    families = [('lat', 2.0), ('lat', -1.0), ('yaw', 5.0), ('speed', 1.5), ('speed', 0.5)]
    for si in scene_ids[:5]:
        tr = tracks[si]
        for aa in tr.anchors(stride=8, min_speed=0.0)[:1]:
            for fam, val in families:
                try:
                    poses = tr.rollout(aa, fam, val)
                except ValueError:
                    continue
                for pose in poses[1:]:
                    fr_sim, _ = rend.sim_frame(pose)
                    fr0 = rend.nd_frame(pose.tmpl_idx)
                    box, _ = nd.boxes_in_ego(fr_sim, fr_sim['ego2global'])
                    if len(box) == 0:
                        continue
                    inst = [b['inst'] for b in fr_sim['boxes']]
                    g = gtw.boxes_in(data.infos[pose.tmpl_idx]['token'], pose.e2g_t, pose.e2g_q, pose.l2e_t, pose.l2e_q)
                    gi = {k: j for j, k in enumerate(g['instance'])}
                    l2e_R = Quaternion(pose.l2e_q).rotation_matrix
                    l2e_t = np.asarray(pose.l2e_t, np.float64)
                    n_poses += 1
                    for j, k in enumerate(inst):
                        if k not in gi:
                            continue
                        c_l = g['boxes'][gi[k], :3].copy()
                        c_l[2] += g['boxes'][gi[k], 5] / 2                       # z_bottom → 中心
                        worst_c = max(worst_c, float(np.abs(c_l @ l2e_R.T + l2e_t - box[j, :3]).max()))
                        n_box += 1
                    # 3. yaw：与日志位姿（δ=0）下同一个框的 ego 系 yaw 相比，变化应为 −dpsi（速度缩放时位姿在路径上，dpsi 未定义，跳过）。
                    # 扰动是绕**全局 z** 转 dψ，而 yaw 是在（有俯仰/侧倾 tilt 的）ego 系里用 arctan2 量的，所以偏差 ≈ tilt²·|dψ|/2；判据留 2 倍余量（tilt²·|dψ|）。
                    # 符号或旋转合成错了会差 ~|dψ|（≥0.05 rad 的位姿 n_big 个），比这个界大三个数量级。
                    if fam != 'speed':
                        box0, _ = nd.boxes_in_ego(fr0, fr0['ego2global'])
                        res = float(np.abs(wrap(box[:, 6] - box0[:, 6] + pose.dpsi)).max())
                        tilt = math.acos(min(1.0, fr0['ego2global'][2, 2]))
                        worst_yaw = max(worst_yaw, res)
                        worst_ratio = max(worst_ratio, res / (tilt ** 2 * abs(pose.dpsi) + 1e-9))
                        n_big += int(abs(pose.dpsi) > 0.05)
    check('2 composed pose boxes == GTWorld.boxes_in (independent verified path), centers in ego frame', n_box > 50 and worst_c < 1e-6,
          f'{n_box} boxes over {n_poses} poses, max center diff {worst_c:.2e} m')
    check('3 box yaw in ego frame changes by −dpsi (within tilt²·|dpsi| from ego pitch/roll)', n_poses > 0 and n_big >= 20 and worst_ratio < 1.0,
          f'max |Δyaw + dpsi| {worst_yaw:.2e} rad, worst residual / (tilt²·|dpsi|) = {worst_ratio:.2f}, poses with |dpsi|>0.05: {n_big}')

    # ---- 4. 图像方向 ----
    def front_centroid(L):
        m = L[0].sum(0) > 0                                                     # CAM_FRONT：[H,W]
        return (float((m.sum(0) * np.arange(m.shape[1])).sum() / m.sum()), int(m.sum())) if m.sum() >= 30 else (np.nan, int(m.sum()))
    shifts = {('lat', 2.0): [], ('lat', -2.0): [], ('yaw', 5.0): []}
    for si in scene_ids:
        tr = tracks[si]
        for aa in tr.anchors(stride=4, min_speed=5.0)[:2]:
            base = tr.rollout(aa, 'lat', 0.0)[6]
            c0, _ = front_centroid(rend.render(base, with_map=False))
            for key in shifts:
                c1, _ = front_centroid(rend.render(tr.rollout(aa, *key)[6], with_map=False))
                if np.isfinite(c0) and np.isfinite(c1):
                    shifts[key].append(c1 - c0)
    med = {k: (float(np.median(v)) if v else np.nan) for k, v in shifts.items()}
    check('4 CAM_FRONT layout shifts right for δ=+2 m / yaw +5°, left for δ=−2 m (median over anchors at k=6)',
          all(len(v) >= 3 for v in shifts.values()) and med[('lat', 2.0)] > 2 and med[('lat', -2.0)] < -2 and med[('yaw', 5.0)] > 2,
          f'median centroid shift (canvas cols; 1 col = 8 px): ' + ', '.join(f'{k}: {m:+.1f} (n={len(shifts[k])})' for k, m in med.items()))

    # ---- 5. 地图关键帧 ----
    ok5, n5 = True, 0
    for si in scene_ids:
        tr = tracks[si]
        for aa in tr.anchors(stride=4, min_speed=0.0)[:1]:
            for fam, val in (('lat', 2.0), ('speed', 1.5)):
                try:
                    poses = tr.rollout(aa, fam, val)
                except ValueError:
                    continue
                p = poses[8]
                fr_sim, fr_map = rend.sim_frame(p)
                want = data.infos[p.cmd_idx]['token']
                ok5 &= fr_map['token'] == want and fr_sim['token'] == data.infos[p.tmpl_idx]['token']
                ok5 &= (fam != 'speed') or (p.cmd_idx != p.tmpl_idx and fr_map['idx'] != fr_sim['idx'])
                ok5 &= (fam != 'lat') or (p.cmd_idx == p.tmpl_idx)
                n5 += 1
    check('5 map keyframe = nearest log keyframe on path (speed) / tmpl (lat); boxes from time-aligned tmpl frame', ok5 and n5 > 0, f'{n5} poses')

    # ---- 6. action 约定 ----
    worst_a, worst_old, n6 = 0.0, 0.0, 0
    for si in scene_ids[:4]:
        tr = tracks[si]
        for aa in tr.anchors(stride=8)[:2]:
            poses = tr.rollout(aa, 'lat', 0.0)
            ref = np.stack([nd.relative_action(rend.nd_frame(p0.tmpl_idx), rend.nd_frame(p1.tmpl_idx)) for p0, p1 in zip(poses[:-1], poses[1:])])
            worst_a = max(worst_a, float(np.abs(sim_relative_actions(poses) - ref).max()))
            worst_old = max(worst_old, float(np.abs(actions(poses) - ref)[:, 2].max()))
            n6 += 1
    check('6 identity-pose sim_relative_actions == nd.relative_action (train/A-table convention)', n6 > 0 and worst_a < 2e-6,
          f'{n6} rollouts, max diff {worst_a:.2e}（对照：offlog.sim_pose.actions 的 dyaw 与训练口径差 {worst_old:.2e} rad，B 表不用它）')
    # 6b. 偏移到位（k≥4：横向偏移 = δ、航向回到日志）后，仿真与日志的相对运动之差有解析式：两个位姿各自沿本帧左轴平移 δ，
    #     相对平移 = 日志的 + δ·(R(Δψ)−I)·[0,1]ᵀ = (−δ·sinΔψ, δ·(cosΔψ−1))，相对 yaw 不变。
    worst_tail, n6b = 0.0, 0
    for si in scene_ids:
        tr = tracks[si]
        for aa in tr.anchors(stride=4, min_speed=5.0)[:2]:
            a_log = sim_relative_actions(tr.rollout(aa, 'lat', 0.0))[4:]
            a_sim = sim_relative_actions(tr.rollout(aa, 'lat', 2.0))[4:]
            dpsi = a_log[:, 2].astype(np.float64)
            pred = np.stack([-2.0 * np.sin(dpsi), 2.0 * (np.cos(dpsi) - 1.0), np.zeros_like(dpsi)], 1)
            worst_tail = max(worst_tail, float(np.abs((a_sim - a_log) - pred).max()))
            n6b += 1
    check('6b lateral δ=+2 m: after the ramp (k≥4) sim − log relative motion == δ·(−sinΔψ, cosΔψ−1, 0)', n6b >= 3 and worst_tail < 5e-3,
          f'{n6b} rollouts, max abs deviation from the analytic prediction {worst_tail:.2e} (m / rad)')

    # ---- 速度 ----
    tr = tracks[scene_ids[0]]
    pose = tr.rollout(tr.anchors(stride=8)[0], 'lat', 2.0)[6]
    t0 = time.time()
    for _ in range(10):
        rend.render(pose, with_map=True)
    print(f'[info] render(with_map) {1000 * (time.time() - t0) / 10:.0f} ms / pose（6 相机）')

    if a.viz:
        import cv2
        best = None
        for si in scene_ids:
            tr = tracks[si]
            for aa in tr.anchors(stride=4, min_speed=5.0)[:2]:
                n = len(rend.nd_frame(tr.rollout(aa, 'lat', 0.0)[6].tmpl_idx)['boxes'])
                if best is None or n > best[0]:
                    best = (n, si, aa)
        _, si, aa = best
        tr = tracks[si]
        base = tr.rollout(aa, 'lat', 0.0)[6]
        fr = rend.nd_frame(base.tmpl_idx)
        im = cv2.imread(fr['cams']['CAM_FRONT']['path'])
        s = nd.LAYOUT_STRIDE
        vh = int(np.ceil(nd.IMG_HW[0] / s))
        panels = []
        for lab, (fam, val) in (('delta=-2m (right)', ('lat', -2.0)), ('delta=0 (log pose, on real image)', ('lat', 0.0)), ('delta=+2m (left)', ('lat', 2.0))):
            L = rend.render(tr.rollout(aa, fam, val)[6], with_map=True)[0].transpose(1, 2, 0)[:vh]
            L = cv2.resize(L, (nd.IMG_HW[1], nd.IMG_HW[0]), interpolation=cv2.INTER_NEAREST)
            ov = np.where(L.sum(2, keepdims=True) > 0, (0.45 * im + 0.55 * L).astype(np.uint8), im)
            cv2.putText(ov, lab, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
            panels.append(cv2.resize(ov, (800, 450)))
        out = os.path.join(common.UNIAD_BENCH, 'offlog', 'viz', 'layout_reproject_front.png')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, np.concatenate(panels, 1))
        print(f'[info] saved {out}（scene idx {si}, anchor {aa}, {best[0]} boxes）')

    print('ALL PASS' if not fails else f'FAILED: {fails}')
    return not fails


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
