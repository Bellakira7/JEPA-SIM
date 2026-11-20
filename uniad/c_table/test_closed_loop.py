"""C 表闭环组件的 CPU 单测（自检 ③ 的主体）：自行车模型、导航命令、can_bus / 位姿合成、碰撞分类、可行驶区域足迹。

    venv/bin/python c_table/test_closed_loop.py [--full]
--full 再在全部 val 场景上：① 日志自车走自行车模型一步的落点误差；② 路线命令与官方命令逐帧一致；③ 日志自车的有责碰撞（健全性：应几乎为 0）。

"""
import argparse
import math
import os
import sys

import numpy as np
from pyquaternion import Quaternion

ROOT = os.environ.get('G3_CODE_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, 'c_table'), os.path.join(ROOT, 'feat'), ROOT, os.path.join(ROOT, 'uniad_port')):
    if p not in sys.path:
        sys.path.insert(0, p)
from offlog import common  # noqa: E402,F401
from offlog.sim_pose import LogTrack, quaternion_yaw, synth_can_bus  # noqa: E402
import closed_loop as cl  # noqa: E402

FAILS = []


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def t_bicycle():
    print('[bicycle]')
    ref = np.array([0.94, 0.0])

    def straight(v, n=6):
        return np.array([ref + [v * 0.5 * i, 0.0] for i in range(n + 1)])
    # 直行匀速：v0=10，规划 10 m/s → a=0，落在第一个路点
    st = cl.track_plan(10.0, straight(10.0))
    check(abs(st['dx'] - 5.0) < 1e-9 and abs(st['dy']) < 1e-12 and abs(st['a']) < 1e-9 and abs(st['v1'] - 10.0) < 1e-9, f'直行 dx={st["dx"]:.6f} a={st["a"]:.3g}')
    # 弯道：规划 = 后轴沿曲率 κ 匀速 v 的圆弧上参考点的位置 → 参考点落在第一个路点（弦长 vs 弧长、参考点半径 ≠ 后轴半径带来 mm 级差）
    for kap, v in ((0.05, 8.0), (-0.08, 6.0), (0.15, 4.0)):
        pts = []
        for i in range(7):
            x, y, psi = cl.arc_pose(kap, v * 0.5 * i)
            pts.append(np.array([x, y]) + cl.rot2(psi) @ ref)
        st = cl.track_plan(v, np.array(pts))
        landed = np.array([st['dx'], st['dy']]) + cl.rot2(st['dpsi']) @ ref
        err = float(np.linalg.norm(landed - pts[1]))
        check(err < 0.02 and not st['clipped_steer'] and not st['clipped_acc'], f'弯道 κ={kap} v={v}：落点误差 {err:.4f} m，v_target={st["v_target"]:.3f}')
    # 规划减速：目标速度 = 两段平均
    pts = np.array([ref, ref + [4.0, 0], ref + [7.0, 0], ref + [9.0, 0], ref + [10.0, 0], ref + [10.5, 0], ref + [10.5, 0]])
    st = cl.track_plan(8.0, pts)
    check(abs(st['v_target'] - 7.0) < 1e-9 and abs(st['a'] + 2.0) < 1e-9 and abs(st['v1'] - 7.0) < 1e-9, f'减速 v_target={st["v_target"]} a={st["a"]}')
    # 停：路点都在原地 → 刹到 0，不倒车
    st = cl.track_plan(3.0, np.array([ref] * 7))
    check(st['v1'] == 0.0 and st['dx'] >= 0, f'原地规划 → 停 v1={st["v1"]} dx={st["dx"]:.3f}')
    st = cl.track_plan(5.0, np.array([ref] + [ref + [-1.0 * i, 0.0] for i in range(1, 7)]))
    check(st['v1'] >= 0 and st['dx'] >= -1e-12 and st['v_target'] == 0.0 and st['kappa'] == 0.0, f'路点在后方 → 刹车不倒车 dx={st["dx"]:.3f} v1={st["v1"]:.2f}')
    # 加速度限幅
    st = cl.track_plan(0.0, straight(20.0))
    check(st['clipped_acc'] and abs(st['a'] - cl.A_MAX) < 1e-12 and abs(st['dx'] - 0.5 * cl.A_MAX * 0.25) < 1e-9, f'加速限幅 a={st["a"]} dx={st["dx"]:.3f}')
    # 低速：路点只有几十厘米、横向噪声 0.1 m → 最小前视距离下不满舵（不加前视就满舵）
    pts = np.array([ref] + [ref + [0.5 * i, 0.12 * (-1) ** i] for i in range(1, 7)])
    st = cl.track_plan(1.0, pts)
    st0 = cl.track_plan(1.0, pts, l_min=0.0)
    check(not st['clipped_steer'] and st0['clipped_steer'], f'低速前视：steer={st["steer"]:.3f}（无前视 {st0["steer"]:.3f}）')
    # 转角限幅：急弯
    pts = np.array([ref] + [ref + [1.5 * i, 2.0 * i] for i in range(1, 7)])
    st = cl.track_plan(2.0, pts)
    check(st['clipped_steer'] and abs(st['steer']) <= cl.MAX_STEER + 1e-12, f'转角限幅 steer={st["steer"]:.3f}')
    check(abs(st['sub'][-1][1] - st['dx']) < 1e-12 and all(st['sub'][i][0] < st['sub'][i + 1][0] for i in range(len(st['sub']) - 1)), '子时刻')


def t_collision():
    print('[collision]')
    ego, front, rear = cl.ego_box((0.0, 0.0), 0.0)
    c = 0.94 + cl.EGO_FWD
    check(abs(ego.centroid.x - c) < 1e-9 and abs(ego.area - cl.EGO_L * cl.EGO_W) < 1e-9, f'自车框中心 {ego.centroid.x:.3f} 面积 {ego.area:.3f}')

    class GTW:
        def __init__(self, boxes0, boxes1):
            self.data = {'t0': self._mk(boxes0), 't1': self._mk(boxes1)}

        @staticmethod
        def _mk(bs):
            n = len(bs)
            return dict(det_name=[b[0] for b in bs], wlh=np.array([[b[4], b[3], 1.5] for b in bs]).reshape(n, 3),
                        rot_g=np.array([list(Quaternion(axis=[0, 0, 1], angle=b[5]).elements) for b in bs]).reshape(n, 4),
                        center_g=np.array([[b[1], b[2], 0.0] for b in bs]).reshape(n, 3), instance=[b[6] for b in bs])

    sub_move = [(0.1 * i, 1.0 * i, 0.0, 0.0, 10.0) for i in range(1, 6)]      # 自车 10 m/s 前进
    sub_stop = [(0.1 * i, 0.0, 0.0, 0.0, 0.0) for i in range(1, 6)]
    # 前方静止车：自车开上去 → stopped_track，有责
    g = GTW([('car', c + 5.5, 0.0, 4.0, 1.8, 0.0, 'A')], [('car', c + 5.5, 0.0, 4.0, 1.8, 0.0, 'A')])
    r = cl.collisions_interval(g, 't0', 't1', sub_move, True, set())
    check(len(r) == 1 and r[0]['kind'] == 'stopped_track' and r[0]['at_fault'], f'撞前方静止车 {[(x["kind"], x["at_fault"]) for x in r]}')
    # 前方慢车（在动）：front 有责
    g = GTW([('car', c + 6.0, 0.0, 4.0, 1.8, 0.0, 'B')], [('car', c + 7.0, 0.0, 4.0, 1.8, 0.0, 'B')])
    r = cl.collisions_interval(g, 't0', 't1', sub_move, True, set())
    check(len(r) == 1 and r[0]['kind'] == 'active_front' and r[0]['at_fault'], f'追尾前车 {[(x["kind"], x["at_fault"]) for x in r]}')
    # 后车追尾（自车在动但慢）：rear 不算有责
    sub_slow = [(0.1 * i, 0.1 * i, 0.0, 0.0, 1.0) for i in range(1, 6)]
    g = GTW([('car', c - 6.0, 0.0, 4.0, 1.8, 0.0, 'C')], [('car', c - 2.5, 0.0, 4.0, 1.8, 0.0, 'C')])
    r = cl.collisions_interval(g, 't0', 't1', sub_slow, True, set())
    check(len(r) == 1 and r[0]['kind'] == 'active_rear' and not r[0]['at_fault'], f'被追尾 {[(x["kind"], x["at_fault"]) for x in r]}')
    # 自车静止被撞：不算
    g = GTW([('car', c - 6.0, 0.0, 4.0, 1.8, 0.0, 'D')], [('car', c - 2.5, 0.0, 4.0, 1.8, 0.0, 'D')])
    r = cl.collisions_interval(g, 't0', 't1', sub_stop, True, set())
    check(len(r) == 1 and r[0]['kind'] == 'stopped_ego' and not r[0]['at_fault'], f'静止被撞 {[(x["kind"], x["at_fault"]) for x in r]}')
    # 侧面剐蹭：在路面上不算有责，出了路面算
    g = GTW([('car', c, 3.0, 4.0, 1.8, 0.0, 'E')], [('car', c + 5.0, 1.2, 4.0, 1.8, 0.0, 'E')])
    r1 = cl.collisions_interval(g, 't0', 't1', sub_move, True, set())
    r2 = cl.collisions_interval(g, 't0', 't1', sub_move, False, set())
    check(len(r1) == 1 and r1[0]['kind'] == 'active_lateral' and not r1[0]['at_fault'] and r2[0]['at_fault'], f'侧面 {[(x["kind"], x["at_fault"]) for x in r1]} / 出路面 {r2[0]["at_fault"]}')
    # 同一实例只记一次；无关类别不计
    seen = set()
    g = GTW([('car', c + 5.5, 0.0, 4.0, 1.8, 0.0, 'A'), (None, c + 5.5, 0.0, 4.0, 1.8, 0.0, 'Z')], [('car', c + 5.5, 0.0, 4.0, 1.8, 0.0, 'A'), (None, c + 5.5, 0.0, 4.0, 1.8, 0.0, 'Z')])
    r = cl.collisions_interval(g, 't0', 't1', sub_move, True, seen) + cl.collisions_interval(g, 't0', 't1', sub_move, True, seen)
    check(len(r) == 1, f'实例去重 / 类别过滤 n={len(r)}')
    # 足迹
    drv = np.zeros((200, 200), np.float32)
    drv[90:110, 95:105] = 1
    ok, frac = cl.footprint_on_drivable(drv)
    check(ok and frac == 1.0, f'足迹全在路面 {ok} {frac}')
    drv[:, 101:] = 0
    ok, frac = cl.footprint_on_drivable(drv)
    check((not ok) and 0 < frac < 1, f'足迹部分出路面 {ok} {frac:.3f}')


def t_log(data, scenes, full):
    from offlog.gt_world import GTWorld
    from offlog.metas import OffLogEnv
    print(f'[log] {len(scenes)} 个场景')
    errs, yerrs, lands, clips, n_cmd, n_bad_cmd, n_clip = [], [], [], [], 0, 0, 0
    env = OffLogEnv(data)
    n_can = []
    gtw = GTWorld([i['token'] for i in data.infos], cache_path=os.path.join(common.CACHE_DIR, 'gt_world_val.pkl')) if full else None
    af = coll = 0
    kinds = {}
    for si in scenes:
        tr = LogTrack(data, data.scenes[si])
        route = cl.Route(data, tr)
        seen = set()
        for j in range(tr.n):
            pose = cl.log_pose(tr, j)
            n_cmd += 1
            n_bad_cmd += int(route.command(pose, j) != data.planning_label(tr.ids[j])[2])
            if j + 1 < tr.n and tr.speed[j] > 1.0:
                plan, mask, _ = data.planning_label(tr.ids[j])
                if not mask[0]:
                    continue
                p1, st = cl.advance(tr, route, pose, float(j), float(tr.speed[j]), plan[:, :2], j + 1)
                clip = st['clipped_acc'] or st['clipped_steer']
                n_clip += int(clip)
                clips.append(clip)
                errs.append(float(np.linalg.norm(np.asarray(p1.e2g_t[:2]) - tr.T[j + 1, :2])))
                yerrs.append(abs((quaternion_yaw(Quaternion(p1.e2g_q)) - tr.yaw[j + 1] + math.pi) % (2 * math.pi) - math.pi))
                lands.append(abs(st['f'] - (j + 1)))
                fr = env.sim_frame(p1)
                n_can.append(abs(fr['img_metas']['can_bus'][13] - tr.can[j + 1][13]))
            if full and j >= 1:
                sub = cl.log_substeps(tr, j - 1, j)
                cs = cl.collisions_interval(gtw, data.infos[tr.ids[j - 1]]['token'], data.infos[tr.ids[j]]['token'], sub, True, seen,
                                            tuple(tr.infos[0]['lidar2ego_translation'][:2]))
                for c in cs:
                    coll += 1
                    af += int(c['at_fault'])
                    kinds[c['kind']] = kinds.get(c['kind'], 0) + 1
    errs, yerrs = np.array(errs), np.degrees(np.array(yerrs))
    print(f'  日志自车走一步（v>1 m/s，{len(errs)} 次，其中限幅 {n_clip}）：后轴位置误差 中位 {np.median(errs):.3f} / p95 {np.percentile(errs, 95):.3f} m，'
          f'航向误差 中位 {np.median(yerrs):.2f}° / p95 {np.percentile(yerrs, 95):.2f}°，路线投影偏差 中位 {np.median(lands):.3f} 帧；'
          f'can vel_x 与日志差 中位 {np.median(n_can):.3f} m/s')
    cm = ~np.array(clips, bool)
    print(f'  其中不限幅的 {cm.sum()} 次：位置误差 中位 {np.median(errs[cm]):.3f} / p95 {np.percentile(errs[cm], 95):.3f} m，航向误差 中位 {np.median(yerrs[cm]):.2f}° / p95 {np.percentile(yerrs[cm], 95):.2f}°')
    check(np.median(errs) < 0.1 and np.percentile(errs, 95) < 0.5, '日志自车走一步落点（中位 <0.1 m，p95 <0.5 m）')
    check(n_bad_cmd == 0, f'路线命令 vs 官方命令：{n_cmd} 帧中不一致 {n_bad_cmd}')
    if full:
        print(f'  日志自车碰撞：{coll} 次，其中有责 {af}，类型 {kinds}')
        check(af <= max(2, 0.02 * len(scenes)), f'日志自车的有责碰撞应几乎为 0（{af} 次 / {len(scenes)} 场景）')


def t_pose(data):
    print('[pose]')
    tr = LogTrack(data, data.scenes[1])
    route = cl.Route(data, tr)
    j = 10
    p = cl.sim_pose(tr, route, j, tr.T[j, :2], tr.yaw[j], float(j), 0.0, float(tr.speed[j]), 0.0, 0.0)
    dq = max(abs(a - b) for a, b in zip(p.e2g_q, tr.infos[j]['ego2global_rotation']))
    dt = max(abs(a - b) for a, b in zip(p.e2g_t, tr.infos[j]['ego2global_translation']))
    check(dq < 1e-12 and dt < 1e-9, f'日志位置 + 日志航向 → 日志位姿（四元数差 {dq:.1e}，平移差 {dt:.1e}）')
    f, lat = route.project(tr.T[j, :2] + np.array([-math.sin(tr.yaw[j]), math.cos(tr.yaw[j])]) * 1.5, j - 1)
    check(abs(lat - 1.5) < 0.05 and abs(f - j) < 0.05, f'左偏 1.5 m 投影 lat={lat:.3f} f={f:.3f}')
    p2 = cl.sim_pose(tr, route, j, tr.T[j, :2], tr.yaw[j] + 0.1, float(j), 0.0, 5.0, 1.0, 0.2)
    check(abs(quaternion_yaw(Quaternion(p2.e2g_q)) - (tr.yaw[j] + 0.1)) < 1e-9, '航向 +0.1 rad')
    cb = synth_can_bus(p2.can_bus_base, p2.e2g_t, Quaternion(p2.e2g_q))
    check(cb[13] == 5.0 and cb[12] == 0.2 and cb[7] == 1.0 and abs(cb[8] - 1.0) < 1e-12 and cb[0] == p2.e2g_t[0], 'can_bus 运动量（vel / rot_rate / accel）与位姿')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true', help='全部 val 场景（否则抽 10 个）')
    a = ap.parse_args()
    from uniad_port.data import UniADValData
    data = UniADValData(num_prefetch=0)
    t_bicycle()
    t_collision()
    t_pose(data)
    scenes = list(range(len(data.scenes))) if a.full else list(range(0, len(data.scenes), 15))
    t_log(data, scenes, a.full)
    print('ALL PASS' if not FAILS else f'FAILED: {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
