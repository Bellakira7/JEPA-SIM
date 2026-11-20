"""C 表（整段闭环，设计稿 §6）里与臂无关的部分：自车运动学、日志路线（投影 / 导航命令 / 地面高度与姿态）、仿真位姿 → SimPose、
自车足迹、碰撞分类、可行驶区域合规。纯 numpy / shapely，不碰 GPU（单测见 test_closed_loop.py）。

约定（与 offlog 一致）
---------------------
* 位姿 = ego（nuScenes `ego_pose`，后轴中心，x 前 y 左）在 global 下的 (t, q)；LiDAR / 相机相对 ego 刚性，取日志值。
* 仿真第 j 步与日志第 j 个关键帧**时间对齐**（0.5 s 一步）；其他 agent 按日志回放（非反应式）。
* UniAD 的规划路点 `out['plan']`：[6,2]，**当前 LiDAR_TOP 系（x 右 y 前）**，是未来 0.5 s…3 s 的 "SDC 点"（LiDAR 原点下移半个车高，
  `uniad_port.data.SDC_POINT_LIDAR`）的位置。只用第一个路点（0.5 s）。

自车更新：运动学自行车模型（后轴参考），每步常转角 + 常加速度，闭式解（圆弧）；纵向跟踪规划的速度、横向纯跟踪规划的第一个路点
----------------------------------------------------------------------------------------------------------------
1. 路点 → ego 系：p_E = R_l2e·[px, py, z_sdc] + t_l2e；参考点（当前 SDC 点）o = R_l2e·[0, 0, z_sdc] + t_l2e（≈ (0.94, 0)）。
2. 纵向：目标速度 = 规划在 t=0.5 s 处的速度 = ½(|wp0 − o| + |wp1 − wp0|) / 0.5 s；a = (v_target − v0) / 0.5 s，限幅 [A_MIN, A_MAX]，
   速度不为负（减到 0 就停住，不倒车）；第一个路点在参考点后方 = 目标速度 0。
   **不用"恰好走到路点"的常加速度**：那等价于 v1 = 2·v̄ − v0（每步过冲），而 UniAD 的规划速度几乎只看 can_bus 里的自车速度 / 加速度
   （11-07 实测：同一帧换真特征 / 预测特征 / 冻结特征，规划速度差 <0.05 m/s；can_bus 加速度 −3 m/s² → 规划速度 −1.2 m/s），
   过冲的加速度经 can_bus 回灌，速度环每步增益约 1.9，几步就刹停或冲到限幅；按速度跟踪时增益约 0.94。
3. 横向：纯跟踪。前视点 = 规划折线 [o, wp0, wp1, …] 上弧长 Ld = max(|wp0 − o|, L_MIN) 处的点（正常车速下就是第一个路点；
   低速时路点只有几十厘米，横向噪声会被放大成满舵，所以给 L_MIN 的最小前视距离，纯跟踪的标准做法）。后轴目标 g = 前视点 − R(Δψ)·o
   （参考点在后轴前方，转向后跟着转），Δψ 由纯跟踪圆弧给出（κ = 2g_y/|g|²，Δψ = 2·atan2(g_y, g_x)），阻尼不动点迭代到收敛；
   转角限幅 |δ| ≤ MAX_STEER（κ ≤ tan δ / L）。速度与规划一致、不限幅时参考点落在第一个路点上（单测验证）。
4. 位置按圆弧闭式解；高度与俯仰 / 侧倾取日志路线在投影点处的插值（`LogTrack._interp`），航向 = 积分出来的仿真航向。

导航命令：照 `UniADValData.planning_label` 的规则（未来 6 个关键帧的 SDC 点，末个有效点在当前 LiDAR 系 x ≥ 2 → RIGHT，≤ −2 → LEFT，否则 FORWARD），
只是"未来"取**日志路线上投影点之后**的 6 个关键帧、换到**仿真位姿**的 LiDAR 系。仿真位姿等于日志第 i 帧时逐位等于官方命令（单测在全部 val 帧上核对）。

碰撞（NAVSIM / nuPlan 的有责碰撞口径，按本数据集可得的信息简化，正文要写）
---------------------------------------------------------------------
* 自车框：4.084 × 1.85 m，中心在 SDC 点前 0.5 m（= UniAD `planning_metrics` 的自车框）；其他框：GT 标注（世界系），相邻关键帧之间按实例线性插值，
  每 0.1 s 检查一次（自车用本步的圆弧闭式解）。每个实例只记第一次接触。
* 分类：自车速度 < 0.05 m/s → stopped_ego（不算有责）；对方速度 < 0.05 m/s → stopped_track（有责）；自车前沿线段与对方相交 → active_front（有责）；
  后沿相交 → active_rear（不算有责，被追尾）；其余 → active_lateral（只有自车足迹不全在可行驶区域时算有责；nuPlan 还看"跨多条车道"，我们没有车道多边形）。
* 对象分两组报：agent（车辆 / 行人 / 两轮）与 static（barrier / traffic_cone）；`det_name` 为空的类别（动物等）不计。
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import LineString, Polygon

from offlog import common
from offlog.sim_pose import LogTrack, SimPose, quaternion_yaw

DT = 0.5
WHEELBASE = 2.588           # Renault ZOE（nuScenes 采集车）轴距
MAX_STEER = 0.6             # rad（≈34°）
A_MAX, A_MIN = 4.0, -8.0    # m/s²
L_MIN = 2.0                 # 纯跟踪最小前视距离（m）
N_SUB = 5                   # 碰撞检查：每步 5 个子时刻（0.1 s）
EGO_L, EGO_W, EGO_FWD = 4.084, 1.85, 0.5     # UniAD planning_metrics 的自车框（相对 SDC 点前移 0.5 m）
STOP_V = 0.05
CMD_STEPS = 6
SDC_Z = float(-np.float32(0.5) * np.float32(1.56))          # = uniad_port.data.SDC_POINT_LIDAR[2]
AGENT_NAMES = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'pedestrian', 'motorcycle', 'bicycle')
STATIC_NAMES = ('barrier', 'traffic_cone')
CMD_NAMES = {0: 'RIGHT', 1: 'LEFT', 2: 'FORWARD'}


def rot2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


# ============================================================================ 自行车模型
def arc_pose(kappa, s):
    """后轴从原点、朝 +x 出发，沿曲率 kappa 走弧长 s 后的 (x, y, ψ)。"""
    if abs(kappa) < 1e-9:
        return s, 0.0, 0.0
    psi = kappa * s
    return math.sin(psi) / kappa, (1.0 - math.cos(psi)) / kappa, psi


def dist_at(v0, a, t):
    """常加速度 a 下 t 时刻走过的距离与速度（速度减到 0 就停住）。"""
    if a < 0 and v0 + a * t < 0:
        ts = -v0 / a
        return v0 * ts + 0.5 * a * ts * ts, 0.0
    return v0 * t + 0.5 * a * t * t, v0 + a * t


def pursuit(target_e, ref_e):
    """后轴从原点朝 +x 出发，求圆弧使参考点（当前在 ref_e）到达 target_e。返回 (κ, 到达时的弧长)；目标在后方 / 原地返回 (0, 0)。"""
    target_e = np.asarray(target_e, np.float64)
    ref_e = np.asarray(ref_e, np.float64)
    dpsi, kappa, s_arc = 0.0, 0.0, 0.0
    for _ in range(100):                     # 不动点 dpsi = 2·atan2(g_y, g_x)，g = target − R(dpsi)·ref；阻尼迭代到收敛
        g = target_e - rot2(dpsi) @ ref_e
        gx, gy = float(g[0]), float(g[1])
        d2 = gx * gx + gy * gy
        if gx <= 1e-3 or d2 < 1e-8:
            return 0.0, 0.0
        kappa = 2.0 * gy / d2
        th = 2.0 * math.atan2(gy, gx)
        s_arc = th / kappa if abs(kappa) > 1e-9 else math.sqrt(d2)
        if abs(th - dpsi) < 1e-12:
            break
        dpsi = 0.5 * (dpsi + th)
    return kappa, s_arc


def along(pts, d):
    """折线 pts[0]→pts[1]→… 上弧长 d 处的点（超出末端按最后一段外推）。"""
    pts = np.asarray(pts, np.float64)
    for i in range(len(pts) - 1):
        seg = pts[i + 1] - pts[i]
        L = float(np.linalg.norm(seg))
        if d <= L or i == len(pts) - 2:
            return pts[i] + (seg / L * d if L > 1e-9 else 0.0)
        d -= L
    return pts[-1]


def track_plan(v0: float, pts_e, dt: float = DT, n_sub: int = N_SUB, l_min: float = L_MIN) -> dict:
    """ego 系（当前时刻，后轴为原点、x 前 y 左）。pts_e：[1+m, 2]，pts_e[0] = 参考点当前位置，pts_e[1:] = 规划路点（0.5 s 间隔）。
    返回 dict(dx, dy, dpsi, v1, a, v_target, kappa, steer, s, clipped_steer, clipped_acc, sub=[(t, x, y, ψ, v)...]，t = dt·i/n_sub，i=1..n_sub)。"""
    pts = np.asarray(pts_e, np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    forward = float(pts[1, 0] - pts[0, 0]) > 0.0
    v_target = 0.5 * (seg[0] + (seg[1] if len(seg) > 1 else seg[0])) / dt if forward else 0.0
    a_des = (v_target - v0) / dt
    a = max(A_MIN, min(A_MAX, a_des))
    kappa = 0.0
    if forward:
        kappa, _ = pursuit(along(pts, max(float(seg[0]), l_min)), pts[0])
    k_max = math.tan(MAX_STEER) / WHEELBASE
    clipped_steer = abs(kappa) > k_max
    kappa_e = max(-k_max, min(k_max, kappa))
    sub = []
    for i in range(1, n_sub + 1):
        t = dt * i / n_sub
        s, v = dist_at(v0, a, t)
        x, y, psi = arc_pose(kappa_e, s)
        sub.append((t, x, y, psi, v))
    _, x1, y1, psi1, v1 = sub[-1]
    return dict(dx=x1, dy=y1, dpsi=psi1, v1=v1, a=a, v_target=v_target, kappa=kappa_e, steer=math.atan(kappa_e * WHEELBASE),
                s=dist_at(v0, a, dt)[0], clipped_steer=bool(clipped_steer), clipped_acc=bool(a != a_des), sub=sub)


def lidar_frame(info_or_pose):
    """(l2e_r, l2e_t)：lidar → ego。"""
    if isinstance(info_or_pose, SimPose):
        q, t = info_or_pose.l2e_q, info_or_pose.l2e_t
    else:
        q, t = info_or_pose['lidar2ego_rotation'], info_or_pose['lidar2ego_translation']
    return Quaternion(q).rotation_matrix, np.asarray(t, np.float64)


def plan_points_ego(plan, l2e_r, l2e_t):
    """规划路点（LiDAR 系 [m,2]）→ ego 系 xy，前面加上当前参考点：[1+m, 2]。"""
    plan = np.asarray(plan, np.float64).reshape(-1, 2)
    P = np.concatenate([np.zeros((1, 2)), plan], 0)
    P3 = np.concatenate([P, np.full((len(P), 1), SDC_Z)], 1)
    return (P3 @ l2e_r.T + l2e_t)[:, :2]


# ============================================================================ 日志路线
class Route:
    """日志自车轨迹（关键帧）当路线：投影（分数下标）、弧长、导航命令。tr = offlog.sim_pose.LogTrack。"""

    def __init__(self, data, tr: LogTrack):
        self.data, self.tr, self.n = data, tr, tr.n
        self.xy = tr.T[:, :2].copy()
        seg = np.diff(self.xy, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        sdc = np.array([0.0, 0.0, SDC_Z], np.float64)
        self.Pg = []                                    # 各关键帧的 SDC 点（global），算子顺序与 planning_label 相同
        for info in tr.infos:
            l2e_r, l2e_t, e2g_r, e2g_t = data._transforms(info)
            p = l2e_r @ sdc + l2e_t
            p = e2g_r @ p + e2g_t
            self.Pg.append(p)

    def project(self, xy, f_prev: float, back: int = 2, ahead: int = 8) -> Tuple[float, float]:
        """xy 投影到路线（只在 f_prev 附近 [−back, +ahead] 段内找，避免掉头 / 回环路线上跳段）。返回 (分数下标 f, 带符号横向距离（左正）)。
        最后一段沿方向外推（f 可以 > n−1）。"""
        if self.n < 2:
            return 0.0, 0.0
        j0 = max(0, int(math.floor(f_prev)) - back)
        j1 = min(self.n - 2, int(math.floor(f_prev)) + ahead)
        p = np.asarray(xy, np.float64)
        best = (np.inf, 0.0, 0.0)
        for j in range(j0, j1 + 1):
            a, b = self.xy[j], self.xy[j + 1]
            ab = b - a
            den = float(ab @ ab)
            u = 0.0 if den < 1e-12 else float((p - a) @ ab / den)
            u = max(u, 0.0) if j == self.n - 2 else float(np.clip(u, 0.0, 1.0))   # 最后一段向前外推：自车开过日志终点后横向偏离 / 进度仍有定义
            q = a + u * ab
            d = float(np.linalg.norm(p - q))
            if d < best[0]:
                side = float(np.sign(ab[0] * (p - a)[1] - ab[1] * (p - a)[0])) if den >= 1e-12 else 0.0
                best = (d, j + u, side * d)
        return best[1], best[2]

    def arc(self, f: float) -> float:
        j = min(int(math.floor(f)), self.n - 2) if self.n > 1 else 0
        u = f - j
        return float(self.s[j] + u * (self.seg_len[j] if self.n > 1 else 0.0))

    def command(self, pose: SimPose, r: int) -> int:
        """路线下标 r（投影点四舍五入）之后的 ≤6 个关键帧 SDC 点，换到 pose 的 LiDAR 系，按 planning_label 的阈值给命令。"""
        js = list(range(r + 1, min(r + CMD_STEPS, self.n - 1) + 1))
        if not js:
            return 2
        l2e_r_i = Quaternion(pose.l2e_q).rotation_matrix
        l2e_t_i = np.array(pose.l2e_t, dtype=np.float64)
        e2g_r_i = Quaternion(pose.e2g_q).rotation_matrix
        e2g_t_i = np.array(pose.e2g_t, dtype=np.float64)
        p = self.Pg[js[-1]]
        p = e2g_r_i.T @ (p - e2g_t_i)
        p = l2e_r_i.T @ (p - l2e_t_i)
        x_last = p[0]
        if x_last >= 2:
            return 0
        if x_last <= -2:
            return 1
        return 2


# ============================================================================ 仿真位姿
def log_pose(tr: LogTrack, j: int) -> SimPose:
    """日志第 j 个关键帧的恒等 SimPose（与 LogTrack.rollout 的恒等分支同一构造，env.sim_frame 对它逐位返回 get_frame）。"""
    return tr.rollout(j, 'lat', 0.0, horizon=0)[0]


def sim_pose(tr: LogTrack, route: Route, j: int, xy, yaw: float, f: float, lat: float, v: float, a: float, omega: float,
             log_kin: bool = False) -> SimPose:
    """仿真第 j 步（时间对齐日志第 j 帧）的 SimPose。高度、俯仰 / 侧倾取路线投影点 f 处的日志插值；航向 = 仿真航向。
    can_bus 的运动量（ego 系）：vel = [v,0,0]，rot_rate_z = ω，accel_x = a，accel_y = v·ω，其余（accel_z 含重力、roll/pitch 角速度）取时间对齐的日志值；
    pos / quat / patch_angle 由 offlog.metas 按位姿重写（synth_can_bus）。"""
    fc = min(max(f, 0.0), tr.n - 1.0)
    t_f, _, _, q_f = tr._interp(fc)
    q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw - quaternion_yaw(q_f)) * q_f
    can = tr.can[j].copy()
    if not log_kin:                          # log_kin=True（诊断）：运动量取时间对齐的日志帧，只有位姿是仿真的
        can[7] = a
        can[8] = v * omega
        can[12] = omega
        can[13:16] = [v, 0.0, 0.0]
    r = int(round(fc))
    info0 = tr.infos[0]
    return SimPose(k=j, tmpl_idx=tr.ids[j], cmd_idx=tr.ids[r], e2g_t=[float(xy[0]), float(xy[1]), float(t_f[2])],
                   e2g_q=list(q.elements), l2e_t=list(info0['lidar2ego_translation']), l2e_q=list(info0['lidar2ego_rotation']),
                   can_bus_base=can, log_f=float(tr.ids[0] + fc), lat_m=float(lat),
                   dpsi=float((yaw - quaternion_yaw(q_f) + math.pi) % (2 * math.pi) - math.pi), identity=False)


def pose_xy_yaw(pose: SimPose):
    return np.asarray(pose.e2g_t[:2], np.float64), quaternion_yaw(Quaternion(pose.e2g_q))


def advance(tr: LogTrack, route: Route, pose: SimPose, f_prev: float, v0: float, plan: np.ndarray, j_next: int, log_kin: bool = False):
    """由第 j 步的仿真位姿 + UniAD 规划，走 0.5 s 到第 j_next 步。返回 (SimPose, 状态 dict（含世界系子时刻轨迹，碰撞检查用）)。"""
    l2e_r, l2e_t = lidar_frame(pose)
    pts = plan_points_ego(plan, l2e_r, l2e_t)
    st = track_plan(v0, pts)
    xy0, yaw0 = pose_xy_yaw(pose)
    R = rot2(yaw0)
    sub_w = [(t, *(xy0 + R @ np.array([x, y])), yaw0 + psi, v) for t, x, y, psi, v in st['sub']]
    xy1 = np.array(sub_w[-1][1:3])
    yaw1 = sub_w[-1][3]
    f1, lat1 = route.project(xy1, f_prev)
    f1 = max(f1, f_prev - 0.5)              # 自车不倒车；远离路线时（|lat| 十几米）投影会在 [f−2, f+8] 窗口里往回漂，限住
    omega = st['dpsi'] / DT
    p1 = sim_pose(tr, route, j_next, xy1, yaw1, f1, lat1, st['v1'], st['a'], omega, log_kin)
    return p1, dict(f=f1, lat=lat1, v=st['v1'], a=st['a'], v_target=st['v_target'], omega=omega, steer=st['steer'], s=st['s'],
                    clipped_steer=st['clipped_steer'], clipped_acc=st['clipped_acc'], sub_w=sub_w)


def log_substeps(tr: LogTrack, j0: int, j1: int, n_sub: int = N_SUB):
    """日志自车在 j0→j1 之间的子时刻 (t, x, y, ψ, v)（线性插值位置 / 航向，速度取两端平均），给"沿日志"参照行的碰撞检查。"""
    out = []
    p0, p1 = tr.T[j0, :2], tr.T[j1, :2]
    y0 = tr.yaw[j0]
    dy = (tr.yaw[j1] - y0 + math.pi) % (2 * math.pi) - math.pi
    v = float(np.linalg.norm(p1 - p0) / max(1e-6, (tr.ts[j1] - tr.ts[j0]) * 1e-6))
    for i in range(1, n_sub + 1):
        u = i / n_sub
        xy = p0 + u * (p1 - p0)
        out.append((DT * u, float(xy[0]), float(xy[1]), y0 + u * dy, v))
    return out


# ============================================================================ 足迹 / 碰撞 / 可行驶区域
def ego_box(xy, yaw, l2e_t=(0.94, 0.0)):
    """自车框（世界系 xy 多边形）：中心 = 后轴 + R(ψ)·(t_l2e.x + EGO_FWD, t_l2e.y)。返回 (Polygon, 前沿 LineString, 后沿 LineString)。"""
    R = rot2(yaw)
    c = np.asarray(xy, np.float64) + R @ np.array([l2e_t[0] + EGO_FWD, l2e_t[1]])
    hl, hw = EGO_L / 2, EGO_W / 2
    loc = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    pts = c + loc @ R.T
    return Polygon(pts), LineString([pts[0], pts[1]]), LineString([pts[2], pts[3]])


def box_poly(cx, cy, l, w, yaw):
    R = rot2(yaw)
    loc = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    return Polygon(np.array([cx, cy]) + loc @ R.T)


def world_boxes(gtw, token):
    """GT 标注（世界系）：dict(inst -> (cx, cy, l, w, yaw, name))，只保留 agent / static 两组。"""
    d = gtw.data[token]
    out = {}
    for i, name in enumerate(d['det_name']):
        if name not in AGENT_NAMES and name not in STATIC_NAMES:
            continue
        q = Quaternion(d['rot_g'][i])
        w, l = d['wlh'][i, 0], d['wlh'][i, 1]
        out[d['instance'][i]] = (float(d['center_g'][i, 0]), float(d['center_g'][i, 1]), float(l), float(w), quaternion_yaw(q), name)
    return out


def collisions_interval(gtw, tok0, tok1, sub, drivable_ok_end: bool, seen: set, l2e_t=(0.94, 0.0)) -> List[dict]:
    """关键帧 tok0→tok1 之间（自车子时刻轨迹 sub = [(t, x, y, ψ, v)], t∈(0, DT]）的新碰撞。seen：已记过的实例（就地更新）。
    对方在两个关键帧都有标注才插值；只在一端有的实例只在那一端（t=0 / t=DT）检查。"""
    b0, b1 = world_boxes(gtw, tok0), world_boxes(gtw, tok1)
    out = []
    for (t, x, y, psi, v) in sub:
        u = t / DT
        ego, front, rear = ego_box((x, y), psi, l2e_t)
        for inst in set(b0) | set(b1):
            if inst in seen:
                continue
            if inst in b0 and inst in b1:
                a, b = b0[inst], b1[inst]
                dyaw = (b[4] - a[4] + math.pi) % (2 * math.pi) - math.pi
                cx, cy = a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])
                box = (cx, cy, a[2], a[3], a[4] + u * dyaw, a[5])
                ov = math.hypot(b[0] - a[0], b[1] - a[1]) / DT
            elif inst in b1 and u >= 1.0 - 1e-9:
                box, ov = b1[inst], float('nan')
            else:
                continue
            if math.hypot(box[0] - x, box[1] - y) > 12.0:
                continue
            poly = box_poly(*box[:5])
            if not ego.intersects(poly):
                continue
            seen.add(inst)
            if v < STOP_V:
                kind = 'stopped_ego'
            elif ov == ov and ov < STOP_V:
                kind = 'stopped_track'
            elif front.intersects(poly):
                kind = 'active_front'
            elif rear.intersects(poly):
                kind = 'active_rear'
            else:
                kind = 'active_lateral'
            at_fault = kind in ('stopped_track', 'active_front') or (kind == 'active_lateral' and not drivable_ok_end)
            out.append(dict(t=round(t, 3), inst=inst, name=box[5], group='static' if box[5] in STATIC_NAMES else 'agent',
                            kind=kind, at_fault=bool(at_fault), ego_v=round(v, 3), other_v=round(ov, 3) if ov == ov else None))
    return out


def footprint_on_drivable(drivable, half=51.2, res=0.512) -> Tuple[bool, float]:
    """自车框（LiDAR 系：x∈±W/2，y∈[EGO_FWD−L/2, EGO_FWD+L/2]，SDC 点≈LiDAR 原点的 xy）4 角 + 4 边中点是否全在 GT drivable 栅格内。
    返回 (全在, 在内的点占比)。栅格约定同 runner.SEG_GRID：row = floor((y+51.2)/0.512)，col = floor((x+51.2)/0.512)。"""
    hw, y0, y1 = EGO_W / 2, EGO_FWD - EGO_L / 2, EGO_FWD + EGO_L / 2
    ym = 0.5 * (y0 + y1)
    pts = np.array([[-hw, y0], [hw, y0], [hw, y1], [-hw, y1], [0, y0], [0, y1], [-hw, ym], [hw, ym]])
    col = np.floor((pts[:, 0] + half) / res).astype(int)
    row = np.floor((pts[:, 1] + half) / res).astype(int)
    vals = drivable[np.clip(row, 0, drivable.shape[0] - 1), np.clip(col, 0, drivable.shape[1] - 1)] > 0.5
    return bool(vals.all()), float(vals.mean())
