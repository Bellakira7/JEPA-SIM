"""仿真自车位姿：沿日志的 LogTrack + 偏离日志的扰动族（横向漂移 / 纵向速度缩放 / 航向偏置）。

约定
----
* 位姿指 ego（nuScenes `ego_pose`，x 前 y 左）在 global 系下的 (t, q)；LiDAR 与相机相对 ego 刚性，`lidar2ego` 取日志值。
* 仿真步与日志关键帧**时间对齐**：第 k 步的时间戳 = 日志第 a+k 个关键帧的时间戳（UniAD 0.5 s 一步）；其他 agent 按日志回放。
* k=0 是锚帧（真实），扰动从 k=1 开始；横向漂移在 `ramp` 个关键帧内按 smoothstep 偏到 δ 后保持。
* δ>0 = 向自车左侧（ego +y）偏。航向跟随偏移后路径的切线：Δψ = atan2(dδ/dt, max(v, v_min))。
* 速度缩放 s：第 k 步位于日志路径上的分数下标 a + s·k（Catmull-Rom 插值 xy、线性插值 z、slerp 旋转）。
* **恒等扰动（δ=0、s=1、航向偏置 0）逐位返回日志位姿**（同一个 Quaternion 构造路径），保证 δ=0 时 img_metas 与
  `UniADValData.get_frame` 逐项相同。

can_bus（BEVFormer 的 18 维：pos3 / quat4 / accel3 / rot_rate3 / vel3 / patch_angle(rad, deg)）
* pos / quat / patch_angle 按仿真位姿重写，写法与 `uniad_port.data.UniADValData._can_bus` 相同（v1.0.1 的 [w,w,w,w] 写法由
  `forward_from_feats` 按 COMPAT 自己处理，这里统一给 rotation.elements）。
* accel / rot_rate / vel（CAN 原始量，ego 系）：横向漂移只给 rot_rate_z 加上 dΔψ/dt，其余沿用日志；速度缩放把 vel、rot_rate
  乘 s、accel 的 x/y 乘 s²（z 含重力不缩放），在相邻关键帧之间线性插值。这是近似，正文里只作为"条件取自仿真位姿"的说明。
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from pyquaternion import Quaternion

from . import common  # noqa: F401


def quaternion_yaw(q: Quaternion) -> float:
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
    return float(np.arctan2(v[1], v[0]))


def synth_can_bus(base: np.ndarray, e2g_t, e2g_q: Quaternion) -> np.ndarray:
    """与 UniADValData._can_bus 同一写法，只是位姿换成给定值。"""
    can_bus = np.array(base, dtype=np.float64).copy()
    can_bus[:3] = e2g_t
    can_bus[3:7] = e2g_q.elements
    patch_angle = quaternion_yaw(e2g_q) / np.pi * 180
    if patch_angle < 0:
        patch_angle += 360
    can_bus[-2] = patch_angle / 180 * np.pi
    can_bus[-1] = patch_angle
    return can_bus


def smoothstep(u):
    u = min(max(u, 0.0), 1.0)
    return 3 * u * u - 2 * u * u * u, 6 * u * (1 - u)          # 值、对 u 的导数


@dataclass
class SimPose:
    k: int                          # 相对锚帧的步数（0 = 锚帧）
    tmpl_idx: int                   # 时间对齐的日志关键帧（全局 info 下标）：提供 timestamp / lidar2img / sample_token
    cmd_idx: int                    # 导航命令取自哪个日志关键帧（速度缩放时 = 路径位置最近的关键帧）
    e2g_t: List[float]              # 原样可交给 Quaternion/np.asarray 的 list（恒等扰动时就是 info 里的 list 本身）
    e2g_q: list
    l2e_t: List[float]
    l2e_q: list
    can_bus_base: np.ndarray        # 18 维，accel/rot_rate/vel 已按扰动修正（pos/quat/angle 在 metas 里再按位姿重写）
    log_f: float                    # 在日志路径上的分数下标（全局 info 下标）
    lat_m: float = 0.0              # 相对日志的横向偏移（m，左正）
    dpsi: float = 0.0               # 相对日志航向（rad）
    identity: bool = True           # 是否逐位等于日志位姿


class LogTrack:
    """一个场景的日志关键帧轨迹。frame_ids 为 UniADValData 的全局 info 下标（场景内时间序）。"""

    def __init__(self, data, scene):
        self.data = data
        self.ids = list(scene['frame_ids'])
        infos = [data.infos[i] for i in self.ids]
        self.infos = infos
        self.n = len(infos)
        self.T = np.array([i['ego2global_translation'] for i in infos], dtype=np.float64)
        self.Q = [Quaternion(i['ego2global_rotation']) for i in infos]
        self.yaw = np.array([quaternion_yaw(q) for q in self.Q])
        self.ts = np.array([i['timestamp'] for i in infos], dtype=np.int64)
        self.can = np.array([i['can_bus'] for i in infos], dtype=np.float64)
        l2e = {(tuple(i['lidar2ego_translation']), tuple(i['lidar2ego_rotation'])) for i in infos}
        assert len(l2e) == 1, f'场景内 lidar2ego 不恒定: {len(l2e)}'
        # 速度：关键帧位置的中心差分（端点单边）
        t = self.ts * 1e-6
        v = np.zeros(self.n)
        for j in range(self.n):
            j0, j1 = max(j - 1, 0), min(j + 1, self.n - 1)
            if j1 > j0:
                v[j] = np.linalg.norm(self.T[j1, :2] - self.T[j0, :2]) / (t[j1] - t[j0])
        self.speed = v
        self.dt = float(np.median(np.diff(t))) if self.n > 1 else 0.5

    # ------------------------------------------------------------------ 插值
    def _interp(self, f: float):
        """分数下标（场景内）处的 (t, q, can18)。整数下标逐位返回日志值。"""
        i0 = int(math.floor(f))
        u = f - i0
        if u == 0.0:
            return list(self.infos[i0]['ego2global_translation']), list(self.infos[i0]['ego2global_rotation']), \
                self.can[i0].copy(), self.Q[i0]
        i1 = min(i0 + 1, self.n - 1)
        im, i2 = max(i0 - 1, 0), min(i0 + 2, self.n - 1)
        p0, p1, p2, p3 = self.T[im, :2], self.T[i0, :2], self.T[i1, :2], self.T[i2, :2]
        xy = 0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)
        z = (1 - u) * self.T[i0, 2] + u * self.T[i1, 2]
        q = Quaternion.slerp(self.Q[i0], self.Q[i1], u)
        can = (1 - u) * self.can[i0] + u * self.can[i1]
        return [float(xy[0]), float(xy[1]), float(z)], list(q.elements), can, q

    # ------------------------------------------------------------------ 扰动
    def anchors(self, stride=4, min_hist=4, horizon=8, max_scale=1.5, min_speed=0.0):
        """可用锚帧（场景内下标）：前面至少 min_hist 帧真实历史，后面路径够 max_scale·horizon 步。

        min_speed：窗口 [a, a+horizon] 内日志最低速度下限（m/s）。横向漂移的航向 = atan2(横向速率, v)，|δ|=2 m、ramp=2 s 时
        峰值横向速率 1.5 m/s；低速下航向会到 50° 以上（原地横摆，物理上不可行，也远超训练分布）。v≥5 时峰值 ≤15.7°，
        对比日志 Δyaw p99=11.4°/步（val 实测）。所有扰动共用同一批锚帧，保证 δ 曲线是配对的。
        """
        need = int(math.ceil(max_scale * horizon))
        out = [a for a in range(min_hist, self.n) if (a - min_hist) % stride == 0 and a + need <= self.n - 1]
        if min_speed > 0:
            out = [a for a in out if self.speed[a:a + horizon + 1].min() >= min_speed]
        return out

    def rollout(self, a: int, family: str, value: float, horizon=8, ramp=4, v_min=1.0) -> List[SimPose]:
        """family: 'lat'（δ, m） | 'speed'（s） | 'yaw'（航向偏置, deg）。返回 k=0..horizon 的 SimPose。"""
        l2e_t = list(self.infos[0]['lidar2ego_translation'])
        l2e_q = list(self.infos[0]['lidar2ego_rotation'])
        out = []
        prev_dpsi = 0.0
        for k in range(horizon + 1):
            i = a + k
            tmpl = self.ids[i]
            if family == 'speed':
                f = a + value * k
                if f > self.n - 1 + 1e-9:
                    raise ValueError(f'speed={value} 超出日志路径：a={a} k={k} n={self.n}')
                t, q_list, can, q = self._interp(f)
                identity = (f == float(i))
                if not identity:
                    can = can.copy()
                    can[7:9] *= value ** 2
                    can[10:13] *= value
                    can[13:16] *= value
                cmd = self.ids[min(int(round(f)), self.n - 1)]
                out.append(SimPose(k, tmpl, cmd, t, q_list, l2e_t, l2e_q, can, float(self.ids[0] + f),
                                   identity=identity))
                continue
            info = self.infos[i]
            r, dr = smoothstep(k / ramp)
            if family == 'lat':
                lat, dpsi_v = value * r, 0.0
                if value != 0.0 and k > 0:
                    latrate = value * dr / ramp / self.dt
                    dpsi_v = math.atan2(latrate, max(self.speed[i], v_min))
            elif family == 'yaw':
                lat, dpsi_v = 0.0, math.radians(value) * r
            else:
                raise ValueError(family)
            if (lat == 0.0 and dpsi_v == 0.0) or k == 0:
                out.append(SimPose(k, tmpl, tmpl, list(info['ego2global_translation']), list(info['ego2global_rotation']),
                                   l2e_t, l2e_q, self.can[i].copy(), float(tmpl), identity=True))
                prev_dpsi = 0.0
                continue
            yaw = self.yaw[i]
            t = (self.T[i] + lat * np.array([-math.sin(yaw), math.cos(yaw), 0.0])).tolist()
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=dpsi_v) * self.Q[i]
            can = self.can[i].copy()
            can[12] += (dpsi_v - prev_dpsi) / self.dt
            prev_dpsi = dpsi_v
            out.append(SimPose(k, tmpl, tmpl, t, list(q.elements), l2e_t, l2e_q, can, float(tmpl),
                               lat_m=lat, dpsi=dpsi_v, identity=False))
        return out


def actions(poses: Sequence[SimPose]) -> np.ndarray:
    """相邻仿真位姿之间的自车 action [Δx, Δy, Δyaw]（前一帧 ego 系，m / rad），长度 len(poses)-1。"""
    out = []
    for p0, p1 in zip(poses[:-1], poses[1:]):
        q0, q1 = Quaternion(p0.e2g_q), Quaternion(p1.e2g_q)
        d = q0.rotation_matrix.T @ (np.asarray(p1.e2g_t) - np.asarray(p0.e2g_t))
        dyaw = (quaternion_yaw(q1) - quaternion_yaw(q0) + math.pi) % (2 * math.pi) - math.pi
        out.append([d[0], d[1], dyaw])
    return np.array(out, dtype=np.float64).reshape(-1, 3)
