"""B 表：相机空间 layout 按仿真位姿重投影（设计稿 §3.3 条件、§5.2 "layout 中的框和车道在仿真位姿下重新投影"）。

    rend = SimLayoutRenderer(data.infos)                 # data = uniad_port.data.UniADValData()
    poses = LogTrack(data, scene).rollout(a, 'lat', 2.0) # k=0..8 的 SimPose（offlog.sim_pose）
    L = rend.render(poses[k])                            # uint8 [6,3,120,200]：预测第 k 步特征时的 layout 条件
    cond = rend.rollout_conditions(poses)                # dict(actions=[K,3], layouts=[K,6,3,H,W])，与 a_eval.ARSource.rollout 的条件约定一致

不复制渲染代码：在仿真位姿下构造一个**合成 nusc 帧**（`sim_frame`），然后原样调训练用的 `feat/nusc_data.render_layout`（框的 filled 编码、
地图折线的 `_draw_map`、画布几何都是同一份），所以 δ=0 时与训练口径**逐字节相同**，偏移时唯一新增的逻辑就是下面的位姿合成。

几何约定
--------
* 其他 agent 按日志回放：第 k 步的框取**时间对齐的日志关键帧**（`pose.tmpl_idx`）的世界系标注（与 `offlog.gt_world` 同一口径），
  变到仿真自车系后投影；类别、可见性筛选（lidar+radar 点数 > 0）与训练一致（用 `nusc_data` 的框，不是 GTWorld 的评测 GT）。
* 相机相对自车刚性。日志里相机与激光不同时刻采样（各相机 ego_pose 不同，差几十 ms × 车速），训练 layout 用的是**各相机自己时刻**的 ego 位姿。
  仿真里没有逐相机时间，所以把这个"激光时刻 → 相机时刻"的相对运动保留下来：``E_cam_sim = E_sim · inv(E_lidar_log) · E_cam_log``，
  δ=0（``E_sim == E_lidar_log``）时恰好还原为 ``E_cam_log``，与训练口径一致。
* 地图：折线取**路径位置最近的日志关键帧**（`pose.cmd_idx`；横向/航向扰动时 = tmpl，速度缩放时才与 tmpl 不同），
  它们在该关键帧 LIDAR 自车系（±51.2 m 方形裁剪），经 global 变到仿真相机系。裁剪边界是日志自车周围的，
  所以仿真自车偏出后，画面最远处（>~50 m）靠近裁剪边的车道会缺一小段——扰动 ≤2 m 时可忽略，速度 ×1.5 时已用最近关键帧补偿。
* 恒等位姿（`np.allclose(E_sim, E_lidar_log, atol=1e-9)`）直接用日志的相机位姿，不经过 inv / 矩阵乘，保证逐字节相同。
"""
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from pyquaternion import Quaternion

from . import common
from .sim_pose import SimPose

_FEAT = os.path.join(common.CODE_ROOT, 'feat')
if _FEAT not in sys.path:
    sys.path.insert(0, _FEAT)
import nusc_data as nd  # noqa: E402
from nusc_map import MapVectors  # noqa: E402

IDENTITY_ATOL = 1e-9    # 仿真位姿与日志激光时刻位姿的差小于 1 nm 就当作恒等


def pose_to_matrix(pose: SimPose) -> np.ndarray:
    """SimPose（ego 在 global 下的 t, q）→ 4×4 ego2global（float64）。"""
    T = np.eye(4)
    T[:3, :3] = Quaternion(pose.e2g_q).rotation_matrix
    T[:3, 3] = np.asarray(pose.e2g_t, dtype=np.float64)
    return T


class SimLayoutRenderer:
    """infos：`UniADValData.infos`（全局 info 下标 → token），SimPose 的 tmpl_idx / cmd_idx 就是这个下标。"""

    def __init__(self, infos, nd_index=None, map_vecs=None):
        self.infos = infos
        self.idx = nd_index if nd_index is not None else nd.build_index()
        self.map_vecs = map_vecs if map_vecs is not None else MapVectors()

    def nd_frame(self, info_idx: int) -> dict:
        return self.idx['frames'][self.infos[info_idx]['token']]

    def sim_frame(self, pose: SimPose, force_compose: bool = False) -> Tuple[dict, dict]:
        """返回 (fr_sim, fr_map)：fr_sim = 仿真位姿下的合成 nusc 帧（可直接喂 `nd.render_layout` / `nd.boxes_in_ego`），
        fr_map = 取地图折线用的日志关键帧。force_compose=True 时即使是恒等位姿也走矩阵合成（只给自检用）。"""
        fr = self.nd_frame(pose.tmpl_idx)
        E_sim = pose_to_matrix(pose)
        E_l = fr['ego2global']
        same = (not force_compose) and bool(np.allclose(E_sim, E_l, rtol=0.0, atol=IDENTITY_ATOL))
        if same:
            return fr, self.nd_frame(pose.cmd_idx)
        rel = np.linalg.inv(E_l)
        cams = {ch: dict(c, ego2global=E_sim @ rel @ c['ego2global']) for ch, c in fr['cams'].items()}
        return dict(fr, ego2global=E_sim, cams=cams), self.nd_frame(pose.cmd_idx)

    def render(self, pose: SimPose, with_map: bool = True, force_compose: bool = False) -> np.ndarray:
        """仿真位姿下的 6 路 filled layout，uint8 [6,3,H,W]。with_map=False = "不画地图"变体（只画框）。"""
        fr_sim, fr_map = self.sim_frame(pose, force_compose)
        mv = self.map_vecs.for_frame(fr_map) if with_map else None
        return nd.render_layout(fr_sim, map_vec=mv)

    def render_cond(self, pose_prev: SimPose, pose_tgt: SimPose, with_map: bool = True, assoc: bool = False,
                    force_compose: bool = False) -> np.ndarray:
        """预测器条件（``nd.render_layout_cond`` 的仿真位姿版）：assoc=False 时就是 ``render(pose_tgt, with_map)``，uint8 [6,3,H,W]；
        assoc=True 时 [6,6,H,W]，后 3 通道 = 关联通道：上一步（pose_prev 时间对齐的日志帧）的框画进**仿真位姿下**目标帧的相机，
        只画两帧都在场的实例。pose_prev 只用它的 ``tmpl_idx``（其他 agent 按日志回放，与自车位姿无关）；恒等位姿时与训练的
        ``nd.render_layout_cond(kf[i], kf[i+1], assoc=True)`` 逐字节相同（test_layout_reproject 检查）。"""
        fr_sim, fr_map = self.sim_frame(pose_tgt, force_compose)
        mv = self.map_vecs.for_frame(fr_map) if with_map else None
        return nd.render_layout_cond(self.nd_frame(pose_prev.tmpl_idx), fr_sim, map_vec=mv, assoc=assoc)

    def rollout_conditions(self, poses: List[SimPose], with_map: bool = True) -> Dict[str, np.ndarray]:
        """poses = k=0..K 的仿真位姿（k=0 为锚帧）。返回第 i 步（i=0..K-1，预测帧 i+1）的条件：
        actions[i] = poses[i]→poses[i+1] 的自车相对位姿 [dx,dy,dyaw]（前一帧 ego 系，训练口径，见 ``sim_relative_actions``）；
        layouts[i] = render(poses[i+1])。与 a_eval.ARSource.rollout 的约定一致（窗口第 i 帧的 action = nds[i]→nds[i+1]，layout = nds[i+1]）。"""
        return dict(actions=sim_relative_actions(poses),
                    layouts=np.stack([self.render(p, with_map) for p in poses[1:]]))


def sim_relative_actions(poses: List[SimPose]) -> np.ndarray:
    """相邻仿真位姿的自车相对位姿 [dx, dy, dyaw]（前一帧 ego 系），float32 [len(poses)-1, 3]。

    **与训练 / A 表的 ``nd.relative_action`` 同一公式**：T = inv(E0)·E1，dyaw = atan2(T[1,0], T[0,0])（相对旋转的 yaw）。
    不要用 ``offlog.sim_pose.actions``：它的 dyaw 是两个**全局航向**之差，自车有俯仰 / 侧倾时与训练口径差最多 ~8e-4 rad
    （11-06 实测，dx / dy 一致；本函数在恒等位姿上与 nd.relative_action 差 3e-12）。差得不多，但 B 表条件应与训练同口径。
    """
    E = [pose_to_matrix(p) for p in poses]
    out = []
    for e0, e1 in zip(E[:-1], E[1:]):
        T = np.linalg.inv(e0) @ e1
        out.append([T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0])])
    return np.asarray(out, np.float32).reshape(-1, 3)
