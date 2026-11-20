"""仿真位姿 → UniAD 输入帧（img_metas 等）与"理想感知输出"（GT 框、地图栅格）。

    env = OffLogEnv(data)                       # data = uniad_port.data.UniADValData()
    fr  = env.sim_frame(pose)                   # 与 data.get_frame(…, load_images=False) 同结构，位姿换成仿真位姿
    gt  = env.gt(pose)                          # dict(boxes=…, det_name=…, map=dict(lane, drivable, …))

相机相对自车刚性：lidar2img 取时间对齐的日志帧（只依赖相机-激光外参与内参，与自车位姿无关）。
timestamp / scene_token / sample_idx 取时间对齐的日志帧；command 取 pose.cmd_idx 的日志命令。
`pose.identity=True` 时返回的 frame 与 data.get_frame 逐项相同（offlog/selfcheck.py 校验）。
"""
import json
import os
from typing import Dict

import numpy as np
import torch
from pyquaternion import Quaternion

from . import common
from .sim_pose import SimPose, synth_can_bus


def scene_locations() -> Dict[str, str]:
    with open(os.path.join(common.NUSC_TABLES, 'scene.json')) as f:
        scenes = json.load(f)
    with open(os.path.join(common.NUSC_TABLES, 'log.json')) as f:
        logs = {l['token']: l['location'] for l in json.load(f)}
    return {s['token']: logs[s['log_token']] for s in scenes}


def l2g_from_pose(e2g_t, e2g_q, l2e_t, l2e_q):
    """与 UniADValData._l2g 同一写法（float64 计算，float32 输出）。"""
    l2e_r_mat = Quaternion(l2e_q).rotation_matrix
    l2e_t = np.array(l2e_t, dtype=np.float64)
    e2g_r_mat = Quaternion(e2g_q).rotation_matrix
    e2g_t = np.array(e2g_t, dtype=np.float64)
    l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T
    l2g_t = l2e_t @ e2g_r_mat.T + e2g_t
    return l2g_r_mat.astype(np.float32), l2g_t.astype(np.float32)


class OffLogEnv:
    def __init__(self, data, gt_world=None, map_gt=None):
        self.data = data
        self.loc = scene_locations()
        self.gt_world = gt_world
        self.map_gt = map_gt

    def sim_frame(self, pose: SimPose) -> dict:
        fr = self.data.get_frame(pose.tmpl_idx, load_images=False)
        if pose.identity and pose.cmd_idx == pose.tmpl_idx:
            fr['pose'] = pose
            return fr
        m = fr['img_metas']
        l2g_r_mat, l2g_t = l2g_from_pose(pose.e2g_t, pose.e2g_q, pose.l2e_t, pose.l2e_q)
        m['can_bus'] = synth_can_bus(pose.can_bus_base, pose.e2g_t, Quaternion(pose.e2g_q))
        m['l2g_r_mat'] = l2g_r_mat.copy()
        m['l2g_t'] = l2g_t.copy()
        fr['l2g_r_mat'] = torch.from_numpy(l2g_r_mat)
        fr['l2g_t'] = torch.from_numpy(l2g_t)
        if pose.cmd_idx != pose.tmpl_idx:
            _, _, fr['command'] = self.data.planning_label(pose.cmd_idx)
        # GT 规划在偏离日志后没有定义，清掉避免误用
        fr['sdc_planning'] = None
        fr['sdc_planning_mask'] = None
        fr['pose'] = pose
        return fr

    def gt(self, pose: SimPose, with_map=True) -> dict:
        info = self.data.infos[pose.tmpl_idx]
        out = {}
        if self.gt_world is not None:
            out.update(self.gt_world.boxes_in(info['token'], pose.e2g_t, pose.e2g_q, pose.l2e_t, pose.l2e_q))
        if with_map and self.map_gt is not None:
            out['map'] = self.map_gt.raster(self.loc[info['scene_token']], pose.e2g_t, pose.e2g_q, pose.l2e_t, pose.l2e_q)
        return out
