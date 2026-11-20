"""世界系 GT 框（nuScenes 标注，不响应自车）→ 任意仿真自车位姿下的 LiDAR_TOP 系。

输出格式与 `UniADRunner.step` 的 boxes 一致（PORTING_NOTES §7）::

    [x, y, z_bottom, l, w, h, yaw, vx, vy]   LiDAR_TOP 系（x 右 y 前），yaw 绕 +z 逆时针，速度是**对地**速度换到 LiDAR 轴

几何与 `uniad_port/uniad_port/gt_boxes.py` 相同（那份只给日志位姿、没有速度）；速度照 devkit `NuScenes.box_velocity`
（前后标注差分，max_time_diff=1.5 s，前后都有时放宽到 3 s，超时记 NaN）。
类别照 devkit `category_to_detection_name` 映射到检测 10 类，其余（动物、静态物体等）记 None。

首次构造会读 737 MB 的 sample_annotation.json（约 1 分钟、数 GB 内存），只把 val 用到的部分缓存到
`offlog/cache/gt_world_<n>.pkl`（几十 MB）。
"""
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, Iterable

import numpy as np
from pyquaternion import Quaternion

from . import common

try:
    from nuscenes.eval.detection.utils import category_to_detection_name
except Exception:  # pragma: no cover - devkit 缺失时的等价实现
    _MAP = {'movable_object.barrier': 'barrier', 'vehicle.bicycle': 'bicycle', 'vehicle.bus.bendy': 'bus',
            'vehicle.bus.rigid': 'bus', 'vehicle.car': 'car', 'vehicle.construction': 'construction_vehicle',
            'vehicle.motorcycle': 'motorcycle', 'human.pedestrian.adult': 'pedestrian',
            'human.pedestrian.child': 'pedestrian', 'human.pedestrian.construction_worker': 'pedestrian',
            'human.pedestrian.police_officer': 'pedestrian', 'movable_object.trafficcone': 'traffic_cone',
            'vehicle.trailer': 'trailer', 'vehicle.truck': 'truck'}

    def category_to_detection_name(name):
        return _MAP.get(name)


def _load(name):
    with open(os.path.join(common.NUSC_TABLES, name + '.json')) as f:
        return json.load(f)


class GTWorld:
    """gtw = GTWorld(sample_tokens); gtw.boxes_in(sample_token, e2g_t, e2g_q, l2e_t, l2e_q) -> dict(boxes[N,9], names, ...)"""

    def __init__(self, sample_tokens: Iterable[str], cache_path: str = None):
        tokens = sorted(set(sample_tokens))
        cache_path = cache_path or os.path.join(common.CACHE_DIR, f'gt_world_{len(tokens)}.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:   # 本模块自己写的缓存
                data = pickle.load(f)
            if set(tokens) <= set(data):
                self.data = data
                return
        self.data = self._build(tokens)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = cache_path + f'.tmp{os.getpid()}'
        with open(tmp, 'wb') as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_path)

    @staticmethod
    def _build(tokens):
        want = set(tokens)
        sample_ts = {s['token']: s['timestamp'] for s in _load('sample')}
        cat = {c['token']: c['name'] for c in _load('category')}
        inst_cat = {i['token']: cat[i['category_token']] for i in _load('instance')}
        anns_all = _load('sample_annotation')
        # 速度要用 prev/next 标注；prev/next 一定在同一场景里，所以只保留 want 里的样本就够
        by_tok = {a['token']: a for a in anns_all if a['sample_token'] in want}
        del anns_all
        per_sample = defaultdict(list)
        for a in by_tok.values():
            per_sample[a['sample_token']].append(a)

        def velocity(a, max_time_diff=1.5):
            has_prev, has_next = a['prev'] != '', a['next'] != ''
            if not has_prev and not has_next:
                return np.array([np.nan, np.nan])
            first = by_tok[a['prev']] if has_prev else a
            last = by_tok[a['next']] if has_next else a
            pos_diff = np.array(last['translation']) - np.array(first['translation'])
            time_diff = 1e-6 * (sample_ts[last['sample_token']] - sample_ts[first['sample_token']])
            if has_prev and has_next:
                max_time_diff *= 2
            if time_diff > max_time_diff:
                return np.array([np.nan, np.nan])
            return pos_diff[:2] / time_diff

        data = {}
        for st in tokens:
            anns = sorted(per_sample.get(st, []), key=lambda a: a['token'])
            n = len(anns)
            data[st] = dict(
                center_g=np.array([a['translation'] for a in anns], dtype=np.float64).reshape(n, 3),
                wlh=np.array([a['size'] for a in anns], dtype=np.float64).reshape(n, 3),
                rot_g=np.array([a['rotation'] for a in anns], dtype=np.float64).reshape(n, 4),
                vel_g=np.array([velocity(a) for a in anns], dtype=np.float64).reshape(n, 2),
                category=[inst_cat[a['instance_token']] for a in anns],
                det_name=[category_to_detection_name(inst_cat[a['instance_token']]) for a in anns],
                num_lidar_pts=np.array([a['num_lidar_pts'] for a in anns], dtype=np.int64),
                instance=[a['instance_token'] for a in anns],
                ann_token=[a['token'] for a in anns],
            )
        return data

    def boxes_in(self, sample_token: str, e2g_t, e2g_q, l2e_t, l2e_q) -> Dict:
        """把 sample_token 时刻的世界系标注换到给定位姿的 LiDAR_TOP 系。e2g_q/l2e_q: [w,x,y,z] 或 Quaternion。"""
        d = self.data[sample_token]
        e2g_r = Quaternion(e2g_q).rotation_matrix
        l2e_r = Quaternion(l2e_q).rotation_matrix
        e2g_t = np.asarray(e2g_t, dtype=np.float64)
        l2e_t = np.asarray(l2e_t, dtype=np.float64)
        g2l = l2e_r.T @ e2g_r.T                                   # global -> lidar 旋转
        c = (d['center_g'] - e2g_t) @ e2g_r                      # = e2g_r.T @ (p - t)
        c = (c - l2e_t) @ l2e_r                                   # = l2e_r.T @ (p - t)
        n = len(c)
        # yaw 与 mmdet3d nuscenes_converter / gt_boxes.py 同一写法：LiDAR 系四元数的 yaw_pitch_roll[0]
        q_g2l = Quaternion(l2e_q).inverse * Quaternion(e2g_q).inverse
        yaw = np.array([(q_g2l * Quaternion(d['rot_g'][i])).yaw_pitch_roll[0] for i in range(n)])
        v3 = np.concatenate([d['vel_g'], np.zeros((n, 1))], axis=1)
        v = v3 @ g2l.T
        w, l, h = d['wlh'][:, 0], d['wlh'][:, 1], d['wlh'][:, 2]
        boxes = np.stack([c[:, 0], c[:, 1], c[:, 2] - h / 2, l, w, h, yaw, v[:, 0], v[:, 1]], axis=1).astype(np.float64)
        return dict(boxes=boxes.reshape(n, 9), det_name=list(d['det_name']), category=list(d['category']),
                    num_lidar_pts=d['num_lidar_pts'].copy(), instance=list(d['instance']))
