"""仅用 json 表（不依赖 NuScenes 类/map.json）提取 val 场景的 GT 3D 框（LIDAR_TOP 系），用于检测 sanity。"""
import featsim_paths as _release_paths
import json
import os
import pickle
from collections import defaultdict

import numpy as np
from pyquaternion import Quaternion

ROOT = _release_paths.NUSC_TABLES
CACHE = str(_release_paths.Path(_release_paths.BENCH) / 'uniad_port/sanity/_gt_boxes_cache.pkl')


def _load(name):
    with open(os.path.join(ROOT, name + '.json')) as f:
        return json.load(f)


class GTBoxes:
    """gt = GTBoxes(); gt.get(sample_token) -> list of dict(center[3], wlh[3], yaw, category, num_lidar_pts, token)
    center/yaw 在 LIDAR_TOP 系；center 是 box 几何中心（gravity center）。"""

    def __init__(self, sample_tokens):
        self.sample_tokens = set(sample_tokens)
        if os.path.exists(CACHE):
            c = pickle.load(open(CACHE, 'rb'))
            if self.sample_tokens <= set(c) and 'center_global' in (c[next(iter(self.sample_tokens))] or [{'center_global': 0}])[0] if c[next(iter(self.sample_tokens))] else False:
                self.data = c
                return
        sample = {s['token']: s for s in _load('sample') if s['token'] in self.sample_tokens}
        sd_by_tok = {}
        lidar_sd = {}
        for sd in _load('sample_data'):
            if sd['sample_token'] in sample and sd['channel'] == 'LIDAR_TOP' if 'channel' in sd else False:
                pass
        calib = {c['token']: c for c in _load('calibrated_sensor')}
        pose = {p['token']: p for p in _load('ego_pose')}
        sensor = {s['token']: s['channel'] for s in _load('sensor')}
        calib_chan = {t: sensor[c['sensor_token']] for t, c in calib.items()}
        for sd in _load('sample_data'):
            if sd['sample_token'] in sample and sd['is_key_frame'] and calib_chan[sd['calibrated_sensor_token']] == 'LIDAR_TOP':
                lidar_sd[sd['sample_token']] = sd
        cat = {c['token']: c['name'] for c in _load('category')}
        inst = {i['token']: cat[i['category_token']] for i in _load('instance')}
        anns = defaultdict(list)
        for a in _load('sample_annotation'):
            if a['sample_token'] in sample:
                anns[a['sample_token']].append(a)
        self.data = {}
        for st in self.sample_tokens:
            sd = lidar_sd[st]
            cs = calib[sd['calibrated_sensor_token']]
            ps = pose[sd['ego_pose_token']]
            q_l2e = Quaternion(cs['rotation']); t_l2e = np.array(cs['translation'])
            q_e2g = Quaternion(ps['rotation']); t_e2g = np.array(ps['translation'])
            out = []
            for a in anns[st]:
                c = np.array(a['translation']) - t_e2g
                c = q_e2g.inverse.rotate(c) - t_l2e
                c = q_l2e.inverse.rotate(c)
                q = q_l2e.inverse * q_e2g.inverse * Quaternion(a['rotation'])
                out.append(dict(center=c, wlh=np.array(a['size']), yaw=q.yaw_pitch_roll[0],
                                category=inst[a['instance_token']], num_lidar_pts=a['num_lidar_pts'],
                                token=a['token'], instance=a['instance_token'],
                                center_global=np.array(a['translation'])))
            self.data[st] = out
            self.data[st + '_pose'] = dict(q_l2e=q_l2e.elements, t_l2e=t_l2e, q_e2g=q_e2g.elements, t_e2g=t_e2g)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        pickle.dump(self.data, open(CACHE, 'wb'))

    def get(self, sample_token):
        return self.data[sample_token]

    def global_to_lidar(self, sample_token, p_global):
        """把 global 系 3D 点转到 sample_token 的 LIDAR_TOP 系。"""
        ps = self.data[sample_token + '_pose']
        q_l2e, q_e2g = Quaternion(ps['q_l2e']), Quaternion(ps['q_e2g'])
        c = np.asarray(p_global) - ps['t_e2g']
        c = q_e2g.inverse.rotate(c) - ps['t_l2e']
        return q_l2e.inverse.rotate(c)
