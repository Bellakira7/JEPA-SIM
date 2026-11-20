"""地图 layout 的矢量缓存（设计稿 §3.3：nuScenes map expansion 的车道线与道路边界，投到相机画成 layout）。

``MapGT.vectors``（`offlog/map_gt.py`，照 UniAD 的 VectorizedLocalMap 移植）对每个关键帧给出 3 类折线，**在该帧 LIDAR 自车系**
（x 前、y 左，±51.2 m 方形，沿线每 1 m 一个点）：0 = road_divider ∪ lane_divider，1 = ped_crossing，2 = contours（道路边界 = road_segment ∪ lane 的外轮廓）。
这正是 UniAD 地图头训练/评测所用的 lane GT（lane = divider ∪ crossing ∪ contour），所以"完整条件"变体画的就是 UniAD 要分割的那几类线。

每次调用约 0.17 s，所以一次性算好存成每场景一个 npz：``feat/nusc_map_cache/<scene>.npz``
  pts   float32 [P,2]    全部折线点（该帧 LIDAR 自车系）
  poff  int64  [M+1]     折线 m 的点区间
  ptype int8   [M]
  foff  int64  [n+1]     关键帧 i 的折线区间（按关键帧顺序）
  E     float64[n,4,4]   关键帧 i 的 LIDAR 自车 → global（与 ``nusc_index`` 的 ``ego2global`` 相同）
训练/评测只读这些 npz（不依赖 shapely / nuscenes-devkit / pyquaternion）；缓存只有构建时才需要它们。

用法：``python nusc_map.py [--workers 48]``（约 2–3 min，纯 CPU）。
"""
import featsim_paths as _release_paths
import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nusc_data as nd

MAP_CACHE = os.environ.get('NUSC_MAP_CACHE') or str(_release_paths.Path(_release_paths.BENCH) / 'feat/nusc_map_cache')
TABLES = _release_paths.NUSC_TABLES
BENCH = _release_paths.BENCH

_G = {}


def _init_worker():
    sys.path.insert(0, str(_release_paths.CODE_ROOT))
    from offlog import common  # noqa: F401  (descartes 桩)
    from offlog.map_gt import MapGT
    _G['mg'] = MapGT()
    _G['idx'] = nd.build_index()
    from nusc_sweeps import build_sweep_index
    _G['sw'] = build_sweep_index()


def _build_scene(args):
    name, location = args
    path = f'{MAP_CACHE}/{name}.npz'
    if os.path.exists(path):
        return name, 0, 0.0
    mg, kf, lid = _G['mg'], _G['idx']['scenes'][name], _G['sw']['scenes'][name]['LIDAR_TOP']
    t0 = time.time()
    pts, poff, ptype, foff, E = [], [0], [], [0], []
    for i, fr in enumerate(kf):
        pose = lid['pose'][lid['kpos'][i]]  # [tx,ty,tz,qw,qx,qy,qz]
        for v in mg.vectors(location, pose[:3], pose[3:7]):
            p = v['pts'][:v['pts_num']].astype(np.float32)
            if len(p) < 2:
                continue
            pts.append(p)
            ptype.append(v['type'])
            poff.append(poff[-1] + len(p))
        foff.append(len(ptype))
        E.append(fr['ego2global'])
        assert np.allclose(nd.make_T(nd.quat_to_R(pose[3:7]), pose[:3]), fr['ego2global'], atol=1e-6), (name, i)
    tmp = path + '.tmp.npz'
    np.savez(tmp, pts=np.concatenate(pts).astype(np.float32), poff=np.array(poff, np.int64), ptype=np.array(ptype, np.int8),
             foff=np.array(foff, np.int64), E=np.array(E, np.float64))
    os.replace(tmp, path)
    return name, len(kf), time.time() - t0


class MapVectors:
    """读缓存：``for_frame(fr)`` → (E_k, [(type, pts[n,2] 该关键帧 LIDAR 自车系), ...])。
    相位增广帧（``nusc_sweeps``）带 ``map_idx`` = 最近的关键帧下标；原关键帧用 ``idx``。每个 worker 里按场景懒加载，LRU 8 个场景。"""

    def __init__(self, cache_dir=None, max_scenes=8):
        self.dir = cache_dir or MAP_CACHE
        self.max = max_scenes
        self._c = collections.OrderedDict()

    def _scene(self, name):
        if name in self._c:
            self._c.move_to_end(name)
            return self._c[name]
        z = np.load(f'{self.dir}/{name}.npz')
        d = {k: z[k] for k in z.files}
        self._c[name] = d
        while len(self._c) > self.max:
            self._c.popitem(last=False)
        return d

    def for_frame(self, fr):
        d = self._scene(fr['scene'])
        k = int(fr.get('map_idx', fr['idx']))
        polys = []
        for m in range(d['foff'][k], d['foff'][k + 1]):
            polys.append((int(d['ptype'][m]), d['pts'][d['poff'][m]:d['poff'][m + 1]]))
        return d['E'][k], polys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--scenes', default='', help='逗号分隔的场景名（冒烟）；默认全部 850 个')
    a = ap.parse_args()
    os.makedirs(MAP_CACHE, exist_ok=True)
    scene = {x['name']: x for x in json.load(open(f'{TABLES}/scene.json'))}
    log = {x['token']: x for x in json.load(open(f'{TABLES}/log.json'))}
    names = a.scenes.split(',') if a.scenes else sorted(scene)
    jobs = [(n, log[scene[n]['log_token']]['location']) for n in names]
    t0 = time.time()
    done = nfr = 0
    with mp.get_context('fork').Pool(a.workers, initializer=_init_worker) as pool:
        for name, n, dt in pool.imap_unordered(_build_scene, jobs):
            done += 1
            nfr += n
            if done % 50 == 0 or done == len(jobs):
                print(f'[map cache] {done}/{len(jobs)} 场景，{nfr} 帧，{time.time() - t0:.0f}s', flush=True)
    print(f'完成，写入 {MAP_CACHE}')


if __name__ == '__main__':
    main()
