"""按任意自车位姿生成与 UniAD 训练 GT 一致的 200×200 地图栅格（lane = divider∪crossing∪contour，drivable）。

移植自（只读）`downloads/UniAD/projects/mmdet3d_plugin/datasets/`：
  * `data_utils/vector_map.py::VectorizedLocalMap.gen_vectorized_samples`  —— 线类（things）矢量，**以 ego 原点、ego yaw 为中心**
  * `data_utils/rasterize.py::preprocess_map`                              —— 矢量 → 实例栅格（thickness=2、overlap_filter）
  * `data_utils/data_utils.py::obtain_map_info`                            —— drivable 栅格，**以 LiDAR 原点、LiDAR yaw 为中心**
  * `nuscenes_e2e_dataset.py::get_data_info` 里的 `np.rot90(k=-1)` / `np.flip(axis=1)`
  * `uniad/dense_heads/panseg_head.py::forward_test` 的评测口径：lanes_gt = 三类 things 并集，drivable_gt = 最后一个 mask

things 与 drivable 的栅格中心不同（ego 原点 vs LiDAR 原点，nuScenes 上相差约 0.94 m ≈ 2 格），是 UniAD 自己的写法，原样保留：
UniAD 就是在这套 GT 上训练和评测的。

只为 shapely 2 做了必要改动（原代码在 shapely 1.x 上写成，2.x 下会直接报错），数值语义不变：
  1. `LinearRing.coords = ...` 原地赋值 → 新建 LinearRing；
  2. 迭代 MultiLineString → `.geoms`；
  3. `intersection` 得到 GeometryCollection 时只取其中的线段（shapely 1.x 下原代码遇到这种情况会抛 NotImplementedError，
     所以训练数据里不会出现；这里计数到 `self.n_geomcollection` 便于核查）。

输出数组约定 = runner 的 seg：arr[row=y_lidar(前), col=x_lidar(右)]，±51.2 m，0.512 m/格。
"""
import math
from typing import Dict, Tuple

import cv2
import numpy as np
from pyquaternion import Quaternion
from shapely import affinity, ops
from shapely.geometry import LinearRing, LineString, MultiLineString, MultiPolygon, box

from . import common  # noqa: F401  (descartes 桩)
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer  # noqa: E402
from nuscenes.eval.common.utils import quaternion_yaw  # noqa: E402

MAPS = ['boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth', 'singapore-queenstown']
CLASS2LABEL = {'road_divider': 0, 'lane_divider': 0, 'ped_crossing': 1, 'contours': 2, 'others': -1}
PATCH_SIZE = (102.4, 102.4)
CANVAS_SIZE = (200, 200)
THICKNESS = 2          # canvas 200 时 UniAD 用 2
ANGLE_CLASS = 36


def _iter_lines(g):
    """把 intersection 结果拆成 LineString 列表（含 shapely 2 兼容处理）。返回 (lines, 是否来自 GeometryCollection)。"""
    if g.is_empty:
        return [], False
    if g.geom_type == 'LineString':
        return [g], False
    if g.geom_type == 'MultiLineString':
        return list(g.geoms), False
    if g.geom_type == 'GeometryCollection':
        out = []
        for p in g.geoms:
            out.extend(_iter_lines(p)[0])
        return out, True
    return [], False        # Point / MultiPoint：原代码 mask_for_lines 里 len(coords)<2 也会跳过


class MapGT:
    def __init__(self, map_root: str = common.NUSC_MAP_ROOT, patch_size=PATCH_SIZE, canvas_size=CANVAS_SIZE):
        self.patch_size = tuple(patch_size)
        self.canvas_size = tuple(canvas_size)
        self.nusc_maps = {loc: NuScenesMap(dataroot=map_root, map_name=loc) for loc in MAPS}
        self.explorer = {loc: NuScenesMapExplorer(m) for loc, m in self.nusc_maps.items()}
        self.n_geomcollection = 0
        # ped_crossing 每次都要遍历整张图的全部记录（原代码如此）；多边形本身与位姿无关，预先取出
        self._ped_polys = {}
        for loc in MAPS:
            polys = []
            for rec in self.nusc_maps[loc].ped_crossing:
                polys.append(np.array(self.explorer[loc].extract_polygon(rec['polygon_token']).exterior.xy))
            self._ped_polys[loc] = polys

    # ------------------------------------------------------------------ things（VectorizedLocalMap）
    def _sample_pts(self, line, sample_dist=1):
        distances = np.arange(0, line.length, sample_dist)
        pts = np.array([list(line.interpolate(d).coords) for d in distances]).reshape(-1, 2)
        return pts, len(pts)

    def _lines_to_vectors(self, geoms):
        out = []
        for line in geoms:
            if line.is_empty:
                continue
            if line.geom_type == 'MultiLineString':
                for single in line.geoms:
                    out.append(self._sample_pts(single))
            elif line.geom_type == 'LineString':
                out.append(self._sample_pts(line))
            elif line.geom_type == 'GeometryCollection':
                self.n_geomcollection += 1
                for single in _iter_lines(line)[0]:
                    out.append(self._sample_pts(single))
            else:
                raise NotImplementedError(line.geom_type)
        return out

    def _ped_crossing_lines(self, patch_box, patch_angle, location):
        patch_x, patch_y = patch_box[0], patch_box[1]
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        px0, py0, px1, py1 = patch.bounds
        out = []
        for poly_xy in self._ped_polys[location]:
            # 快速剔除：多边形包围盒与 patch 包围盒不相交时，两条边的 intersection 必为空（与原代码结果相同）
            if poly_xy[0].max() < px0 or poly_xy[0].min() > px1 or poly_xy[1].max() < py0 or poly_xy[1].min() > py1:
                continue
            dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
            x1, x2 = np.argsort(dist)[-2:]
            for idx in (x1, x2):
                pts = [(p0, p1) for p0, p1 in zip(poly_xy[0, idx:idx + 2], poly_xy[1, idx:idx + 2])]
                line = LineString(pts).intersection(patch)
                if not line.is_empty:
                    line = affinity.rotate(line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                    line = affinity.affine_transform(line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    out.append(line)
        return out

    def _contours(self, patch_box, patch_angle, location):
        ex = self.explorer[location]
        roads = ex._get_layer_polygon(patch_box, patch_angle, 'road_segment')
        lanes = ex._get_layer_polygon(patch_box, patch_angle, 'lane')
        union_segments = ops.unary_union([ops.unary_union(roads), ops.unary_union(lanes)])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        if union_segments.is_empty:
            return []
        if union_segments.geom_type == 'Polygon':
            union_segments = MultiPolygon([union_segments])
        elif union_segments.geom_type == 'GeometryCollection':
            union_segments = MultiPolygon([g for g in union_segments.geoms if g.geom_type == 'Polygon'])
        exteriors, interiors = [], []
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            interiors.extend(poly.interiors)
        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext = LinearRing(list(ext.coords)[::-1])
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)
        for inter in interiors:
            if not inter.is_ccw:
                inter = LinearRing(list(inter.coords)[::-1])
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)
        return self._lines_to_vectors(results)

    def vectors(self, location, e2g_t, e2g_q):
        map_pose = np.asarray(e2g_t, dtype=np.float64)[:2]
        rotation = Quaternion(e2g_q)
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        ex = self.explorer[location]
        vec = []
        for layer in ('road_divider', 'lane_divider'):
            for pts, n in self._lines_to_vectors(ex._get_layer_line(patch_box, patch_angle, layer)):
                vec.append(dict(pts=pts.astype(float), pts_num=n, type=CLASS2LABEL[layer]))
        for pts, n in self._lines_to_vectors(self._ped_crossing_lines(patch_box, patch_angle, location)):
            vec.append(dict(pts=pts.astype(float), pts_num=n, type=CLASS2LABEL['ped_crossing']))
        for pts, n in self._contours(patch_box, patch_angle, location):
            vec.append(dict(pts=pts.astype(float), pts_num=n, type=CLASS2LABEL['contours']))
        return vec

    # ------------------------------------------------------------------ rasterize.preprocess_map（只保留 instance mask）
    def _line_geom_to_mask(self, layer_geom, local_box, thickness, idx):
        patch_x, patch_y, patch_h, patch_w = local_box
        patch = NuScenesMapExplorer.get_patch_coord(local_box)
        canvas_h, canvas_w = self.canvas_size
        scale_h, scale_w = canvas_h / patch_h, canvas_w / patch_w
        trans_x, trans_y = -patch_x + patch_w / 2.0, -patch_y + patch_h / 2.0
        mask = np.zeros(self.canvas_size, np.uint8)
        for line in layer_geom:
            new_line = line.intersection(patch)
            if new_line.is_empty:
                continue
            new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
            new_line = affinity.scale(new_line, xfact=scale_w, yfact=scale_h, origin=(0, 0))
            parts, from_gc = _iter_lines(new_line)
            if from_gc:
                self.n_geomcollection += 1
            for single in parts:
                coords = np.asarray(list(single.coords), np.int32).reshape((-1, 2))
                if len(coords) < 2:
                    continue
                cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
                idx += 1
        return mask, idx

    def _instance_masks(self, vectors, num_classes=3):
        per_cls = {i: [] for i in range(num_classes)}
        for v in vectors:
            if v['pts_num'] >= 2:
                per_cls[v['type']].append(LineString(v['pts'][:v['pts_num']]))
        local_box = (0.0, 0.0, self.patch_size[0], self.patch_size[1])
        idx = 1
        inst, filt = [], []
        for i in range(num_classes):
            m, idx = self._line_geom_to_mask(per_cls[i], local_box, THICKNESS, idx)
            inst.append(m)
            f, _ = self._line_geom_to_mask(per_cls[i], local_box, THICKNESS + 4, 1)
            filt.append(f)
        inst = np.stack(inst)
        filt = np.stack(filt)
        # overlap_filter：高类别覆盖低类别
        for c in range(num_classes - 1, -1, -1):
            f = np.repeat((filt[c] != 0)[None, :], c, axis=0)
            inst[:c][f] = 0
        return inst

    # ------------------------------------------------------------------ drivable（obtain_map_info）
    def _drivable(self, location, l2g_t, l2g_r_mat):
        """= UniAD 副本 `eval_utils/map_api.py` 的 get_map_mask(['drivable_area'])。

        与 devkit 1.2.0 只有一处语义差别（照 UniAD）：`_polygon_geom_to_mask` 里裁剪结果既不是 Polygon 也不是
        MultiPolygon（GeometryCollection）时**跳过**该多边形；devkit 会把它交给 mask_for_polygons 而报错。
        """
        patch_box = (l2g_t[0], l2g_t[1], self.patch_size[0], self.patch_size[1])
        patch_angle = math.degrees(Quaternion(matrix=l2g_r_mat).yaw_pitch_roll[0])
        layer_geom = self.explorer[location]._get_layer_polygon(patch_box, patch_angle, 'drivable_area')
        local_box = (0.0, 0.0, patch_box[2], patch_box[3])
        patch = NuScenesMapExplorer.get_patch_coord(local_box)
        canvas_h, canvas_w = self.canvas_size
        scale_h, scale_w = canvas_h / local_box[2], canvas_w / local_box[3]
        trans_x, trans_y = -local_box[0] + local_box[3] / 2.0, -local_box[1] + local_box[2] / 2.0
        mask = np.zeros(self.canvas_size, np.uint8)
        for polygon in layer_geom:
            new_polygon = polygon.intersection(patch)
            if new_polygon.is_empty:
                continue
            new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
            new_polygon = affinity.scale(new_polygon, xfact=scale_w, yfact=scale_h, origin=(0, 0))
            if new_polygon.geom_type == 'Polygon':
                new_polygon = MultiPolygon([new_polygon])
            if not isinstance(new_polygon, MultiPolygon):
                self.n_geomcollection += 1
                continue
            if new_polygon.is_empty:
                continue
            ext = [np.array(p.exterior.coords).round().astype(np.int32) for p in new_polygon.geoms]
            inn = [np.array(pi.coords).round().astype(np.int32) for p in new_polygon.geoms for pi in p.interiors]
            cv2.fillPoly(mask, ext, 1)
            cv2.fillPoly(mask, inn, 0)
        erode = mask[None]                                   # get_map_mask 返回 [n_layers, H, W]
        erode = erode.transpose((2, 1, 0)).squeeze(2)
        return erode

    # ------------------------------------------------------------------ 对外接口
    def raster(self, location: str, e2g_t, e2g_q, l2e_t, l2e_q) -> Dict[str, np.ndarray]:
        """返回 dict(lane, divider, crossing, contour, drivable)：bool [200,200]，约定同 runner 的 seg。"""
        inst = self._instance_masks(self.vectors(location, e2g_t, e2g_q))
        inst = np.rot90(inst, k=-1, axes=(1, 2))
        # obtain_map_info 的位姿：lidar 原点、lidar->global 旋转
        l2e_r_mat = Quaternion(l2e_q).rotation_matrix
        e2g_r_mat = Quaternion(e2g_q).rotation_matrix
        l2g_r_mat = (l2e_r_mat.T @ e2g_r_mat.T).T
        l2g_t = np.asarray(l2e_t, dtype=np.float64) @ e2g_r_mat.T + np.asarray(e2g_t, dtype=np.float64)
        erode = self._drivable(location, l2g_t, l2g_r_mat)
        # get_data_info: map_mask = concat([erode, divider]) -> flip(axis=1) -> rot90(k=-1, axes=(1,2))；GT 只用第 0 通道
        drv = np.rot90(np.flip(erode[None], axis=1), k=-1, axes=(1, 2))[0]
        divider, crossing, contour = inst[0] != 0, inst[1] != 0, inst[2] != 0
        return dict(lane=np.ascontiguousarray(divider | crossing | contour), divider=np.ascontiguousarray(divider),
                    crossing=np.ascontiguousarray(crossing), contour=np.ascontiguousarray(contour),
                    drivable=np.ascontiguousarray(drv != 0))
