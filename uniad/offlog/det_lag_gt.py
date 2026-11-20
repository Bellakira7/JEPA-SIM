"""det lag gt.

定义（与 jepa_det_lag 一致，步长从 0.1 s 换成 UniAD 的 0.5 s 关键帧）
  对第 k 步（仿真位姿 P_k）里每个 GT 车辆实例，取它在 m∈{-2,-1,0,+1} 个关键帧时刻的世界位置，各自换到**该时刻的仿真自车系**
  （LiDAR_TOP）：q(m) = boxes_in(GT 在 k+m 帧, 位姿 P_{k+m})。这条折线就是"如果检测器整帧复制了 m 步前的输出，框会落在哪"。
  预测框到折线的最近点对应的 m* 给出 lag_s = −m*·0.5：0 = 对齐，+0.5 = 整帧复制上一个关键帧，正值 = 框落在车**后面**。
  只统计相对运动够快（折线首尾 ≥ vmin m/s）的目标，否则时间偏移不可辨识；自车在动，所以静止物体在自车系里也是"快"的。
  匹配 = 预测框（车辆类、score ≥ thr）与 GT 折线的最近距离做 Hungarian，门限 gate（默认 2 m，与 recall@2m 同）。
  perp_m = 预测框到折线的最近距离（滞后解释不了的那部分误差）。

和 jepa_det_lag 的两点差别（正文要写）
  * 折线只有 4 个点（跨 1.5 s，0.5 s 一个点），线性插值得到小数 m*；步长比 0.1 s 粗 5 倍，所以 lag 的分辨率约 0.1 s。
  * 没有 jitter（0.5 s 步长下相邻决策步差分没有意义）。
"""
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import common

LAG_MS = (-2, -1, 0, 1)       # 折线覆盖的关键帧偏移
LAG_DT = 0.5                  # 关键帧间隔（s）


def lag_on_polyline(p, poly, ks):
    """p:[2]；poly:[K,2] 对应偏移 ks。返回 (lag_frames=-m*, perp)，m* 在相邻点间线性插值。"""
    best = (np.inf, 0.0)
    for j in range(len(poly) - 1):
        a, b = poly[j], poly[j + 1]
        ab = b - a
        u = 0.0 if ab @ ab < 1e-12 else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
        d = float(np.linalg.norm(p - (a + u * ab)))
        if d < best[0]:
            best = (d, ks[j] + u * (ks[j + 1] - ks[j]))
    return -best[1], best[0]


def _dist_to_polys(P, polys):
    """P:[n,2]，polys:[g,K,2] → [g,n] 到各折线的最近距离（向量化）。"""
    g, K, _ = polys.shape
    d = np.full((g, len(P)), np.inf)
    for j in range(K - 1):
        a, b = polys[:, j][:, None, :], polys[:, j + 1][:, None, :]            # [g,1,2]
        ab = b - a
        den = (ab * ab).sum(-1)
        u = np.where(den < 1e-12, 0.0, np.clip(((P[None] - a) * ab).sum(-1) / np.maximum(den, 1e-12), 0.0, 1.0))
        d = np.minimum(d, np.linalg.norm(P[None] - (a + u[..., None] * ab), axis=-1))
    return d


def build_polys(gtw, data, pose_by_j: Dict[int, object], k: int, gt_k: dict, max_range=40.0,
                ms=LAG_MS) -> Optional[np.ndarray]:
    """第 k 步的 GT 车辆折线 [g,len(ms),2]（g 个实例）。pose_by_j：场景内下标 → SimPose（k+m 的位姿）。

    入选条件：m=0 时是车辆类、num_lidar_pts≥1、距 LiDAR 原点 ≤ max_range（与 `selfcheck.det_match` 一致）；
    且该实例在全部 m 时刻都有标注。位姿缺失（超出可用的仿真窗口）时返回 None，该步不统计 lag。
    """
    if any((k + m) not in pose_by_j for m in ms):
        return None
    veh = set(common.VEHICLE_CLASSES)
    sel = [i for i, n in enumerate(gt_k['det_name']) if n in veh and gt_k['num_lidar_pts'][i] >= 1
           and np.hypot(*gt_k['boxes'][i, :2]) <= max_range]
    if not sel:
        return np.zeros((0, len(ms), 2))
    inst0 = [gt_k['instance'][i] for i in sel]
    pts = {inst: {} for inst in inst0}
    for m in ms:
        p = pose_by_j[k + m]
        if m == 0:
            for i, inst in zip(sel, inst0):
                pts[inst][m] = gt_k['boxes'][i, :2]
            continue
        g = gtw.boxes_in(data.infos[p.tmpl_idx]['token'], p.e2g_t, p.e2g_q, p.l2e_t, p.l2e_q)
        where = {inst: j for j, inst in enumerate(g['instance'])}
        for inst in inst0:
            if inst in where:
                pts[inst][m] = g['boxes'][where[inst], :2]
    full = [inst for inst in inst0 if len(pts[inst]) == len(ms)]
    if not full:            # 有入选车辆但没有一辆在全部 m 时刻都在场（11-06 正式作业 9 个场景因此崩溃）
        return np.zeros((0, len(ms), 2))
    return np.stack([np.stack([pts[inst][m] for m in ms]) for inst in full]).reshape(len(full), len(ms), 2)


def lag_match(pred_boxes, pred_scores, pred_labels, polys: Optional[np.ndarray], score_thr=0.4, gate=2.0, vmin=1.5,
              ms=LAG_MS, dt=LAG_DT) -> dict:
    """预测框 vs GT 折线。polys=None（该步无折线）时返回 None。"""
    if polys is None:
        return None
    veh = set(common.VEHICLE_CLASSES)
    psel = [i for i in range(len(pred_scores)) if pred_scores[i] >= score_thr
            and common.CLASS_NAMES[int(pred_labels[i])] in veh]
    res = dict(lag_n_gt=int(len(polys)), lag_n_match=0, lag_s=[], lag_perp=[])
    if len(polys) == 0 or not psel:
        return res
    P = np.asarray(pred_boxes)[psel, :2].astype(np.float64)
    C = _dist_to_polys(P, polys)
    r, c = linear_sum_assignment(np.where(C < gate, C, 1e6))
    ks = np.asarray(ms, np.float64)
    for a, b in zip(r, c):
        if C[a, b] >= gate:
            continue
        res['lag_n_match'] += 1
        speed = float(np.linalg.norm(polys[a, -1] - polys[a, 0]) / ((ms[-1] - ms[0]) * dt))
        if speed < vmin:
            continue
        lag, perp = lag_on_polyline(P[b], polys[a], ks)
        res['lag_s'].append(round(float(lag * dt), 4))
        res['lag_perp'].append(round(float(perp), 4))
    return res
