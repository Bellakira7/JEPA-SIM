# -*- coding: utf-8 -*-
"""FlowSim (arXiv 2610.18030) Tab. I 的 source-to-reference agreement 指标。

纯 numpy / scipy，CPU 即可，不依赖 UniAD / torch。每个指标的定义、依据、备选见同目录 DEFINITIONS.md。

用法::

    cfg = AgreementConfig()
    fr  = frame_agreement(ref_out, src_out, cfg)      # 单帧
    agg = aggregate([fr1, fr2, ...])                  # 跨帧
    print(make_table({'Warp': {0.5: agg}}))           # 论文 Tab. I 格式的 markdown

输入（`ref_out` / `src_out`，规范化 UniAD 输出，缺字段一律按 "无此项" 处理，不抛异常）::

    boxes   float [N,>=7]   x,y,z,dim0,dim1,h,yaw,(vx,vy)   自车/LiDAR 系，m
                            footprint = 以 (x,y) 为中心、局部 x 方向边长 dim0、局部 y 方向边长 dim1、绕 z 逆时针旋转 yaw 的矩形
                            （对 mmdet3d 0.17 的 (w,l,h)+旧 yaw 与 1.x 的 (l,w,h)+新 yaw 两种成对约定都成立）
    scores  float [N]
    labels  int   [N]
    seg     dict(lane=[H,W], drivable=[H,W], divider=[H,W], crossing=[H,W])   概率 / logit / 二值（见 cfg.seg_kind）
    motion  dict(traj=[N,K,T,>=2], score=[N,K])    与 boxes 逐行对齐；位置为自车系绝对坐标（含 agent 中心平移）
    plan    float [P,2]            自车系累计位置（UniAD: P=6，0.5 s 间隔）
    meta    dict(seg_kind=..., seg_grid=...)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

NAN = float('nan')

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #


@dataclass
class AgreementConfig:
    """所有可调项。默认值即 DEFINITIONS.md 里选定的主定义。"""

    # ---- 公共 ----
    range_m: float = 50.0
    """评测区域：box 中心 |x|,|y| ≤ range_m（方形）；seg 栅格按格心同样裁到该范围。"""
    score_thr: float = 0.3
    """检测分数阈值；ref 与 src 用同一个阈值（对称）。"""
    labels: Optional[Sequence[int]] = None
    """只评这些类别；None = 全部类别（类别无关）。"""

    # ---- detection ----
    det_grid_res: float = 0.1
    """Detection IoU 的 BEV 栅格分辨率（m/格）。0.1 m 时单车 footprint 的离散误差 < 1%。"""
    det_match: str = 'nn'
    """cADE 的匹配方式：'nn'=对称最近邻（Chamfer，主定义）；'hungarian'=一对一、无门限，未配对框忽略（备选，会偏袒稀疏输出）。"""
    cade_empty_m: float = 50.0
    """一侧无框、另一侧有框时，每个框的最近邻距离记作该值；同时是 NN 距离的上限（= 评测范围）。"""
    match_gate_m: float = 2.0
    """门控一对一匹配（Hungarian + 中心距离门限）的门限；用于 det_box_iou / precision / recall 以及 motion 配对。"""
    match_same_label: bool = False
    """门控匹配是否要求同类别。"""

    # ---- seg ----
    seg_threshold: float = 0.5
    """概率图阈值（seg_kind='logit' 时在 sigmoid 之后比较）。"""
    seg_kind: str = 'meta'
    """'meta'=读 meta['seg_kind']（缺省 'prob'）；或强制 'prob' | 'logit' | 'mask'。"""
    seg_crop: bool = True
    """是否把栅格按格心裁到 ±range_m（栅格须以自车为中心、两轴对称）。"""
    seg_half_extent_m: float = 51.2
    """meta 里没给 seg_grid 时假定的栅格半边长（UniAD: 200×200 覆盖 ±51.2 m，0.512 m/格）。"""

    # ---- motion ----
    motion_mode: str = 'top1'
    """主定义用的模式选择：'top1'（各分支取自己 score 最大的模式）| 'wmean'（softmax 加权平均轨迹）| 'mink'（src 的 K 个模式里取离 ref top1 最近者，ADE/FDE 各自独立取 min）。"""
    motion_traj_kind: str = 'absolute'
    """输入 motion.traj 的含义：'absolute'=自车系绝对位置（含 agent 中心，lead 约定的契约）；'relative'=相对该 agent 当前中心的偏移
    （UniAD motion head 的原始 `traj[..., :2]`：cumsum 位移，方向已与自车轴对齐，见 DEFINITIONS §1）。内部统一转成绝对位置。"""
    motion_frame: str = 'relative'
    """'relative'=每个分支的轨迹先减去该 agent 自己的 box 中心（只比运动，不含位置差，位置差归 detection）；'absolute'=直接比自车系绝对位置。"""
    motion_unmatched: str = 'skip'
    """'skip'=只评配上的 agent（覆盖率另报）；'stay'=未配上的 ref/src agent 与 "原地不动" 比（对称惩罚）。"""
    motion_steps: Optional[int] = None
    """只用前 motion_steps 步；None=全部 T。"""

    # ---- planning ----
    plan_steps: Optional[int] = None
    """只用前 plan_steps 个路点；None=全部（UniAD 6 个 = 3.0 s）。"""

    def validate(self) -> None:
        assert self.det_match in ('nn', 'hungarian'), self.det_match
        assert self.motion_mode in ('top1', 'wmean', 'mink'), self.motion_mode
        assert self.motion_frame in ('relative', 'absolute'), self.motion_frame
        assert self.motion_traj_kind in ('relative', 'absolute'), self.motion_traj_kind
        assert self.motion_unmatched in ('skip', 'stay'), self.motion_unmatched
        assert self.seg_kind in ('meta', 'prob', 'logit', 'mask'), self.seg_kind


# --------------------------------------------------------------------------- #
# 输入整理
# --------------------------------------------------------------------------- #

def _np(x, dtype=np.float64):
    if x is None:
        return None
    if hasattr(x, 'detach'):  # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


class _Dets:
    """过滤（分数 / 类别 / 范围 / 非有限）之后的检测集合，带回原始行号以对齐 motion。"""
    __slots__ = ('boxes', 'scores', 'labels', 'idx', 'n_raw')

    def __init__(self, boxes, scores, labels, idx, n_raw):
        self.boxes, self.scores, self.labels, self.idx, self.n_raw = boxes, scores, labels, idx, n_raw

    @property
    def n(self) -> int:
        return len(self.idx)

    @property
    def xy(self) -> np.ndarray:
        return self.boxes[:, :2]


def _prep_dets(out: dict, cfg: AgreementConfig) -> _Dets:
    boxes = _np(out.get('boxes')) if out else None
    if boxes is None or boxes.size == 0:
        return _Dets(np.zeros((0, 9)), np.zeros(0), np.zeros(0, np.int64), np.zeros(0, np.int64), 0)
    boxes = boxes.reshape(len(boxes), -1)
    if boxes.shape[1] < 7:
        raise ValueError(f'boxes 需要至少 7 列 (x,y,z,dim0,dim1,h,yaw)，得到 {boxes.shape}')
    n = len(boxes)
    sc = _np(out.get('scores'))
    sc = np.ones(n) if sc is None or sc.size == 0 else sc.reshape(-1)
    lb = _np(out.get('labels'), np.int64)
    lb = np.zeros(n, np.int64) if lb is None or lb.size == 0 else lb.reshape(-1)
    if len(sc) != n or len(lb) != n:
        raise ValueError(f'boxes/scores/labels 行数不一致: {n}/{len(sc)}/{len(lb)}')
    keep = np.isfinite(boxes[:, :7]).all(1) & np.isfinite(sc)
    keep &= sc >= cfg.score_thr
    keep &= (np.abs(boxes[:, 0]) <= cfg.range_m) & (np.abs(boxes[:, 1]) <= cfg.range_m)
    if cfg.labels is not None:
        keep &= np.isin(lb, np.asarray(list(cfg.labels), np.int64))
    idx = np.nonzero(keep)[0]
    return _Dets(boxes[idx], sc[idx], lb[idx], idx, n)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

def rect_corners(x: float, y: float, a: float, b: float, yaw: float) -> np.ndarray:
    """矩形 4 角点（逆时针）。a=局部 x 方向边长，b=局部 y 方向边长，绕 z 逆时针转 yaw。"""
    c, s = math.cos(yaw), math.sin(yaw)
    loc = np.array([[a / 2, b / 2], [-a / 2, b / 2], [-a / 2, -b / 2], [a / 2, -b / 2]])
    R = np.array([[c, -s], [s, c]])
    return loc @ R.T + np.array([x, y])


def _poly_area(p: np.ndarray) -> float:
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland–Hodgman：凸多边形 subject 被凸多边形 clip（二者均逆时针）裁剪。"""
    out = [tuple(p) for p in subject]
    n = len(clip)
    for i in range(n):
        if not out:
            break
        a, b = clip[i], clip[(i + 1) % n]
        inp, out = out, []

        def inside(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12

        def inter(p, q):
            d1 = np.array(q) - np.array(p)
            d2 = b - a
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-15:
                return q
            t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / den
            return (p[0] + t * d1[0], p[1] + t * d1[1])

        s = inp[-1]
        for e in inp:
            if inside(e):
                if not inside(s):
                    out.append(inter(s, e))
                out.append(e)
            elif inside(s):
                out.append(inter(s, e))
            s = e
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def box_pair_iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    """两个 BEV 旋转矩形的精确 IoU（b = x,y,z,dim0,dim1,h,yaw；只用 x,y,dim0,dim1,yaw）。"""
    p1 = rect_corners(b1[0], b1[1], b1[3], b1[4], b1[6])
    p2 = rect_corners(b2[0], b2[1], b2[3], b2[4], b2[6])
    inter = abs(_poly_area(_clip_convex(p1, p2)))
    a1, a2 = abs(_poly_area(p1)), abs(_poly_area(p2))
    union = a1 + a2 - inter
    return float(inter / union) if union > 0 else 0.0


def rasterize_boxes(boxes: np.ndarray, half: float, res: float) -> np.ndarray:
    """把所有 footprint 的并集栅格化成 bool [n,n]（索引 [ix,iy]，ix 对应 x），格心采样。"""
    n = int(round(2 * half / res))
    mask = np.zeros((n, n), dtype=bool)
    for x, y, _, a, b, _, yaw in boxes[:, :7]:
        c, s = math.cos(yaw), math.sin(yaw)
        ex = abs(c) * a / 2 + abs(s) * b / 2
        ey = abs(s) * a / 2 + abs(c) * b / 2
        i0 = max(int(math.floor((x - ex + half) / res)), 0)
        i1 = min(int(math.ceil((x + ex + half) / res)), n)
        j0 = max(int(math.floor((y - ey + half) / res)), 0)
        j1 = min(int(math.ceil((y + ey + half) / res)), n)
        if i1 <= i0 or j1 <= j0:
            continue
        xs = -half + (np.arange(i0, i1) + 0.5) * res - x
        ys = -half + (np.arange(j0, j1) + 0.5) * res - y
        dx, dy = np.meshgrid(xs, ys, indexing='ij')
        u = c * dx + s * dy
        v = -s * dx + c * dy
        mask[i0:i1, j0:j1] |= (np.abs(u) <= a / 2) & (np.abs(v) <= b / 2)
    return mask


def _gated_match(ref: _Dets, src: _Dets, gate: float, same_label: bool) -> List[Tuple[int, int]]:
    """一对一匹配（Hungarian，cost=中心距离），距离 > gate（或类别不同且 same_label）的配对作废。
    返回 [(i_ref, j_src)]，下标是过滤后集合里的行号。"""
    if ref.n == 0 or src.n == 0:
        return []
    D = np.linalg.norm(ref.xy[:, None, :] - src.xy[None, :, :], axis=-1)
    bad = D > gate
    if same_label:
        bad |= ref.labels[:, None] != src.labels[None, :]
    BIG = 1e6
    cost = np.where(bad, BIG, D)
    r, c = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(r, c) if not bad[i, j]]


def _detection_metrics(ref: _Dets, src: _Dets, cfg: AgreementConfig, pool: dict):
    res: Dict[str, Any] = {}
    nr, ns = ref.n, src.n
    res.update(det_n_ref=nr, det_n_src=ns, det_both_empty=int(nr == 0 and ns == 0),
               det_one_empty=int((nr == 0) != (ns == 0)))

    # ---- IoU：BEV footprint 并集的栅格 IoU（主定义）----
    if nr == 0 and ns == 0:
        iou, inter_m2, union_m2 = NAN, 0.0, 0.0
    else:
        half = cfg.range_m
        mr = rasterize_boxes(ref.boxes, half, cfg.det_grid_res)
        ms = rasterize_boxes(src.boxes, half, cfg.det_grid_res)
        cell = cfg.det_grid_res ** 2
        inter_m2 = float((mr & ms).sum()) * cell
        union_m2 = float((mr | ms).sum()) * cell
        iou = inter_m2 / union_m2 if union_m2 > 0 else NAN
    res['det_iou'] = iou
    pool['det_iou'] = (inter_m2, union_m2)

    # ---- cADE ----
    if nr == 0 and ns == 0:
        d_r2s = d_s2r = np.zeros(0)
    else:
        cap = cfg.cade_empty_m
        if nr and ns:
            D = np.linalg.norm(ref.xy[:, None, :] - src.xy[None, :, :], axis=-1)
            d_r2s = np.minimum(D.min(1), cap)
            d_s2r = np.minimum(D.min(0), cap)
        else:
            d_r2s = np.full(nr, cap)
            d_s2r = np.full(ns, cap)
    nn_sum = float(d_r2s.sum() + d_s2r.sum())
    nn_cnt = nr + ns
    cade_nn = nn_sum / nn_cnt if nn_cnt else NAN
    res['det_cade_nn'] = cade_nn
    res['det_cade_ref2src'] = float(d_r2s.mean()) if nr else NAN
    res['det_cade_src2ref'] = float(d_s2r.mean()) if ns else NAN
    pool['det_cade_nn'] = (nn_sum, nn_cnt)
    # 备选：一对一无门限 Hungarian，只对配上的 min(nr,ns) 对取平均
    if nr == 0 and ns == 0:
        hsum, hcnt = 0.0, 0
    elif nr == 0 or ns == 0:
        hsum, hcnt = cfg.cade_empty_m * (nr + ns), nr + ns
    else:
        D = np.linalg.norm(ref.xy[:, None, :] - src.xy[None, :, :], axis=-1)
        r, c = linear_sum_assignment(D)
        hsum, hcnt = float(D[r, c].sum()), len(r)
    res['det_cade_hung'] = hsum / hcnt if hcnt else NAN
    pool['det_cade_hung'] = (hsum, hcnt)
    res['det_cade'] = res['det_cade_nn'] if cfg.det_match == 'nn' else res['det_cade_hung']
    pool['det_cade'] = pool['det_cade_nn'] if cfg.det_match == 'nn' else pool['det_cade_hung']

    # ---- 门控匹配：box-level IoU / precision / recall（辅助）----
    pairs = _gated_match(ref, src, cfg.match_gate_m, cfg.match_same_label)
    nm = len(pairs)
    res['det_n_match'] = nm
    if nr == 0 and ns == 0:
        res['det_box_iou'] = NAN
        pool['det_box_iou'] = (0.0, 0)
    else:
        s_iou = sum(box_pair_iou(ref.boxes[i], src.boxes[j]) for i, j in pairs)
        den = nr + ns - nm
        res['det_box_iou'] = s_iou / den if den else NAN
        pool['det_box_iou'] = (s_iou, den)
    res['det_recall'] = nm / nr if nr else NAN
    res['det_precision'] = nm / ns if ns else NAN
    return res, pairs


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #

_SEG_KEYS = (('lane', 'lane_iou'), ('drivable', 'drive_iou'),
             ('divider', 'divider_iou'), ('crossing', 'crossing_iou'))


def _seg_kind(ref_out, src_out, cfg) -> str:
    if cfg.seg_kind != 'meta':
        return cfg.seg_kind
    for o in (ref_out, src_out):
        m = (o or {}).get('meta') or {}
        if m.get('seg_kind'):
            return str(m['seg_kind'])
        if m.get('seg_is_logit'):
            return 'logit'
    return 'prob'


def _seg_grid_half(out, shape, cfg) -> Tuple[float, float]:
    g = ((out or {}).get('meta') or {}).get('seg_grid')
    H, W = shape
    if isinstance(g, dict):
        if g.get('half_extent') is not None:
            h = np.broadcast_to(np.asarray(g['half_extent'], float), (2,))
            return float(h[0]), float(h[1])
        if g.get('res') is not None:
            res = np.broadcast_to(np.asarray(g['res'], float), (2,))
            return float(res[0] * H / 2), float(res[1] * W / 2)
    elif isinstance(g, (int, float)) and g:
        return float(g) * H / 2, float(g) * W / 2
    return cfg.seg_half_extent_m, cfg.seg_half_extent_m


def _binarize(a: np.ndarray, kind: str, thr: float) -> np.ndarray:
    a = np.nan_to_num(a.astype(np.float64), nan=0.0)
    if kind == 'logit':
        a = 1.0 / (1.0 + np.exp(-a))
    elif kind == 'mask':
        return a > 0.5
    return a >= thr


def _crop_mask(shape, half_hw, range_m) -> np.ndarray:
    H, W = shape
    ys = -half_hw[0] + (np.arange(H) + 0.5) * (2 * half_hw[0] / H)
    xs = -half_hw[1] + (np.arange(W) + 0.5) * (2 * half_hw[1] / W)
    return (np.abs(ys)[:, None] <= range_m) & (np.abs(xs)[None, :] <= range_m)


def _seg_get(seg: dict, key: str):
    """取 seg[key]；缺 lane 时用 divider / crossing / contour 里提供了的几张取并集（UniAD 的 lanes_iou 口径）。"""
    v = seg.get(key)
    if v is None and key == 'lane':
        parts = [_np(seg[k]) for k in ('divider', 'crossing', 'contour') if seg.get(k) is not None]
        if parts:
            return np.maximum.reduce([np.squeeze(p) for p in parts])
    return v


def _seg_metrics(ref_out, src_out, cfg, pool) -> dict:
    res: Dict[str, Any] = {}
    kind = _seg_kind(ref_out, src_out, cfg)
    sr = (ref_out or {}).get('seg') or {}
    ss = (src_out or {}).get('seg') or {}
    for key, name in _SEG_KEYS:
        a, b = _seg_get(sr, key), _seg_get(ss, key)
        if a is None or b is None:
            res[name] = NAN
            pool[name] = (0.0, 0.0)
            continue
        a, b = _np(a), _np(b)
        a, b = np.squeeze(a), np.squeeze(b)
        if a.shape != b.shape or a.ndim != 2:
            raise ValueError(f'seg[{key}] 形状不一致或不是 2D: {a.shape} vs {b.shape}')
        ma, mb = _binarize(a, kind, cfg.seg_threshold), _binarize(b, kind, cfg.seg_threshold)
        if cfg.seg_crop:
            c = _crop_mask(a.shape, _seg_grid_half(ref_out, a.shape, cfg), cfg.range_m)
            ma, mb = ma & c, mb & c
        inter, union = float((ma & mb).sum()), float((ma | mb).sum())
        res[name] = inter / union if union > 0 else NAN  # 两侧都全空 -> 无信息
        pool[name] = (inter, union)
    return res


# --------------------------------------------------------------------------- #
# Motion
# --------------------------------------------------------------------------- #

def _mode_probs(score: np.ndarray) -> np.ndarray:
    """score [K] -> 概率。已是概率单纯形就原样用；否则当作 (log-)logit 做 softmax（UniAD 输出是 log_softmax）。"""
    s = np.asarray(score, np.float64)
    if (s >= 0).all() and abs(s.sum() - 1.0) < 1e-3:
        return s
    e = np.exp(s - s.max())
    return e / e.sum()


def _motion_arrays(out: dict, dets: _Dets, cfg: AgreementConfig):
    """返回 (traj[M,K,T,2], score[M,K]) 或 None；M=过滤后的检测数。"""
    m = (out or {}).get('motion')
    if not m or m.get('traj') is None:
        return None
    traj = _np(m['traj'])
    if traj is None or traj.size == 0:
        return None
    if traj.ndim != 4 or traj.shape[-1] < 2:
        raise ValueError(f'motion.traj 需为 [N,K,T,>=2]，得到 {traj.shape}')
    if traj.shape[0] != dets.n_raw:
        raise ValueError(f'motion.traj 的 N={traj.shape[0]} 与 boxes 的 N={dets.n_raw} 不对齐')
    traj = traj[dets.idx][..., :2]
    if cfg.motion_traj_kind == 'relative':
        traj = traj + dets.xy[:, None, None, :]
    K = traj.shape[1]
    sc = _np(m.get('score'))
    sc = np.zeros((dets.n_raw, K)) if sc is None or sc.size == 0 else sc.reshape(dets.n_raw, K)
    sc = sc[dets.idx]
    if cfg.motion_steps:
        traj = traj[:, :, :cfg.motion_steps]
    return traj, sc


def _agent_traj(traj_k: np.ndarray, sc: np.ndarray, center: np.ndarray, frame: str, how: str):
    """取某个 agent 的 (代表轨迹 [T,2] 或模式集 [K,T,2])。"""
    t = traj_k - (center[None, None, :] if frame == 'relative' else 0.0)
    if how == 'top1':
        return t[int(np.argmax(sc))]
    if how == 'wmean':
        return np.tensordot(_mode_probs(sc), t, axes=(0, 0))
    return t  # 'all'


def _pair_ade_fde(ref_t, ref_s, ref_c, src_t, src_s, src_c, frame, mode) -> Tuple[float, float]:
    if mode == 'mink':
        r = _agent_traj(ref_t, ref_s, ref_c, frame, 'top1')            # [T,2]
        s = _agent_traj(src_t, src_s, src_c, frame, 'all')             # [K,T,2]
        err = np.linalg.norm(s - r[None], axis=-1)                     # [K,T]
        return float(err.mean(1).min()), float(err[:, -1].min())
    r = _agent_traj(ref_t, ref_s, ref_c, frame, mode)
    s = _agent_traj(src_t, src_s, src_c, frame, mode)
    e = np.linalg.norm(s - r, axis=-1)
    return float(e.mean()), float(e[-1])


def _stay_err(t, s, c, frame) -> Tuple[float, float]:
    """某个未配上的 agent 的 top-1 轨迹 与 "原地不动" 的 ADE/FDE。"""
    tr = _agent_traj(t, s, c, frame, 'top1')
    e = np.linalg.norm(tr - (0.0 if frame == 'relative' else c[None]), axis=-1)
    return float(e.mean()), float(e[-1])


def _put(res: dict, pool: dict, name: str, a: List[float], f: List[float]) -> None:
    """name 形如 'motion' / 'motion_top1'；写 {name}_ade / {name}_fde 及其 (sum, count) 池化项。"""
    a = [x for x in a if np.isfinite(x)]          # 轨迹里有 NaN 的配对直接丢弃
    f = [x for x in f if np.isfinite(x)]
    res[f'{name}_ade'] = float(np.mean(a)) if a else NAN
    res[f'{name}_fde'] = float(np.mean(f)) if f else NAN
    pool[f'{name}_ade'] = (float(np.sum(a)), len(a))
    pool[f'{name}_fde'] = (float(np.sum(f)), len(f))


def _motion_metrics(ref_out, src_out, ref: _Dets, src: _Dets, pairs, cfg, pool) -> dict:
    res: Dict[str, Any] = {}
    for nm in ('motion', 'motion_top1', 'motion_wmean', 'motion_mink', 'motion_stay',
               'motion_nn_rel', 'motion_nn_abs', 'motion_idx_rel', 'motion_idx_abs', 'motion_base'):
        _put(res, pool, nm, [], [])
    res.update(motion_n_pairs=0, motion_cov=NAN)
    mr, ms = _motion_arrays(ref_out, ref, cfg), _motion_arrays(src_out, src, cfg)
    # 一侧没有 motion 数据：若它过滤后本来就没有 agent（N=0），当作空 agent 集合（与另一侧同形状）；
    # 若它有框却没有 motion，则数据缺失 -> 全部 NaN。
    if mr is None and ms is not None and ref.n == 0:
        mr = (np.zeros((0,) + ms[0].shape[1:]), np.zeros((0, ms[1].shape[1])))
    if ms is None and mr is not None and src.n == 0:
        ms = (np.zeros((0,) + mr[0].shape[1:]), np.zeros((0, mr[1].shape[1])))
    if mr is None or ms is None or ref.n == 0:
        return res
    rt, rs = mr
    st, ss = ms
    if rt.shape[2] != st.shape[2]:
        raise ValueError(f'ref/src motion 的 T 不一致: {rt.shape} vs {st.shape}')
    rc, sc_ = ref.xy, src.xy

    def acc(pair_list, frame, mode):
        a, f = [], []
        for i, j in pair_list:
            x, y = _pair_ade_fde(rt[i], rs[i], rc[i], st[j], ss[j], sc_[j], frame, mode)
            a.append(x)
            f.append(y)
        return a, f

    # 主定义 + 三种模式选择变体（配对 = 门控 Hungarian）
    for mode in ('top1', 'wmean', 'mink'):
        a, f = acc(pairs, cfg.motion_frame, mode)
        _put(res, pool, f'motion_{mode}', a, f)
        if mode == cfg.motion_mode:
            _put(res, pool, 'motion', a, f)
            res['motion_n_pairs'] = int(np.isfinite(a).sum())
    res['motion_cov'] = len(pairs) / ref.n

    # 无信息基线：ref 各 agent 的 top-1 相对轨迹 与 "原地不动" 的 ADE/FDE（只依赖 ref，与 src 无关）。
    # 任何 "agreement" 数值如果不明显低于它，就说明几乎没有一致性信息。
    a, f = [], []
    for i in range(ref.n):
        x, y = _stay_err(rt[i], rs[i], rc[i], 'relative')
        a.append(x)
        f.append(y)
    _put(res, pool, 'motion_base', a, f)

    # 未配上的 agent 与 "原地不动" 比（对称：ref 与 src 的未配上者都算），模式固定 top1
    mi, mj = {i for i, _ in pairs}, {j for _, j in pairs}
    a, f = acc(pairs, cfg.motion_frame, 'top1')
    for i in range(ref.n):
        if i not in mi:
            x, y = _stay_err(rt[i], rs[i], rc[i], cfg.motion_frame)
            a.append(x)
            f.append(y)
    for j in range(src.n):
        if j not in mj:
            x, y = _stay_err(st[j], ss[j], sc_[j], cfg.motion_frame)
            a.append(x)
            f.append(y)
    _put(res, pool, 'motion_stay', a, f)
    if cfg.motion_unmatched == 'stay':
        _put(res, pool, 'motion', a, f)
        res['motion_n_pairs'] = int(np.isfinite(a).sum())

    # 校准变体：按数组下标配对（i<->i，朴素实现会这样），top-1
    idx_pairs = [(i, i) for i in range(min(ref.n, src.n))]
    for frame, tag in (('relative', 'motion_idx_rel'), ('absolute', 'motion_idx_abs')):
        a, f = acc(idx_pairs, frame, 'top1')
        _put(res, pool, tag, a, f)

    # 校准变体：无门限最近邻配对（每个 ref agent 配最近的 src agent，多对一），top-1
    if src.n:
        D = np.linalg.norm(rc[:, None, :] - sc_[None, :, :], axis=-1)
        nn = [(i, int(D[i].argmin())) for i in range(ref.n)]
        for frame, tag in (('relative', 'motion_nn_rel'), ('absolute', 'motion_nn_abs')):
            a, f = acc(nn, frame, 'top1')
            _put(res, pool, tag, a, f)
    return res


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #

def _plan_array(out) -> Optional[np.ndarray]:
    p = (out or {}).get('plan')
    if p is None:
        return None
    p = _np(p)
    if p is None or p.size == 0:
        return None
    p = np.squeeze(p) if p.ndim > 2 else p
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError(f'plan 需为 [P,2]，得到 {p.shape}')
    return p[:, :2]


def _plan_metrics(ref_out, src_out, cfg, pool) -> dict:
    pr, ps = _plan_array(ref_out), _plan_array(src_out)
    if pr is None or ps is None:
        pool['plan_ade'] = pool['plan_fde'] = (0.0, 0)
        return dict(plan_ade=NAN, plan_fde=NAN)
    if pr.shape != ps.shape:
        raise ValueError(f'plan 形状不一致: {pr.shape} vs {ps.shape}')
    if cfg.plan_steps:
        pr, ps = pr[:cfg.plan_steps], ps[:cfg.plan_steps]
    e = np.linalg.norm(ps - pr, axis=-1)           # ‖p̂_k − p_k‖₂, p̂ 来自 src，p 来自 ref
    ade, fde = float(e.mean()), float(e[-1])
    pool['plan_ade'], pool['plan_fde'] = (ade, 1), (fde, 1)
    return dict(plan_ade=ade, plan_fde=fde)


# --------------------------------------------------------------------------- #
# 单帧入口
# --------------------------------------------------------------------------- #

def frame_agreement(ref_out: dict, src_out: dict, cfg: Optional[AgreementConfig] = None) -> dict:
    """单帧 agreement。返回扁平 dict：指标 + 计数 + `_pool`（每个指标的 (分子, 分母)，供 aggregate 做池化）。

    约定（详见 DEFINITIONS.md §5）：
      * 一个指标"无信息"（例如两侧都没检测框时的 det_iou；两侧 seg 都为空时的 lane_iou）记 NaN，
        aggregate 时从帧均值里剔除并在 n 里体现；
      * 一侧空、另一侧非空：det_iou=0，det_cade=cfg.cade_empty_m（对 ref 空 / src 空完全对称）；
      * 缺 seg / motion / plan 字段：对应指标 NaN，不抛异常。
    """
    cfg = cfg or AgreementConfig()
    cfg.validate()
    ref_out, src_out = ref_out or {}, src_out or {}
    pool: Dict[str, Tuple[float, float]] = {}
    ref, src = _prep_dets(ref_out, cfg), _prep_dets(src_out, cfg)
    det, pairs = _detection_metrics(ref, src, cfg, pool)
    out: Dict[str, Any] = {}
    out.update(det)
    out.update(_seg_metrics(ref_out, src_out, cfg, pool))
    out.update(_motion_metrics(ref_out, src_out, ref, src, pairs, cfg, pool))
    out.update(_plan_metrics(ref_out, src_out, cfg, pool))
    for k in ('sample_token', 'scene_token', 'timestamp'):
        v = (ref_out.get('meta') or {}).get(k)
        if v is not None:
            out[k] = v
    out['_pool'] = pool
    return out


# --------------------------------------------------------------------------- #
# 跨帧聚合
# --------------------------------------------------------------------------- #

HEADLINE_KEYS = ('det_iou', 'det_cade', 'lane_iou', 'drive_iou',
                 'motion_ade', 'motion_fde', 'plan_ade', 'plan_fde')
COUNT_KEYS = ('det_n_ref', 'det_n_src', 'det_n_match', 'det_both_empty', 'det_one_empty', 'motion_n_pairs')
_META_KEYS = ('sample_token', 'scene_token', 'timestamp', '_pool')


def aggregate(frames: Iterable[dict]) -> dict:
    """跨帧聚合。

    **主口径 = 帧级先算后平均**（每帧一个值，帧间等权；NaN 帧剔除）：`mean` / `median` / `std` / `sem` / `n`。
    同时给 **池化口径** `pooled`（Σ分子/Σ分母：IoU 为总交/总并，cADE/Motion 为所有框/配对等权）。
    返回::

        {'n_frames': int,
         'det_iou': {'mean','median','std','sem','n','pooled'}, ... ,   # 每个出现过的指标
         'counts': {'det_n_ref': 总和, ...}}
    """
    frames = list(frames)
    out: Dict[str, Any] = {'n_frames': len(frames)}
    if not frames:
        out['counts'] = {}
        return out
    keys: List[str] = []
    for fr in frames:
        for k, v in fr.items():
            if k in _META_KEYS or k in COUNT_KEYS or k in keys:
                continue
            if isinstance(v, (int, float, np.floating, np.integer)):
                keys.append(k)
    for k in keys:
        vals = np.array([fr.get(k, NAN) for fr in frames], dtype=np.float64)
        ok = np.isfinite(vals)
        n = int(ok.sum())
        st: Dict[str, Any] = dict(n=n)
        if n:
            v = vals[ok]
            st.update(mean=float(v.mean()), median=float(np.median(v)),
                      std=float(v.std(ddof=1)) if n > 1 else 0.0,
                      sem=float(v.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0)
        else:
            st.update(mean=NAN, median=NAN, std=NAN, sem=NAN)
        num = den = 0.0
        has_pool = False
        for fr in frames:
            p = (fr.get('_pool') or {}).get(k)
            if p is not None and np.isfinite(p[0]) and np.isfinite(p[1]):
                has_pool = True
                num += float(p[0])
                den += float(p[1])
        st['pooled'] = num / den if has_pool and den > 0 else NAN
        out[k] = st
    out['counts'] = {k: float(sum(fr.get(k, 0) for fr in frames)) for k in COUNT_KEYS}
    return out


# --------------------------------------------------------------------------- #
# Tab. I 风格 markdown 表
# --------------------------------------------------------------------------- #

# (聚合键, 表头, 越大越好?)
TABLE_COLUMNS = (
    ('det_iou', 'Det IoU↑', True), ('det_cade', 'Det cADE↓', False),
    ('lane_iou', 'Lane↑', True), ('drive_iou', 'Drive.↑', True),
    ('motion_ade', 'Mot ADE↓', False), ('motion_fde', 'Mot FDE↓', False),
    ('plan_ade', 'Plan ADE↓', False), ('plan_fde', 'Plan FDE↓', False),
)


def _stat(agg: Any, key: str, stat: str) -> float:
    if agg is None:
        return NAN
    v = agg.get(key) if isinstance(agg, dict) else None
    if v is None:
        return NAN
    if isinstance(v, dict):
        v = v.get(stat, NAN)
    try:
        return float(v)
    except (TypeError, ValueError):
        return NAN


def _dt_sort_key(dt):
    try:
        return (0, float(str(dt).replace('s', '').strip()))
    except ValueError:
        return (1, 0.0)


def make_table(results_by_source_and_dt: Dict[Any, Dict[Any, dict]], stat: str = 'mean',
               bold_best: bool = True, show_n: bool = True, decimals: int = 2) -> str:
    """按 Tab. I 的列顺序输出 markdown 表。

    参数 `results_by_source_and_dt`：`{source: {dt: agg}}`，`agg` 是 `aggregate()` 的返回值，
    或 `{指标: 数值}` 的扁平 dict；也接受 `{(source, dt): agg}`。`dt` 可为 0.5 / '0.5' / '0.5s' / None（不适用，排最前）。
    行按 Δt 分块（同论文），块内按 source 的出现顺序；`bold_best` 在每个 Δt 块内对每列加粗最优值。
    """
    rows: List[Tuple[Any, Any, dict]] = []
    for k, v in results_by_source_and_dt.items():
        if isinstance(k, tuple) and len(k) == 2:        # {(source, dt): agg}
            rows.append((k[0], k[1], v))
        else:                                            # {source: {dt: agg}}
            for dt, agg in v.items():
                rows.append((k, dt, agg))
    dts: List[Any] = []
    for _, dt, _ in rows:
        if dt not in dts:
            dts.append(dt)
    dts.sort(key=lambda d: (-1, 0.0) if d is None else _dt_sort_key(d))

    head = ['Source', 'Δt (s)'] + [h for _, h, _ in TABLE_COLUMNS] + (['n'] if show_n else [])
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join(['---'] * 2 + ['---:'] * (len(head) - 2)) + '|']
    fmt = f'{{:.{decimals}f}}'
    for dt in dts:
        block = [(s, a) for s, d, a in rows if d == dt]
        best: Dict[str, float] = {}
        if bold_best:
            for key, _, hib in TABLE_COLUMNS:
                vals = [_stat(a, key, stat) for _, a in block]
                vals = [v for v in vals if np.isfinite(v)]
                if vals:
                    best[key] = max(vals) if hib else min(vals)
        for s, a in block:
            cells = [str(s), '–' if dt is None else str(dt).replace('s', '')]
            for key, _, _ in TABLE_COLUMNS:
                v = _stat(a, key, stat)
                if not np.isfinite(v):
                    cells.append('–')
                    continue
                txt = fmt.format(v)
                if bold_best and key in best and fmt.format(best[key]) == txt:
                    txt = f'**{txt}**'
                cells.append(txt)
            if show_n:
                n = a.get('n_frames') if isinstance(a, dict) else None
                cells.append('–' if n is None else str(int(n)))
            lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)
