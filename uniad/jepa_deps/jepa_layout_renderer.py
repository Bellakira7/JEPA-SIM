"""Standalone box and lane projection for camera layout conditioning.

Boxes: ego coordinates [x, y, z, length, width, height, yaw], Z up.
OpenCV color values use BGR channel order.
"""

import cv2
import numpy as np



# `lane_type_colors`/`border_type_colors`(那两个 dict 是 RGB, 这里手动转成 BGR 并直接写死),

# 不跨模块 import"的既有约定(见文件头 docstring)冲突。key 是 hdmap_lane_map/hdmap_border_map

_LANE_TYPE_COLORS_BGR = {
    -1: (255, 255, 255),
    0: (0, 192, 192),
    1: (51, 153, 51),
    2: (0, 192, 192),
    3: (51, 153, 51),
    4: (0, 192, 192),
    5: (0, 192, 192),
    6: (0, 192, 192),
    7: (0, 192, 192),
    8: (0, 128, 255),
    9: (0, 192, 192),
    10: (0, 192, 192),
    11: (51, 153, 51),
    12: (0, 192, 192),
}
_BORDER_TYPE_COLORS_BGR = {
    -1: (255, 255, 255),
    0: (0, 0, 255),
    1: (0, 0, 255),
    2: (255, 0, 128),
    3: (255, 0, 128),
    4: (0, 128, 255),
    5: (0, 0, 0),
    6: (51, 153, 51),
    7: (255, 0, 0),
}


# color_list[label] 是 RGB，cv2 画的时候做了 [::-1]，这里直接存 BGR。
# 下标 = gt_labels_3d / pred_labels 的 9 类编号（data_utils.subtype_to_8cls_mapping）。
# 0 car, 1 ped, 2 cyc, 3 bus, 4 truck, 5 cone, 6 tricycle, 7 charging, 8 dontcare。
_BOX_COLOR_BGR = {
    0: (0, 140, 255),
    1: (0, 225, 0),
    2: (255, 0, 0),
    3: (0, 255, 255),
    4: (255, 0, 255),
    5: (255, 255, 0),
    6: (255, 255, 255),
    7: (128, 128, 128),
    8: (0, 69, 255),
}
_BOX_COLOR_FALLBACK_BGR = (255, 255, 255)


def box_colors_bgr(n: int, labels, fallback=_BOX_COLOR_FALLBACK_BGR) -> np.ndarray:
    """``(n, 3)`` uint8 BGR。``labels`` 与框一一对应；缺标注或长度不够的框用 ``fallback``（默认白）。
    负值或表外编号用 dontcare（下标 8），和可视化里「不关心」那一档同色。"""
    colors = np.empty((n, 3), np.uint8)
    colors[:] = fallback
    if labels is None or n == 0:
        return colors
    labs = np.asarray(labels).reshape(-1)
    for i in range(min(n, labs.shape[0])):
        lab = int(labs[i])
        colors[i] = _BOX_COLOR_BGR.get(lab, _BOX_COLOR_BGR[8])
    return colors


_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def _box_corners_ego(boxes: np.ndarray) -> np.ndarray:
    """(N,7) [x,y,z,l,w,h,yaw] -> (N,8,3) 角点, 顶点顺序按 _EDGES 索引
    (0-3 底面, 4-7 顶面)。"""
    x, y, z, l, w, h, yaw = boxes.T
    dx = np.stack([l, l, -l, -l], axis=1) / 2  # (N,4)
    dy = np.stack([-w, w, w, -w], axis=1) / 2
    cos_y, sin_y = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    cx = x[:, None] + dx * cos_y - dy * sin_y
    cy = y[:, None] + dx * sin_y + dy * cos_y
    z_bot = np.broadcast_to((z - h / 2)[:, None], cx.shape)
    z_top = np.broadcast_to((z + h / 2)[:, None], cx.shape)
    bottom = np.stack([cx, cy, z_bot], axis=2)  # (N,4,3)
    top = np.stack([cx, cy, z_top], axis=2)
    return np.concatenate([bottom, top], axis=1)  # (N,8,3)


def _draw_box_pinhole(img, corners_cam, K, color, thickness, near_clip, w_img, h_img):
    z = corners_cam[:, 2]
    if (z <= near_clip).all():
        return
    uv = K @ corners_cam.T
    uv = uv[:2] / np.clip(uv[2:3], 1e-6, None)
    for i, j in _EDGES:
        # 跳过相机后方的端点, 以及离图像过远的投影(避免数值爆炸的乱线),
        
        if z[i] <= near_clip or z[j] <= near_clip:
            continue
        if np.abs(uv[:, [i, j]]).max() > 8 * max(w_img, h_img):
            continue
        p1 = (round(uv[0, i]), round(uv[1, i]))
        p2 = (round(uv[0, j]), round(uv[1, j]))
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def _draw_box_fisheye(img, corners_cam, K, color, thickness, segments, near_clip):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    for i, j in _EDGES:
        p_i, p_j = corners_cam[i], corners_cam[j]
        if p_i[2] <= near_clip and p_j[2] <= near_clip:
            continue
        t = np.linspace(0.0, 1.0, segments + 1)[:, None]
        pts = p_i[None, :] * (1 - t) + p_j[None, :] * t  # (segments+1, 3)
        valid = pts[:, 2] > near_clip
        if not valid.any():
            continue
        r = np.clip(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2), 1e-6, None)
        theta = np.arctan2(r, pts[:, 2])
        u = fx * theta * pts[:, 0] / r + cx
        v = fy * theta * pts[:, 1] / r + cy
        line_pts = np.stack([u, v], axis=1)[valid].astype(np.int32)
        if line_pts.shape[0] >= 2:
            cv2.polylines(img, [line_pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _fill_box_pinhole(img, corners_cam, K, weight, near_clip, w_img, h_img):
    z = corners_cam[:, 2]
    valid = z > near_clip
    if valid.sum() < 3:
        return
    uv = K @ corners_cam[valid].T
    uv = (uv[:2] / np.clip(uv[2:3], 1e-6, None)).T  # (M,2)
    # 与 _draw_box_pinhole 同款数值爆炸防护, 换成 clip(而非丢弃整框) 以尽量保留可见部分凸包。
    bound = 8 * max(w_img, h_img)
    uv = np.clip(uv, -bound, bound)
    hull = cv2.convexHull(uv.astype(np.float32))
    cv2.fillConvexPoly(img, hull.astype(np.int32), float(weight))


def _fill_box_fisheye(img, corners_cam, K, weight, segments, near_clip):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts_uv = []
    for i, j in _EDGES:
        p_i, p_j = corners_cam[i], corners_cam[j]
        if p_i[2] <= near_clip and p_j[2] <= near_clip:
            continue
        t = np.linspace(0.0, 1.0, segments + 1)[:, None]
        pts = p_i[None, :] * (1 - t) + p_j[None, :] * t  # (segments+1, 3)
        valid = pts[:, 2] > near_clip
        if not valid.any():
            continue
        r = np.clip(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2), 1e-6, None)
        theta = np.arctan2(r, pts[:, 2])
        u = fx * theta * pts[:, 0] / r + cx
        v = fy * theta * pts[:, 1] / r + cy
        pts_uv.append(np.stack([u, v], axis=1)[valid])
    if not pts_uv:
        return
    all_pts = np.concatenate(pts_uv, axis=0)
    if all_pts.shape[0] < 3:
        return
    hull = cv2.convexHull(all_pts.astype(np.float32))
    cv2.fillPoly(img, [hull.astype(np.int32)], float(weight))


def _fill_box_color_pinhole(img, corners_cam, K, color, near_clip, w_img, h_img):
    """同 ``_fill_box_pinhole``，但填充 BGR 三元色而非标量权重（供 ``render_filled`` 用）。"""
    z = corners_cam[:, 2]
    valid = z > near_clip
    if valid.sum() < 3:
        return
    uv = K @ corners_cam[valid].T
    uv = (uv[:2] / np.clip(uv[2:3], 1e-6, None)).T
    bound = 8 * max(w_img, h_img)
    uv = np.clip(uv, -bound, bound)
    hull = cv2.convexHull(uv.astype(np.float32))
    cv2.fillConvexPoly(img, hull.astype(np.int32), color)


def _fill_box_color_fisheye(img, corners_cam, K, color, segments, near_clip):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts_uv = []
    for i, j in _EDGES:
        p_i, p_j = corners_cam[i], corners_cam[j]
        if p_i[2] <= near_clip and p_j[2] <= near_clip:
            continue
        t = np.linspace(0.0, 1.0, segments + 1)[:, None]
        pts = p_i[None, :] * (1 - t) + p_j[None, :] * t
        valid = pts[:, 2] > near_clip
        if not valid.any():
            continue
        r = np.clip(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2), 1e-6, None)
        theta = np.arctan2(r, pts[:, 2])
        u = fx * theta * pts[:, 0] / r + cx
        v = fy * theta * pts[:, 1] / r + cy
        pts_uv.append(np.stack([u, v], axis=1)[valid])
    if not pts_uv:
        return
    all_pts = np.concatenate(pts_uv, axis=0)
    if all_pts.shape[0] < 3:
        return
    hull = cv2.convexHull(all_pts.astype(np.float32))
    cv2.fillPoly(img, [hull.astype(np.int32)], color)


def render_filled(
    boxes,
    K,
    imu2cam,
    image_size,
    cam_model='pinhole',
    color=(255, 255, 255),
    near_clip=0.1,
    fisheye_segments=9,
    canvas=None,
) -> np.ndarray:
    """实心剪影版 layout 条件（消融"filled"臂，与 ``render_wireframe`` 二选一）。

    动机（灵敏度探针 workdir/jepa_probe/probe_layout_sensitivity.py 的实测结论）：
    2px 线框经 ``CamLayoutAdapter``/本文件同款的 avg-pool + uint8 量化后，格内平移
    不变——前车在典型决策距离(20-50m)上逼近 1m，池化输出的 token 层 L1 变化不到
    线框自身能量的 3%，绝大多数在决策相关的相对速度下处于死区。实心填充的边界格
    覆盖率随位移连续变化，同等 Δd 下响应幅度实测高 5~20 倍，无该死区。

    与 ``render_wireframe`` 同一套投影/坐标/调用约定（``color`` 可以是 BGR 三元组
    或 ``(N,3)`` 逐框颜色数组，例如调用方传入 ``box_colors_bgr(...)`` 的结果），
    唯一差异是用凸包填充代替描边，因此丢失了线框的 3D 结构/朝向信息（重叠车辆会
    糊成一片）——这正是这个消融臂要单独验证的代价，若要两者兼得见 ``render_layout_3ch``。

    遮挡处理：按框中心相机系深度降序（远→近）依次填充，使近处框覆盖远处框的重叠
    区域，避免纯并集式填充在前后车重叠时"透视穿帮"（``render_wireframe`` 画线不需要
    这一步，因为线不互相覆盖也不影响可读性）。
    """
    if cam_model not in ('pinhole', 'fisheye'):
        raise ValueError(f'unknown cam_model: {cam_model!r}')
    h_img, w_img = image_size
    img = canvas if canvas is not None else np.zeros((h_img, w_img, 3), dtype=np.uint8)

    boxes = np.zeros((0, 7)) if boxes is None else np.asarray(boxes, dtype=np.float64)
    if boxes.shape[0] == 0:
        return img

    K = np.asarray(K, dtype=np.float64)
    imu2cam = np.asarray(imu2cam, dtype=np.float64)
    colors = np.broadcast_to(np.asarray(color), (boxes.shape[0], 3))

    corners_ego = _box_corners_ego(boxes)  # (N,8,3)
    corners_h = np.concatenate([corners_ego, np.ones((*corners_ego.shape[:2], 1))], axis=2)
    corners_cam = np.einsum('ij,nkj->nki', imu2cam, corners_h)[..., :3]  # (N,8,3)

    order = np.argsort(-corners_cam[:, :, 2].mean(axis=1))  # 深度降序: 远->近, 近处后画覆盖远处
    fill = _fill_box_color_fisheye if cam_model == 'fisheye' else _fill_box_color_pinhole
    for n in order:
        col = tuple(int(c) for c in colors[n])
        if cam_model == 'fisheye':
            fill(img, corners_cam[n], K, col, fisheye_segments, near_clip)
        else:
            fill(img, corners_cam[n], K, col, near_clip, w_img, h_img)
    return img


def render_lane_mask(
    lane_polys,
    border_polys,
    K,
    imu2cam,
    image_size,
    cam_model='pinhole',
    thickness=2,
    fisheye_segments=9,
    near_clip=0.1,
    canvas=None,
) -> np.ndarray:
    """车道分隔线(lane)+道路边界(border)的单通道结构掩码, 供 ``render_layout_3ch`` 的
    ch1(结构通道)叠加用。与 ``render_lane_wireframe`` 同一套投影, 只是不做类型上色
    (单通道装不下调色板, 与 ch1 里 box 边缘丢类别色是同一约束), 统一标量强度 255。
    不区分 lane/border 类型(两者都是"这里有道路结构"的同一种信号), 如需要保留类型
    区分, 用 ``render_lane_wireframe`` 走 3 通道彩色路径(``render_wireframe``/
    ``render_filled`` 两个模式用的就是它)。
    """
    if cam_model not in ('pinhole', 'fisheye'):
        raise ValueError(f'unknown cam_model: {cam_model!r}')
    h_img, w_img = image_size
    img = canvas if canvas is not None else np.zeros((h_img, w_img), dtype=np.uint8)
    K = np.asarray(K, dtype=np.float64)
    imu2cam = np.asarray(imu2cam, dtype=np.float64)
    draw = _draw_polyline_fisheye if cam_model == 'fisheye' else _draw_polyline_pinhole
    for polys in (lane_polys, border_polys):
        if polys is None or len(polys) == 0:
            continue
        polys = np.asarray(polys, dtype=np.float64)  # (N, pt_num, 3)
        pts_h = np.concatenate([polys, np.ones((*polys.shape[:2], 1))], axis=2)
        pts_cam = np.einsum('ij,nkj->nki', imu2cam, pts_h)[..., :3]  # (N, pt_num, 3)
        for n in range(polys.shape[0]):
            if polys[n].shape[0] < 2:
                continue
            if cam_model == 'fisheye':
                draw(img, pts_cam[n], K, 255, thickness, fisheye_segments, near_clip)
            else:
                draw(img, pts_cam[n], K, 255, thickness, near_clip, w_img, h_img)
    return img


def render_layout_3ch(
    boxes,
    velocities,
    K,
    imu2cam,
    image_size,
    cam_model='pinhole',
    near_clip=0.1,
    fisheye_segments=9,
    thickness=2,
    v_max=20.0,
    lane_polys=None,
    border_polys=None,
    canvas=None,
) -> np.ndarray:
    """三通道结构分解版 layout 条件（消融"3ch"臂）。

    现行 predictor 的 ``layout_in_chans=3`` 一直固定是 3——``render_wireframe`` 只是把
    这 3 个通道全部塞成同一套按类别上色的线框描边（无类别信息时 3 通道数值恒等，见
    ``box_colors_bgr`` 的 fallback）。本函数不改架构（仍输出 ``(H,W,3)`` uint8，Conv2d
    ``in_chans=3`` 无需改动），只是把 3 个通道的语义从"同一份描边的 3 份拷贝"换成
    3 个正交信号：

    - ch0（数组最后一维下标 0）= 实心剪影占据：任意框内=255，同 ``render_filled`` 的
      连续位置信号（见该函数 docstring 的灵敏度依据），用于"这里有没有东西/东西多大"。
      只画目标框，不含车道/道路（车道不是"占据"语义）；
    - ch1（下标 1）= 结构：车道分隔线/道路边界(``render_lane_mask``, 先画) + 目标框
      线框边缘(``render_wireframe`` 同款描边, 后画、叠在车道之上, 与其它模式"车道先画、
      框再叠"的既有约定一致)，统一标量强度 255（不做类别/车道类型上色——单通道装不下
      调色板，个体区分交给 ch0 的独立轮廓)；
    - ch2（下标 2）= 纵向相对速度编码：``128 + clip(v, -v_max, v_max)/v_max*127``，
      有效范围 ``[1,255]``；背景/速度未知严格留 ``0``（与"黑=0=空条件"的既有约定一致，
      不能和"静止 v=0 → 128"混淆），只画目标框（车道没有速度语义），用于把逼近快慢
      直接喂给 predictor，不逼它从相邻帧栅格差分里学 TTC。

    通道顺序沿用历史上 BGR 排布只是习惯延续，语义以上述定义为准——不是真的色彩图，
    下游 ``CamLayoutAdapter`` 只按 ``in_chans=3`` 吃张量。

    Args:
        velocities: ``(N,)`` 每个框的纵向相对速度（m/s），``NaN``/非有限值表示未知，
            对应框的 ch2 留背景 0（ch0/ch1 不受影响，仍正常画占据/边缘）。口径见调用方
            ``jepa_pipelines.py::FormatJEPAClip._render_layout`` 里从 ``gt_speed_ego``
            取值的说明与已知待验证项——训练前建议先挑 1 个真实前车匀速/刹车 clip 目视
            核对这条曲线与轨迹 GT 前后帧位置差分是否一致。
        v_max: 编码量程（m/s），超出部分截断，不是物理限速。
        lane_polys / border_polys: 同 ``render_lane_wireframe`` 的同名参数（ego 系
            ``(N, pt_num, 3)``），为空/None 时跳过、ch1 只有目标框边缘，不抛异常。不需要
            ``lane_labels``/``border_labels``——``render_lane_mask`` 不做类型上色。

    遮挡处理：ch0/ch2 按深度降序（远→近）逐框覆盖式填充，语义同 ``render_filled``；
    ch1 是描边不是填充，车道/边界/多个框的边线重叠时直接共存，不需要排序。
    """
    if cam_model not in ('pinhole', 'fisheye'):
        raise ValueError(f'unknown cam_model: {cam_model!r}')
    h_img, w_img = image_size
    if canvas is not None:
        ch0, ch1, ch2 = canvas[..., 0], canvas[..., 1], canvas[..., 2]
    else:
        ch0 = np.zeros((h_img, w_img), dtype=np.uint8)
        ch1 = np.zeros((h_img, w_img), dtype=np.uint8)
        ch2 = np.zeros((h_img, w_img), dtype=np.uint8)

    K = np.asarray(K, dtype=np.float64)
    imu2cam = np.asarray(imu2cam, dtype=np.float64)
    if lane_polys is not None or border_polys is not None:
        render_lane_mask(
            lane_polys,
            border_polys,
            K,
            imu2cam,
            image_size,
            cam_model=cam_model,
            thickness=thickness,
            fisheye_segments=fisheye_segments,
            near_clip=near_clip,
            canvas=ch1,
        )

    boxes = np.zeros((0, 7)) if boxes is None else np.asarray(boxes, dtype=np.float64)
    if boxes.shape[0] == 0:
        return np.stack([ch0, ch1, ch2], axis=-1).astype(np.uint8)

    velocities = np.full(boxes.shape[0], np.nan) if velocities is None else np.asarray(velocities, dtype=np.float64)

    corners_ego = _box_corners_ego(boxes)  # (N,8,3)
    corners_h = np.concatenate([corners_ego, np.ones((*corners_ego.shape[:2], 1))], axis=2)
    corners_cam = np.einsum('ij,nkj->nki', imu2cam, corners_h)[..., :3]  # (N,8,3)

    order = np.argsort(-corners_cam[:, :, 2].mean(axis=1))  # 深度降序: 远->近
    draw_edge = _draw_box_fisheye if cam_model == 'fisheye' else _draw_box_pinhole
    fill_scalar = _fill_box_fisheye if cam_model == 'fisheye' else _fill_box_pinhole
    for n in order:
        c = corners_cam[n]
        if cam_model == 'fisheye':
            draw_edge(ch1, c, K, 255, thickness, fisheye_segments, near_clip)
            fill_scalar(ch0, c, K, 255.0, fisheye_segments, near_clip)
        else:
            draw_edge(ch1, c, K, 255, thickness, near_clip, w_img, h_img)
            fill_scalar(ch0, c, K, 255.0, near_clip, w_img, h_img)
        v = velocities[n]
        if np.isfinite(v):
            val = float(np.clip(128.0 + np.clip(v, -v_max, v_max) / v_max * 127.0, 1.0, 255.0))
            if cam_model == 'fisheye':
                fill_scalar(ch2, c, K, val, fisheye_segments, near_clip)
            else:
                fill_scalar(ch2, c, K, val, near_clip, w_img, h_img)
    return np.stack([ch0, ch1, ch2], axis=-1).astype(np.uint8)


def render_object_weight(
    boxes,
    tier_keys,
    weights_by_tier: dict,
    K,
    imu2cam,
    image_size,
    cam_model='pinhole',
    background: float = 1.0,
    near_clip: float = 0.1,
    fisheye_segments: int = 9,
    canvas=None,
) -> np.ndarray:
    """把 ego 系 3D 框列表渲染成单通道 float32 对象级 loss 权重图。

    与 ``render_wireframe``(predictor 条件输入, 画线框)不同, 本函数产出的是 JEPA 训练
    loss 的空间加权 mask(实心填充, 供 ``jepa_world_model.py::_loss_fn`` 逐像素相乘), 对应
    路线文档 §6.1 "对象级监督"。是否属于某个 tier(前车/近距离障碍等)由调用方(见
    ``jepa_pipelines.py::FormatJEPAClip._render_obj_weight``)按类别+几何位置预先判定,
    本函数只负责投影与填充, 不关心具体语义, 保持与 ``render_wireframe`` 一样的纯投影职责。

    Args:
        boxes: (N,7) array-like [x,y,z,l,w,h,yaw], ego(imu) 系, 与 render_wireframe 同约定。
            为空/None 时返回纯背景权重图。
        tier_keys: 长度 N 的序列, 每个框对应的 tier key(如 'front_vehicle'/'obstacle');
            不在 ``weights_by_tier`` 里的 key 视为背景, 不填充(既不覆盖已画内容, 也不单独画)。
        weights_by_tier: ``{tier_key: weight_value}``。
        K / imu2cam / image_size / cam_model / near_clip / fisheye_segments: 同 render_wireframe。
        background: 画布初始权重值(默认 1.0, 与均匀 L1 的隐含权重一致, 即"不额外加权")。
        canvas: 若给定则原地画在此图上((H,W) float32), 否则新建纯背景画布。
    Returns:
        (H, W) float32。
    """
    if cam_model not in ('pinhole', 'fisheye'):
        raise ValueError(f'unknown cam_model: {cam_model!r}')
    h_img, w_img = image_size
    img = canvas if canvas is not None else np.full((h_img, w_img), float(background), dtype=np.float32)

    boxes = np.zeros((0, 7)) if boxes is None else np.asarray(boxes, dtype=np.float64)
    if boxes.shape[0] == 0:
        return img
    tier_keys = list(tier_keys)
    assert len(tier_keys) == boxes.shape[0], 'tier_keys 长度须与 boxes 一致'

    K = np.asarray(K, dtype=np.float64)
    imu2cam = np.asarray(imu2cam, dtype=np.float64)

    corners_ego = _box_corners_ego(boxes)  # (N,8,3)
    corners_h = np.concatenate([corners_ego, np.ones((*corners_ego.shape[:2], 1))], axis=2)
    corners_cam = np.einsum('ij,nkj->nki', imu2cam, corners_h)[..., :3]  # (N,8,3)

    # 按权重升序画: 权重更高(更重要)的框最后画、覆盖权重较低的框, 避免重叠区域被稀释。
    order = sorted(range(boxes.shape[0]), key=lambda n: weights_by_tier.get(tier_keys[n], background))
    for n in order:
        weight = weights_by_tier.get(tier_keys[n])
        if weight is None:
            continue  # 不在权重表里(背景/忽略类别): 保持画布底值, 不填充
        if cam_model == 'fisheye':
            _fill_box_fisheye(img, corners_cam[n], K, weight, fisheye_segments, near_clip)
        else:
            _fill_box_pinhole(img, corners_cam[n], K, weight, near_clip, w_img, h_img)
    return img


def render_wireframe(
    boxes,
    K,
    imu2cam,
    image_size,
    cam_model='pinhole',
    color=(255, 255, 255),
    thickness=2,
    fisheye_segments=9,
    near_clip=0.1,
    canvas=None,
) -> np.ndarray:
    """把 ego 系 3D 框列表渲染成线框图。

    Args:
        boxes: (N,7) array-like [x,y,z,l,w,h,yaw], ego(imu) 系, yaw 绕 Z。
            为空/None 时返回全黑图(黑底=0=空条件, 与 fastvideo 口径一致)。
        K: (3,3) 相机内参。
        imu2cam: (4,4) ego->相机外参, 与 simulator gs_model.get_imu2cam() /
            本仓 inv(cam2imu) 同约定。
        image_size: (H, W)。
        cam_model: 'pinhole' 或 'fisheye'。
        color: BGR 三元组(全部框同色), 或 (N,3) 逐框颜色。
        canvas: 若给定则原地画在此图上((H,W,3) uint8), 否则新建全黑画布。
    Returns:
        (H, W, 3) uint8。
    """
    if cam_model not in ('pinhole', 'fisheye'):
        raise ValueError(f'unknown cam_model: {cam_model!r}')
    h_img, w_img = image_size
    img = canvas if canvas is not None else np.zeros((h_img, w_img, 3), dtype=np.uint8)

    boxes = np.zeros((0, 7)) if boxes is None else np.asarray(boxes, dtype=np.float64)
    if boxes.shape[0] == 0:
        return img

    K = np.asarray(K, dtype=np.float64)
    imu2cam = np.asarray(imu2cam, dtype=np.float64)
    colors = np.broadcast_to(np.asarray(color), (boxes.shape[0], 3))

    corners_ego = _box_corners_ego(boxes)  # (N,8,3)
    corners_h = np.concatenate([corners_ego, np.ones((*corners_ego.shape[:2], 1))], axis=2)
    corners_cam = np.einsum('ij,nkj->nki', imu2cam, corners_h)[..., :3]  # (N,8,3)

    draw_box = _draw_box_fisheye if cam_model == 'fisheye' else _draw_box_pinhole
    for n in range(boxes.shape[0]):
        col = tuple(int(c) for c in colors[n])
        if cam_model == 'fisheye':
            draw_box(img, corners_cam[n], K, col, thickness, fisheye_segments, near_clip)
        else:
            draw_box(img, corners_cam[n], K, col, thickness, near_clip, w_img, h_img)
    return img


def _draw_polyline_pinhole(img, pts_cam, K, color, thickness, near_clip, w_img, h_img):
    """逐点折线版 ``_draw_box_pinhole``: 边 = 相邻点对, 而不是固定的 12 条 box 棱边。
    裁剪规则(相机后方跳过、数值爆炸跳过)与 ``_draw_box_pinhole`` 完全一致, 单独实现
    (而不是让两者共用一份代码)是为了不改动已在真实 checkpoint/数据上验证过的 box 渲染路径。"""
    z = pts_cam[:, 2]
    if (z <= near_clip).all():
        return
    uv = K @ pts_cam.T
    uv = uv[:2] / np.clip(uv[2:3], 1e-6, None)
    for i in range(len(pts_cam) - 1):
        j = i + 1
        if z[i] <= near_clip or z[j] <= near_clip:
            continue
        if np.abs(uv[:, [i, j]]).max() > 8 * max(w_img, h_img):
            continue
        p1 = (round(uv[0, i]), round(uv[1, i]))
        p2 = (round(uv[0, j]), round(uv[1, j]))
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def _draw_polyline_fisheye(img, pts_cam, K, color, thickness, segments, near_clip):
    """逐点折线版 ``_draw_box_fisheye``: 边 = 相邻点对。等距投影公式、逐边等分插值裁剪
    与 ``_draw_box_fisheye`` 完全一致, 理由同 ``_draw_polyline_pinhole``。"""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    for i in range(len(pts_cam) - 1):
        p_i, p_j = pts_cam[i], pts_cam[i + 1]
        if p_i[2] <= near_clip and p_j[2] <= near_clip:
            continue
        t = np.linspace(0.0, 1.0, segments + 1)[:, None]
        pts = p_i[None, :] * (1 - t) + p_j[None, :] * t  # (segments+1, 3)
        valid = pts[:, 2] > near_clip
        if not valid.any():
            continue
        r = np.clip(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2), 1e-6, None)
        theta = np.arctan2(r, pts[:, 2])
        u = fx * theta * pts[:, 0] / r + cx
        v = fy * theta * pts[:, 1] / r + cy
        line_pts = np.stack([u, v], axis=1)[valid].astype(np.int32)
        if line_pts.shape[0] >= 2:
            cv2.polylines(img, [line_pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def render_lane_wireframe(
    lane_polys,
    lane_labels,
    border_polys,
    border_labels,
    K,
    imu2cam,
    image_size,
    cam_model='pinhole',
    thickness=2,
    fisheye_segments=9,
    near_clip=0.1,
    canvas=None,
) -> np.ndarray:
    """把 ego 系车道折线(lane/border)渲染成线框图, 与 ``render_wireframe`` 是同一套
    投影公式(见文件头), 只是边 = 折线里相邻的点对而不是 box 的 12 条棱边。

    Args:
        lane_polys / border_polys: (N, pt_num, 3) array-like, ego(imu) 系, 来自
            ``GenLanePoly3D`` 产出的 ``gt_lane_polys``/``gt_border_polys``。为空/None
            时该组直接跳过, 不抛异常。
        lane_labels / border_labels: (N,) 每条折线的类型 code(``gt_..._polys_labels``),
            按 ``_LANE_TYPE_COLORS_BGR``/``_BORDER_TYPE_COLORS_BGR`` 上色, 未知/负值兜底白色。
        K / imu2cam / image_size / cam_model / near_clip / fisheye_segments: 同 ``render_wireframe``。
        canvas: 若给定则原地画在此图上((H,W,3) uint8, 通常是已画好车道线、还要再叠 box 的画布,
            或反过来先画车道再叠 box), 否则新建全黑画布。
    Returns:
        (H, W, 3) uint8。
    """
    if cam_model not in ('pinhole', 'fisheye'):
        raise ValueError(f'unknown cam_model: {cam_model!r}')
    h_img, w_img = image_size
    img = canvas if canvas is not None else np.zeros((h_img, w_img, 3), dtype=np.uint8)

    K = np.asarray(K, dtype=np.float64)
    imu2cam = np.asarray(imu2cam, dtype=np.float64)
    draw = _draw_polyline_fisheye if cam_model == 'fisheye' else _draw_polyline_pinhole

    for polys, labels, color_table in (
        (lane_polys, lane_labels, _LANE_TYPE_COLORS_BGR),
        (border_polys, border_labels, _BORDER_TYPE_COLORS_BGR),
    ):
        if polys is None or len(polys) == 0:
            continue
        polys = np.asarray(polys, dtype=np.float64)  # (N, pt_num, 3)
        n_inst = polys.shape[0]
        labels = np.full(n_inst, -1, dtype=np.int64) if labels is None else np.asarray(labels).reshape(-1)

        pts_h = np.concatenate([polys, np.ones((*polys.shape[:2], 1))], axis=2)
        pts_cam = np.einsum('ij,nkj->nki', imu2cam, pts_h)[..., :3]  # (N, pt_num, 3)

        for n in range(n_inst):
            if polys[n].shape[0] < 2:
                continue
            label = int(labels[n]) if n < len(labels) else -1
            color = color_table.get(label, color_table[-1])
            if cam_model == 'fisheye':
                draw(img, pts_cam[n], K, color, thickness, fisheye_segments, near_clip)
            else:
                draw(img, pts_cam[n], K, color, thickness, near_clip, w_img, h_img)
    return img


if __name__ == '__main__':
    cols = box_colors_bgr(4, [0, 1, -1, 99])
    assert tuple(cols[0]) == (0, 140, 255) and tuple(cols[1]) == (0, 225, 0)
    assert tuple(cols[2]) == tuple(cols[3]) == (0, 69, 255)
    assert np.all(box_colors_bgr(2, None) == 255)
    K = np.array([[200.0, 0, 40], [0, 200.0, 40], [0, 0, 1]])
    box = np.array([[0, 0, 8, 2, 2, 2, 0]])
    img = render_wireframe(box, K, np.eye(4), (80, 80), color=box_colors_bgr(1, [0]))
    assert (img == np.array((0, 140, 255), np.uint8)).all(axis=2).any()
    assert not (img == 255).all(axis=2).any()
    print('[box color] car/ped/dontcare/fallback + pinhole paint: PASS')
