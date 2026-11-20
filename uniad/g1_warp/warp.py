"""FlowSim 的 Warp（R=0）基线：把锚帧真实 BEV 按相对自车运动重采样到目标帧的自车（LiDAR_TOP）系，不预测残差。

BEV 栅格约定（与 BEVFormer encoder 的 SCA ref_3d / UniAD seg 栅格一致，见 uniad_port modules/encoder.py
`get_reference_points`）：bev_embed [H*W, 1, C]，flatten 下标 = row*W + col；
  col w ↔ LiDAR x = (w + 0.5) / W * 102.4 - 51.2（x 右），row h ↔ LiDAR y = (h + 0.5) / H * 102.4 - 51.2（y 前）。
这个约定对 GitHub v1.0.1 与 HF 2.0 两份 ckpt 相同（compat.py 的差异都在 BEV encoder 内部的 shift / ref_2d 与 head 的
后处理，不改 BEV 栅格的几何含义）。LiDAR 系本身在两个 mmdet3d 版本下也相同（变的是框的 dims/yaw 命名）。

grid_sample(align_corners=False) 下归一化坐标 u = x / 51.2、v = y / 51.2 正好落在格心，所以零运动时是恒等
（float32 网格坐标的舍入带来 ~1e-4 量级的插值残差）。
位姿用 UniAD 的 l2g_r_mat / l2g_t（**行向量约定**，见 rel_points）。
"""
import numpy as np
import torch
import torch.nn.functional as F

BEV_H = BEV_W = 200
HALF = 51.2


def rel_points(l2g_r_src, l2g_t_src, l2g_r_dst, l2g_t_dst, bev_h=BEV_H, bev_w=BEV_W, half=HALF, z=0.0):
    """目标帧（dst）每个 BEV 格心在源帧（src）LiDAR 系下的 (x, y)，float64 numpy [H, W, 2]。

    ⚠️ l2g_r_mat 用 UniAD 的**行向量约定**（data.py `_l2g`：l2g_r_mat = l2e_r.T @ e2g_r.T，velo_update 里
    `p_g = p @ l2g_r_mat + l2g_t`），不是列向量的旋转矩阵。于是
        p_g = p_dst @ M_dst + t_dst ；p_src = (p_g - t_src) @ M_src^T（M 正交）。"""
    Ms = np.asarray(l2g_r_src, np.float64).reshape(3, 3)
    ts = np.asarray(l2g_t_src, np.float64).reshape(3)
    Md = np.asarray(l2g_r_dst, np.float64).reshape(3, 3)
    td = np.asarray(l2g_t_dst, np.float64).reshape(3)
    xs = (np.arange(bev_w) + 0.5) / bev_w * 2 * half - half
    ys = (np.arange(bev_h) + 0.5) / bev_h * 2 * half - half
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    P = np.stack([X, Y, np.full_like(X, z)], -1).reshape(-1, 3)          # dst LiDAR
    Pg = P @ Md + td
    Ps = (Pg - ts) @ Ms.T
    return Ps[:, :2].reshape(bev_h, bev_w, 2)


@torch.no_grad()
def warp_bev(bev, l2g_r_src, l2g_t_src, l2g_r_dst, l2g_t_dst, mode='bilinear', padding_mode='zeros',
             bev_h=BEV_H, bev_w=BEV_W, half=HALF, return_valid=False):
    """bev: [H*W, 1, C]（或 [H*W, C]），src 帧的 BEV -> [H*W, 1, C]，dst 帧自车系下的 BEV。
    视野外（源帧 ±51.2 m 以外）按 padding_mode 填（默认 0）。return_valid=True 时另返回 [H, W] bool（采样点在源栅格内）。"""
    squeeze = bev.dim() == 2
    b = bev[:, None] if squeeze else bev
    C = b.shape[-1]
    img = b[:, 0].reshape(bev_h, bev_w, C).permute(2, 0, 1)[None].float()   # [1,C,H,W]，row↔y、col↔x
    xy = rel_points(l2g_r_src, l2g_t_src, l2g_r_dst, l2g_t_dst, bev_h, bev_w, half)
    grid = torch.from_numpy(xy / half).to(device=img.device, dtype=torch.float32)[None]   # [1,H,W,2]，(u=x, v=y)
    out = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=False)
    out = out[0].permute(1, 2, 0).reshape(bev_h * bev_w, 1, C).to(bev.dtype)
    if squeeze:
        out = out[:, 0]
    if return_valid:
        valid = torch.from_numpy((np.abs(xy) <= half).all(-1)).to(img.device)
        return out, valid
    return out


def warp_frames(bev, fr_src, fr_dst, **kw):
    """便捷：fr_* 是 UniADValData.get_frame() 的 dict（用其 l2g_r_mat / l2g_t）。"""
    return warp_bev(bev, fr_src['l2g_r_mat'], fr_src['l2g_t'], fr_dst['l2g_r_mat'], fr_dst['l2g_t'], **kw)


ARMS = ('nowarp', 'warp', 'warp_border', 'warp_fillsrc', 'warp_rot')


def warp_variant(arm, bev_a, fr_a, fr_t):
    """论文没写视野外（源帧 ±51.2 m 以外）怎么填，这里都实现：
    warp          视野外填 0（grid_sample 默认）
    warp_border   视野外取最近边界格（padding_mode='border'）
    warp_fillsrc  视野外用锚帧同一格的原特征（不 warp）补
    warp_rot      只按航向差绕自车中心旋转、不平移（相当于 BEVFormer 对 prev_bev 的 rotate 对齐），视野外填 0
    nowarp        不 warp，直接用锚帧 BEV（BEV 层的 copy-last，作参照）"""
    if arm == 'nowarp':
        return bev_a
    if arm == 'warp':
        return warp_frames(bev_a, fr_a, fr_t)
    if arm == 'warp_border':
        return warp_frames(bev_a, fr_a, fr_t, padding_mode='border')
    if arm == 'warp_fillsrc':
        W, valid = warp_frames(bev_a, fr_a, fr_t, return_valid=True)
        v = valid.reshape(-1, 1, 1).to(W.dtype)
        return W * v + bev_a * (1 - v)
    if arm == 'warp_rot':
        return warp_bev(bev_a, fr_a['l2g_r_mat'], fr_a['l2g_t'], fr_t['l2g_r_mat'], fr_a['l2g_t'])
    raise ValueError(arm)


def rel_xy(pts_dst, l2g_r_src, l2g_t_src, l2g_r_dst, l2g_t_dst, z=0.0):
    """rel_points 的逐点版（11-07 追加，B 表 warp 臂自检用）：目标帧 LiDAR 系下任意点 pts_dst [N,2] → 源帧 LiDAR 系 (x, y) [N,2]，同一行向量约定。"""
    Ms = np.asarray(l2g_r_src, np.float64).reshape(3, 3)
    ts = np.asarray(l2g_t_src, np.float64).reshape(3)
    Md = np.asarray(l2g_r_dst, np.float64).reshape(3, 3)
    td = np.asarray(l2g_t_dst, np.float64).reshape(3)
    P = np.concatenate([np.asarray(pts_dst, np.float64).reshape(-1, 2), np.full((len(pts_dst), 1), z)], 1)
    return ((P @ Md + td - ts) @ Ms.T)[:, :2]
