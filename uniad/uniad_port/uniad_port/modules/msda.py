"""MultiScaleDeformableAttn 调用封装。

官方 `multi_scale_deformable_attn_function.py` 里的 `MultiScaleDeformableAttnFunction_fp32/fp16` 是对 mmcv-full 1.x
CUDA 算子的包装（custom_fwd 把输入转 fp32）。新栈直接用 mmcv 2.2 的
`mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction`（同一个 `_ext.ms_deform_attn_forward` 算子，

"""
import torch
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)


def msda_fp32(value, spatial_shapes, level_start_index, sampling_locations,
              attention_weights, im2col_step=64):
    """等价于官方 `MultiScaleDeformableAttnFunction_fp32.apply(...)`（CUDA 版）或 pytorch 兜底。

    形状：value (bs, num_keys, heads, dim/heads)；sampling_locations (bs, nq, heads, levels, points, 2)；
    attention_weights (bs, nq, heads, levels, points)。返回 (bs, nq, embed_dims)。
    """
    if value.is_cuda:
        with torch.autocast(device_type='cuda', enabled=False):
            return MultiScaleDeformableAttnFunction.apply(
                value.float(), spatial_shapes, level_start_index,
                sampling_locations.float(), attention_weights.float(), im2col_step)
    return multi_scale_deformable_attn_pytorch(
        value, spatial_shapes, sampling_locations, attention_weights)
