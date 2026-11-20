"""mmdet 2.14 的 DETR 系 Transformer 组件（mmdet 3.x 已删除/重构，key 命名也不同）。

UniAD 的 seg_head（SegDeformableTransformer）与 occ_head（transformer_decoder）用的是 mmdet 2.x 的
`DetrTransformerEncoder/Decoder`、`DetrTransformerDecoderLayer`、`DeformableDetrTransformerDecoder`，
权重 key 为 `attentions.i / ffns.i / norms.i` 风格（即 mmcv 的 BaseTransformerLayer）。这里按 mmdet 2.14 源码重写，
底层仍用 mmcv 2.2 的 `BaseTransformerLayer` / `TransformerLayerSequence`（二者与 mmcv 1.4 同构）。

对应 mmdet 2.14 源码：
  mmdet/models/utils/transformer.py: DetrTransformerEncoder / DetrTransformerDecoder / DetrTransformerDecoderLayer / Transformer
  mmdet/models/utils/transformer.py: DeformableDetrTransformerDecoder
"""
import torch
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer, TransformerLayerSequence,
                                         build_transformer_layer_sequence)

from ..utils import inverse_sigmoid


@MODELS.register_module(force=True)
class DetrTransformerEncoder(TransformerLayerSequence):
    """mmdet 2.14 DetrTransformerEncoder：TransformerLayerSequence + 可选 post_norm（仅 pre_norm 时才建）。"""

    def __init__(self, *args, post_norm_cfg=dict(type='LN'), **kwargs):
        super().__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(
                post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            assert not self.pre_norm, f'Use prenorm in {self.__class__.__name__}, Please specify post_norm_cfg'
            self.post_norm = None

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        if self.post_norm is not None:
            x = self.post_norm(x)
        return x


@MODELS.register_module(force=True)
class DetrTransformerDecoder(TransformerLayerSequence):
    """mmdet 2.14 DetrTransformerDecoder（始终建 post_norm=LN；occ_head 只用其 .layers，但 post_norm 参数在 ckpt 里）。"""

    def __init__(self, *args, post_norm_cfg=dict(type='LN'), return_intermediate=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg, self.embed_dims)[1]
        else:
            self.post_norm = None

    def forward(self, query, *args, **kwargs):
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        intermediate = []
        for layer in self.layers:
            query = layer(query, *args, **kwargs)
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
        return torch.stack(intermediate)


@MODELS.register_module(force=True)
class DetrTransformerDecoderLayer(BaseTransformerLayer):
    """mmdet 2.14 DetrTransformerDecoderLayer。"""

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super().__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(['self_attn', 'norm', 'cross_attn', 'ffn'])


@MODELS.register_module(force=True)
class DeformableDetrTransformerDecoder(TransformerLayerSequence):
    """mmdet 2.14 DeformableDetrTransformerDecoder（with_box_refine 时用 reg_branches 迭代修正参考点）。

    注意 2.14 的一个行为：2 维参考点 + reg_branches 时，第 0 层之后 reference_points 变成 4 维（x,y 加偏移，w,h 为
    reg 分支原始输出再 sigmoid）；PansegformerHead.forward 里按 `reference.shape[-1]==4/2` 两种情况处理。
    mmdet 3.3 的 DeformableDetrTransformerDecoder 有同样逻辑（已对照）。
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                **kwargs):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * \
                    valid_ratios[:, None]
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


class DetrTransformerBase(BaseModule):
    """mmdet 2.14 `Transformer` 基类：按 cfg 建 encoder / decoder。"""

    def __init__(self, encoder=None, decoder=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.encoder.embed_dims
