"""BEVFormerTrackHead（推理部分）。

搬自 UniAD `uniad/dense_heads/track_head.py`。官方继承 mmdet 2.14 的 `DETRHead`（新栈 mmdet 3.3 的 DETRHead 结构完全不同），
这里只保留推理需要的构造与 `get_bev_features` / `get_detections`；损失/匹配相关代码不移植。
`DETRHead.__init__` 在本 head 中实际产生的成员：positional_encoding（LearnedPositionalEncoding）、transformer、
embed_dims、cls_out_channels(=num_classes，sigmoid)、num_reg_fcs=2，其余是训练用的。参数名/形状与 ckpt 一一对应。
"""
import copy

import torch
import torch.nn as nn
from mmengine.registry import MODELS
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence  # noqa: F401

from ..utils import inverse_sigmoid
from ..modules import transformer as _t  # noqa: F401  (registry side-effect: PerceptionTransformer)


class LearnedPositionalEncoding(nn.Module):
    """mmdet 2.14 `LearnedPositionalEncoding`（行/列可学习 embedding），逐行等价。"""

    def __init__(self, num_feats, row_num_embed=50, col_num_embed=50):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(1, w, 1)),
            dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos


class BEVFormerTrackHead(nn.Module):
    """Head of Detr3D / BEVFormer（UniAD 版，推理）。"""

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 past_steps=4,
                 fut_steps=4,
                 num_query=100,
                 num_classes=10,
                 in_channels=256,
                 num_reg_fcs=2,
                 positional_encoding=None,
                 **kwargs):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.with_box_refine = with_box_refine
        assert as_two_stage is False, 'as_two_stage is not supported yet.'
        self.as_two_stage = as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.pc_range = list(bbox_coder['pc_range'])
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        self.past_steps = past_steps
        self.fut_steps = fut_steps

        # ---- 以下对应 DETRHead.__init__ 里推理相关的部分 ----
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.cls_out_channels = num_classes  # loss_cls.use_sigmoid=True
        self.positional_encoding = LearnedPositionalEncoding(
            positional_encoding['num_feats'], positional_encoding['row_num_embed'],
            positional_encoding['col_num_embed'])
        self.transformer = MODELS.build(transformer)
        self.embed_dims = self.transformer.embed_dims
        assert positional_encoding['num_feats'] * 2 == self.embed_dims
        self._init_layers()
        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False), requires_grad=False)

    def _init_layers(self):
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        past_traj_reg_branch = []
        for _ in range(self.num_reg_fcs):
            past_traj_reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            past_traj_reg_branch.append(nn.ReLU())
        past_traj_reg_branch.append(nn.Linear(self.embed_dims, (self.past_steps + self.fut_steps) * 2))
        past_traj_reg_branch = nn.Sequential(*past_traj_reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = (self.transformer.decoder.num_layers + 1) if self.as_two_stage else \
            self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.past_traj_reg_branches = _get_clones(past_traj_reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])
            self.past_traj_reg_branches = nn.ModuleList([past_traj_reg_branch for _ in range(num_pred)])
        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

    def get_bev_features(self, mlvl_feats, img_metas, prev_bev=None):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_embed = self.transformer.get_bev_features(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            self.real_h,
            self.real_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            img_metas=img_metas,
        )
        return bev_embed, bev_pos

    def get_detections(self, bev_embed, object_query_embeds=None, ref_points=None, img_metas=None):
        assert bev_embed.shape[0] == self.bev_h * self.bev_w
        hs, init_reference, inter_references = self.transformer.get_states_and_refs(
            bev_embed,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            reference_points=ref_points,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
        )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_trajs = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = ref_points.sigmoid()
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])  # xydxdyxdz
            outputs_past_traj = self.past_traj_reg_branches[lvl](hs[lvl]).view(
                tmp.shape[0], -1, self.past_steps + self.fut_steps, 2)
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            last_ref_points = torch.cat([tmp[..., 0:2], tmp[..., 4:5]], dim=-1)

            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_trajs.append(outputs_past_traj)
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_trajs = torch.stack(outputs_trajs)
        last_ref_points = inverse_sigmoid(last_ref_points)
        outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'all_past_traj_preds': outputs_trajs,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'last_ref_points': last_ref_points,
            'query_feats': hs,
        }
        return outs
