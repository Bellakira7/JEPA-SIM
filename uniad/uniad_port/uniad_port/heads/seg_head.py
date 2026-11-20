"""PansegformerHead（map 分割 head，推理部分）+ SegDeformableTransformer + SinePositionalEncoding。

搬自 UniAD：
  uniad/dense_heads/panseg_head.py（forward / get_bboxes）
  uniad/dense_heads/seg_head_plugin/seg_deformable_transformer.py（SegDeformableTransformer）
  mmdet 2.14：SinePositionalEncoding / SegDETRHead.__init__ 的推理相关成员
改动：
  - 官方 PansegformerHead 继承 mmdet 2.14 `SegDETRHead(AnchorFreeHead)`；这里直接写成 nn.Module，仅保留推理成员；
  - forward_test 去掉对 GT（gt_lane_masks/IoU）的计算；get_bboxes 去掉 panoptic_list（需要 pts_filename，推理不用）；
  - 去掉 force_fp32；其余逐行搬运。
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS
from torch.nn.init import normal_

from ..utils import inverse_sigmoid
from ..modules.detr_layers import DetrTransformerBase
from .seg_mask_head import SegMaskHead


# ----------------------------------------------------------------------------
# mmdet 2.14 SinePositionalEncoding
# ----------------------------------------------------------------------------
class SinePositionalEncoding(nn.Module):

    def __init__(self, num_feats, temperature=10000, normalize=False, scale=2 * math.pi, eps=1e-6, offset=0.):
        super().__init__()
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set, scale should be provided and in float or int type'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        B, H, W = mask.size()
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).view(B, H, W, -1)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# ----------------------------------------------------------------------------
# SegDeformableTransformer
# ----------------------------------------------------------------------------
class SegDeformableTransformer(DetrTransformerBase):

    def __init__(self, as_two_stage=False, num_feature_levels=4, two_stage_num_proposals=300, **kwargs):
        super().__init__(**kwargs)
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        self.embed_dims = self.encoder.embed_dims
        self.init_layers()

    def init_layers(self):
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        assert not self.as_two_stage
        self.reference_points = nn.Linear(self.embed_dims, 2)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, mlvl_feats, mlvl_masks, query_embed, mlvl_pos_embeds, reg_branches=None, cls_branches=None, **kwargs):
        assert self.as_two_stage or query_embed is not None
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)

        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        query_pos, query = torch.split(query_embed, c, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos).sigmoid()
        init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs)
        inter_references_out = inter_references
        return (memory, lvl_pos_embed_flatten, mask_flatten, query_pos), inter_states, init_reference_out, \
            inter_references_out, None, None


def _bbox_cxcywh_to_xyxy(bbox):
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)


# ----------------------------------------------------------------------------
# PansegformerHead
# ----------------------------------------------------------------------------
class PansegformerHead(nn.Module):
    """Panoptic-SegFormer 风格的 map 分割 head（UniAD 版），仅推理。"""

    def __init__(self,
                 bev_h, bev_w, canvas_size, pc_range,
                 num_query=100, num_classes=4, num_things_classes=3, num_stuff_classes=1,
                 in_channels=2048, num_reg_fcs=2,
                 with_box_refine=False, as_two_stage=False,
                 transformer=None, positional_encoding=None,
                 quality_threshold_things=0.25, quality_threshold_stuff=0.25,
                 overlap_threshold_things=0.4, overlap_threshold_stuff=0.2,
                 thing_transformer_head=None, stuff_transformer_head=None,
                 test_cfg=dict(max_per_img=100),
                 **kwargs):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.canvas_size = canvas_size
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        assert not as_two_stage
        # NOTE: 官方 __init__ 里把 quality_threshold_things 硬编码为 0.1（忽略传参），这里保持一致
        self.quality_threshold_things = 0.1
        self.quality_threshold_stuff = quality_threshold_stuff
        self.overlap_threshold_things = overlap_threshold_things
        self.overlap_threshold_stuff = overlap_threshold_stuff
        self.num_dec_things = thing_transformer_head['num_decoder_layers']
        self.num_dec_stuff = stuff_transformer_head['num_decoder_layers']

        self.num_query = num_query
        self.num_classes = num_classes
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.test_cfg = test_cfg
        self.cls_out_channels = num_things_classes  # loss_cls.use_sigmoid=True
        self.positional_encoding = SinePositionalEncoding(
            positional_encoding['num_feats'], normalize=positional_encoding.get('normalize', False),
            offset=positional_encoding.get('offset', 0.))
        t_cfg = {k: v for k, v in transformer.items() if k != 'type'}
        self.transformer = SegDeformableTransformer(**t_cfg)
        self.embed_dims = self.transformer.embed_dims
        assert positional_encoding['num_feats'] * 2 == self.embed_dims

        self.things_mask_head = SegMaskHead(**{k: v for k, v in thing_transformer_head.items() if k != 'type'})
        self.stuff_mask_head = SegMaskHead(**{k: v for k, v in stuff_transformer_head.items() if k != 'type'})
        self._init_layers()

    def _init_layers(self):
        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

        fc_cls = nn.Linear(self.embed_dims, self.cls_out_channels)
        fc_cls_stuff = nn.Linear(self.embed_dims, 1)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = (self.transformer.decoder.num_layers + 1) if self.as_two_stage else \
            self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
        self.stuff_query = nn.Embedding(self.num_stuff_classes, self.embed_dims * 2)
        self.reg_branches2 = _get_clones(reg_branch, self.num_dec_things)  # used in mask decoder
        self.cls_thing_branches = _get_clones(fc_cls, self.num_dec_things)  # used in mask decoder
        self.cls_stuff_branches = _get_clones(fc_cls_stuff, self.num_dec_stuff)  # used in mask decoder

    def forward(self, bev_embed):
        _, bs, _ = bev_embed.shape

        mlvl_feats = [torch.reshape(bev_embed, (bs, self.bev_h, self.bev_w, -1)).permute(0, 3, 1, 2)]
        img_masks = mlvl_feats[0].new_zeros((bs, self.bev_h, self.bev_w))

        hw_lvl = [feat_lvl.shape[-2:] for feat_lvl in mlvl_feats]
        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight
        (memory, memory_pos, memory_mask, query_pos), hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_coord = self.transformer(
                mlvl_feats,
                mlvl_masks,
                query_embeds,
                mlvl_positional_encodings,
                reg_branches=self.reg_branches if self.with_box_refine else None,
                cls_branches=self.cls_branches if self.as_two_stage else None)

        memory = memory.permute(1, 0, 2)
        query = hs[-1].permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        memory_pos = memory_pos.permute(1, 0, 2)

        # we should feed these to mask deocder.
        args_tuple = [memory, memory_mask, memory_pos, query, None, query_pos, hw_lvl]

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            'bev_embed': bev_embed,
            'outputs_classes': outputs_classes,
            'outputs_coords': outputs_coords,
            'enc_outputs_class': None,
            'enc_outputs_coord': None,
            'args_tuple': args_tuple,
            'reference': reference,
        }
        return outs

    def forward_test(self, pts_feats=None, img_metas=None, rescale=False):
        bbox_list = [dict() for i in range(len(img_metas))]

        pred_seg_dict = self(pts_feats)
        results = self.get_bboxes(pred_seg_dict['outputs_classes'],
                                  pred_seg_dict['outputs_coords'],
                                  pred_seg_dict['enc_outputs_class'],
                                  pred_seg_dict['enc_outputs_coord'],
                                  pred_seg_dict['args_tuple'],
                                  pred_seg_dict['reference'],
                                  img_metas,
                                  rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, results):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['args_tuple'] = pred_seg_dict['args_tuple']
        return bbox_list

    def _get_bboxes_single(self, cls_score, bbox_pred, img_shape, scale_factor, rescale=False):
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)

        # exclude background (use_sigmoid)
        cls_score = cls_score.sigmoid()
        scores, indexes = cls_score.view(-1).topk(max_per_img)
        det_labels = indexes % self.num_things_classes
        bbox_index = indexes // self.num_things_classes
        bbox_pred = bbox_pred[bbox_index]

        det_bboxes = _bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        return bbox_index, det_bboxes, det_labels

    def get_bboxes(self, all_cls_scores, all_bbox_preds, enc_cls_scores, enc_bbox_preds, args_tuple, reference,
                   img_metas, rescale=False):
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        memory, memory_mask, memory_pos, query, _, query_pos, hw_lvl = args_tuple

        seg_list = []
        stuff_score_list = []
        bbox_list = []
        labels_list = []
        drivable_list = []
        lane_list = []
        lane_score_list = []
        score_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = (self.canvas_size[0], self.canvas_size[1], 3)
            ori_shape = (self.canvas_size[0], self.canvas_size[1], 3)
            scale_factor = 1

            index, bbox, labels = self._get_bboxes_single(cls_score, bbox_pred, img_shape, scale_factor, rescale)

            i = img_id
            thing_query = query[i:i + 1, index, :]
            thing_query_pos = query_pos[i:i + 1, index, :]
            joint_query = torch.cat([thing_query, self.stuff_query.weight[None, :, :self.embed_dims]], 1)

            stuff_query_pos = self.stuff_query.weight[None, :, self.embed_dims:]

            mask_things, mask_inter_things, query_inter_things = self.things_mask_head(
                memory[i:i + 1], memory_mask[i:i + 1], None,
                joint_query[:, :-self.num_stuff_classes], None, None, hw_lvl=hw_lvl)
            mask_stuff, mask_inter_stuff, query_inter_stuff = self.stuff_mask_head(
                memory[i:i + 1], memory_mask[i:i + 1], None,
                joint_query[:, -self.num_stuff_classes:], None, stuff_query_pos, hw_lvl=hw_lvl)

            attn_map = torch.cat([mask_things, mask_stuff], 1)
            attn_map = attn_map.squeeze(-1)  # BS, NQ, N_head,LEN

            stuff_query = query_inter_stuff[-1]
            scores_stuff = self.cls_stuff_branches[-1](stuff_query).sigmoid().reshape(-1)  # noqa: F841

            mask_pred = attn_map.reshape(-1, *hw_lvl[0])

            mask_pred = F.interpolate(mask_pred.unsqueeze(0), size=ori_shape[:2], mode='bilinear').squeeze(0)

            masks_all = mask_pred
            score_list.append(masks_all)
            drivable_list.append(masks_all[-1] > 0.5)
            masks_all = masks_all[:-self.num_stuff_classes]
            seg_all = masks_all > 0.5
            sum_seg_all = seg_all.sum((1, 2)).float() + 1
            scores_all = bbox[:, -1]
            bboxes_all = bbox
            labels_all = labels

            # mask wise merging
            seg_scores = (masks_all * seg_all.float()).sum((1, 2)) / sum_seg_all
            scores_all *= (seg_scores ** 2)

            scores_all, index = torch.sort(scores_all, descending=True)

            masks_all = masks_all[index]
            labels_all = labels_all[index]
            bboxes_all = bboxes_all[index]
            seg_all = seg_all[index]

            bboxes_all[:, -1] = scores_all

            # MDS: select things for instance segmeantion
            things_selected = labels_all < self.num_things_classes
            stuff_selected = labels_all >= self.num_things_classes
            bbox_th = bboxes_all[things_selected][:100]
            labels_th = labels_all[things_selected][:100]
            seg_th = seg_all[things_selected][:100]
            labels_st = labels_all[stuff_selected]  # noqa: F841
            scores_st = scores_all[stuff_selected]
            masks_st = masks_all[stuff_selected]  # noqa: F841

            stuff_score_list.append(scores_st)

            results = torch.zeros((2, *mask_pred.shape[-2:]), device=mask_pred.device).to(torch.long)
            id_unique = 1
            lane = torch.zeros((self.num_things_classes, *mask_pred.shape[-2:]), device=mask_pred.device).to(torch.long)
            lane_score = torch.zeros((self.num_things_classes, *mask_pred.shape[-2:]),
                                     device=mask_pred.device).to(mask_pred.dtype)
            for i, scores in enumerate(scores_all):
                # MDS: things and sutff have different threholds may perform a little bit better
                if labels_all[i] < self.num_things_classes and scores < self.quality_threshold_things:
                    continue
                elif labels_all[i] >= self.num_things_classes and scores < self.quality_threshold_stuff:
                    continue
                _mask = masks_all[i] > 0.5
                mask_area = _mask.sum().item()
                intersect = _mask & (results[0] > 0)
                intersect_area = intersect.sum().item()
                if labels_all[i] < self.num_things_classes:
                    if mask_area == 0 or (intersect_area * 1.0 / mask_area) > self.overlap_threshold_things:
                        continue
                else:
                    if mask_area == 0 or (intersect_area * 1.0 / mask_area) > self.overlap_threshold_stuff:
                        continue
                if intersect_area > 0:
                    _mask = _mask & (results[0] == 0)
                results[0, _mask] = labels_all[i]
                if labels_all[i] < self.num_things_classes:
                    lane[labels_all[i], _mask] = 1
                    lane_score[labels_all[i], _mask] = masks_all[i][_mask]
                    results[1, _mask] = id_unique
                    id_unique += 1

            bbox_list.append(bbox_th)
            labels_list.append(labels_th)
            seg_list.append(seg_th)
            lane_list.append(lane)
            lane_score_list.append(lane_score)
        results = []
        for i in range(len(img_metas)):
            results.append({
                'bbox': bbox_list[i],
                'segm': seg_list[i],
                'labels': labels_list[i],
                'drivable': drivable_list[i],
                'score_list': score_list[i],
                'lane': lane_list[i],
                'lane_score': lane_score_list[i],
                'stuff_score_list': stuff_score_list[i],
            })
        return results
