"""Track 相关插件：MemoryBank / QueryInteractionModule / RuntimeTrackerBase / DETRTrack3DCoder。

搬自 UniAD：
  uniad/dense_heads/track_head_plugin/{modules.py,tracker.py}、core/bbox/coders/detr3d_track_coder.py
改动：
  - RuntimeTrackerBase.update 由逐元素 Python 循环（每个元素多次 GPU 同步）改为**等价向量化**实现（tests/test_tracker.py
    与官方循环版随机对拍逐位一致）；`iou_thre` 分支（官方推理里恒为 None）不移植。
  - MemoryBank/QueryInteractionModule 逐字搬运（去掉 training 分支不影响，因为推理恒 eval）。
  - DETRTrack3DCoder：去掉 with_nms 分支（config 里 with_nms=False）；post_center_range 不再原地改成 tensor。
"""
import torch
import torch.nn.functional as F
from torch import nn

from ..utils import Instances, denormalize_bbox


# ----------------------------------------------------------------------------
# MemoryBank
# ----------------------------------------------------------------------------
class MemoryBank(nn.Module):

    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        self.save_thresh = args['memory_bank_score_thresh']
        self.save_period = 3
        self.max_his_length = args['memory_bank_len']

        self.save_proj = nn.Linear(dim_in, dim_in)

        self.temporal_attn = nn.MultiheadAttention(dim_in, 8, dropout=0)
        self.temporal_fc1 = nn.Linear(dim_in, hidden_dim)
        self.temporal_fc2 = nn.Linear(hidden_dim, dim_in)
        self.temporal_norm1 = nn.LayerNorm(dim_in)
        self.temporal_norm2 = nn.LayerNorm(dim_in)

    def update(self, track_instances):
        embed = track_instances.output_embedding[:, None]  # (N, 1, 256)
        scores = track_instances.scores
        mem_padding_mask = track_instances.mem_padding_mask
        device = embed.device

        save_period = track_instances.save_period
        if self.training:
            saved_idxes = scores > 0
        else:
            saved_idxes = (save_period == 0) & (scores > self.save_thresh)
            save_period[save_period > 0] -= 1
            save_period[saved_idxes] = self.save_period

        saved_embed = embed[saved_idxes]
        if len(saved_embed) > 0:
            prev_embed = track_instances.mem_bank[saved_idxes]
            save_embed = self.save_proj(saved_embed)
            mem_padding_mask[saved_idxes] = torch.cat(
                [mem_padding_mask[saved_idxes, 1:],
                 torch.zeros((len(saved_embed), 1), dtype=torch.bool, device=device)], dim=1)
            track_instances.mem_bank = track_instances.mem_bank.clone()
            track_instances.mem_bank[saved_idxes] = torch.cat([prev_embed[:, 1:], save_embed], dim=1)

    def _forward_temporal_attn(self, track_instances):
        if len(track_instances) == 0:
            return track_instances

        key_padding_mask = track_instances.mem_padding_mask  # [n_, memory_bank_len]

        valid_idxes = key_padding_mask[:, -1] == 0
        embed = track_instances.output_embedding[valid_idxes]  # (n, 256)

        if len(embed) > 0:
            prev_embed = track_instances.mem_bank[valid_idxes]
            key_padding_mask = key_padding_mask[valid_idxes]
            embed2 = self.temporal_attn(
                embed[None],  # (num_track, dim) to (1, num_track, dim)
                prev_embed.transpose(0, 1),  # (num_track, mem_len, dim) to (mem_len, num_track, dim)
                prev_embed.transpose(0, 1),
                key_padding_mask=key_padding_mask,
            )[0][0]

            embed = self.temporal_norm1(embed + embed2)
            embed2 = self.temporal_fc2(F.relu(self.temporal_fc1(embed)))
            embed = self.temporal_norm2(embed + embed2)
            track_instances.output_embedding = track_instances.output_embedding.clone()
            track_instances.output_embedding[valid_idxes] = embed

        return track_instances

    def forward_temporal_attn(self, track_instances):
        return self._forward_temporal_attn(track_instances)

    def forward(self, track_instances: Instances, update_bank=True) -> Instances:
        track_instances = self._forward_temporal_attn(track_instances)
        if update_bank:
            self.update(track_instances)
        return track_instances


# ----------------------------------------------------------------------------
# QIM
# ----------------------------------------------------------------------------
class QueryInteractionBase(nn.Module):

    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.args = args
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        self._reset_parameters()

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        raise NotImplementedError()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _select_active_tracks(self, data: dict) -> Instances:
        raise NotImplementedError()

    def _update_track_embedding(self, track_instances):
        raise NotImplementedError()


class QueryInteractionModule(QueryInteractionBase):

    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args["random_drop"]
        self.fp_ratio = args["fp_ratio"]
        self.update_query_pos = args["update_query_pos"]

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args["merger_dropout"]

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args["update_query_pos"]:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
            return track_instances
        dim = track_instances.query.shape[1]
        out_embed = track_instances.output_embedding
        query_pos = track_instances.query[:, :dim // 2]
        query_feat = track_instances.query[:, dim // 2:]
        q = k = query_pos + out_embed

        # attention
        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query[:, :dim // 2] = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query[:, dim // 2:] = query_feat
        return track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data["track_instances"]
        # 推理(eval)：active = obj_idxes >= 0（训练分支的 random_drop / fp 注入不移植）
        active_track_instances = track_instances[track_instances.obj_idxes >= 0]
        return active_track_instances

    def forward(self, data) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances)
        init_track_instances: Instances = data["init_track_instances"]
        merged_track_instances = Instances.cat([init_track_instances, active_track_instances])
        return merged_track_instances


# ----------------------------------------------------------------------------
# RuntimeTrackerBase（向量化等价实现）
# ----------------------------------------------------------------------------
class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.5, filter_score_thresh=0.4, miss_tolerance=5):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    @torch.no_grad()
    def update(self, track_instances: Instances, iou_thre=None):
        """与官方逐元素循环版逐位等价：
           1) scores>=score_thresh 的 disappear_time 清零；
           2) obj_idx==-1 且 score>=score_thresh -> 按下标顺序分配新 id（max_obj_id 递增）；
           3) obj_idx>=0 且 score<filter_thresh -> disappear_time+=1，>=miss_tolerance 时 obj_idx=-1。
           (2)(3) 的条件互斥且逐元素独立，因此可一次性向量化。"""
        assert iou_thre is None
        scores = track_instances.scores
        obj = track_instances.obj_idxes
        track_instances.disappear_time[scores >= self.score_thresh] = 0

        new_mask = (obj == -1) & (scores >= self.score_thresh)
        dec_mask = (obj >= 0) & (scores < self.filter_score_thresh)

        n_new = int(new_mask.sum().item())
        if n_new > 0:
            ids = torch.arange(self.max_obj_id, self.max_obj_id + n_new, dtype=obj.dtype, device=obj.device)
            obj[new_mask] = ids
            self.max_obj_id += n_new
        if bool(dec_mask.any()):
            track_instances.disappear_time[dec_mask] += 1
            kill = dec_mask & (track_instances.disappear_time >= self.miss_tolerance)
            obj[kill] = -1


class RuntimeTrackerBaseRef(object):
    """官方逐元素循环版（仅供 tests/test_tracker.py 对拍；推理不使用）。"""

    def __init__(self, score_thresh=0.5, filter_score_thresh=0.4, miss_tolerance=5):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def update(self, track_instances: Instances, iou_thre=None):
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if (track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh):
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif (track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh):
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    track_instances.obj_idxes[i] = -1


# ----------------------------------------------------------------------------
# DETRTrack3DCoder
# ----------------------------------------------------------------------------
class DETRTrack3DCoder(object):
    """Bbox coder for DETR3D tracking（UniADTrack 默认 coder；config 里 with_nms=False、score_threshold=0.0）。"""

    def __init__(self, pc_range, post_center_range=None, max_num=100, score_threshold=0.2,
                 num_classes=7, with_nms=False, iou_thres=0.3, **kwargs):
        assert not with_nms, 'with_nms 分支未移植（官方 config 为 False）'
        self.pc_range = pc_range
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def decode_single(self, cls_scores, bbox_preds, track_scores, obj_idxes, with_mask=True, img_metas=None):
        max_num = min(cls_scores.size(0), self.max_num)

        cls_scores = cls_scores.sigmoid()
        _, indexs = cls_scores.max(dim=-1)
        labels = indexs % self.num_classes

        _, bbox_index = track_scores.topk(max_num)

        labels = labels[bbox_index]
        bbox_preds = bbox_preds[bbox_index]
        track_scores = track_scores[bbox_index]
        obj_idxes = obj_idxes[bbox_index]

        scores = track_scores

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_scores = track_scores
        final_preds = labels

        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            post_center_range = torch.tensor(self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask
            if not with_mask:
                mask = torch.ones_like(mask) > 0
            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            track_scores = track_scores[mask]
            obj_idxes = obj_idxes[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
                'track_scores': track_scores,
                'obj_idxes': obj_idxes,
                'bbox_index': bbox_index,
                'mask': mask,
            }
        else:
            raise NotImplementedError('Need to reorganize output as a batch, only '
                                      'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts, with_mask=True, img_metas=None):
        all_cls_scores = preds_dicts['cls_scores']
        all_bbox_preds = preds_dicts['bbox_preds']
        track_scores = preds_dicts['track_scores']
        obj_idxes = preds_dicts['obj_idxes']

        predictions_list = []
        # bs size = 1
        predictions_list.append(self.decode_single(
            all_cls_scores, all_bbox_preds, track_scores, obj_idxes, with_mask, img_metas))
        return predictions_list
