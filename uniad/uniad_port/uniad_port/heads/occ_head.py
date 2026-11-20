"""OccHead（占用/实例预测 head，推理部分）。

搬自 UniAD `uniad/dense_heads/occ_head.py`。改动：删训练相关（forward_train / get_occ_labels / loss）；
forward_test 去掉对 GT（gt_segmentation/gt_instance）的依赖，只产出预测（seg_out / ins_seg_out）；
mmcv 1.x API 迁移。planning head 的 collision optimization 只用 `seg_out`。
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmengine.model import BaseModule
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from ..modules import detr_layers  # noqa: F401  (registry side-effect: DetrTransformerDecoder(+Layer))
from .occ_plugin import (MLP, BevFeatureSlicer, SimpleConv2d, CVT_Decoder, Bottleneck, UpsamplingAdd,
                         predict_instance_segmentation_and_trajectories)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class OccHead(BaseModule):
    def __init__(self, 
                 # General
                 receptive_field=3,
                 n_future=4,
                 spatial_extent=(50, 50),
                 ignore_index=255,

                 # BEV
                 grid_conf = None,

                 bev_size=(200, 200),
                 bev_emb_dim=256,
                 bev_proj_dim=64,
                 bev_proj_nlayers=1,

                 # Query
                 query_dim=256,
                 query_mlp_layers=3,
                 detach_query_pos=True,
                 temporal_mlp_layer=2,

                 # Transformer
                 transformer_decoder=None,

                 attn_mask_thresh=0.5,
                 # Loss
                 sample_ignore_mode='all_valid',
                 aux_loss_weight=1.,

                 loss_mask=None,
                 loss_dice=None,

                 # Cfgs
                 init_cfg=None,

                 # Eval
                 pan_eval=False,
                 test_seg_thresh:float=0.5,
                 test_with_track_score=False,
                 ):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg)
        self.receptive_field = receptive_field  # NOTE: Used by prepare_future_labels in E2EPredTransformer
        self.n_future = n_future
        self.spatial_extent = spatial_extent
        self.ignore_index  = ignore_index

        bevformer_bev_conf = {
            'xbound': [-51.2, 51.2, 0.512],
            'ybound': [-51.2, 51.2, 0.512],
            'zbound': [-10.0, 10.0, 20.0],
        }
        self.bev_sampler =  BevFeatureSlicer(bevformer_bev_conf, grid_conf)
        
        self.bev_size = bev_size
        self.bev_proj_dim = bev_proj_dim

        if bev_proj_nlayers == 0:
            self.bev_light_proj = nn.Sequential()
        else:
            self.bev_light_proj = SimpleConv2d(
                in_channels=bev_emb_dim,
                conv_channels=bev_emb_dim,
                out_channels=bev_proj_dim,
                num_conv=bev_proj_nlayers,
            )

        # Downscale bev_feat -> /4
        self.base_downscale = nn.Sequential(
            Bottleneck(in_channels=bev_proj_dim, downsample=True),
            Bottleneck(in_channels=bev_proj_dim, downsample=True)
        )

        # Future blocks with transformer
        self.n_future_blocks = self.n_future + 1

        # - transformer
        self.attn_mask_thresh = attn_mask_thresh
        
        self.num_trans_layers = transformer_decoder.num_layers
        assert self.num_trans_layers % self.n_future_blocks == 0

        self.num_heads = transformer_decoder.transformerlayers.\
            attn_cfgs.num_heads
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)

        # - temporal-mlps
        # query_out_dim = bev_proj_dim

        temporal_mlp = MLP(query_dim, query_dim, bev_proj_dim, num_layers=temporal_mlp_layer)
        self.temporal_mlps = _get_clones(temporal_mlp, self.n_future_blocks)
            
        # - downscale-convs
        downscale_conv = Bottleneck(in_channels=bev_proj_dim, downsample=True)
        self.downscale_convs = _get_clones(downscale_conv, self.n_future_blocks)
        
        # - upsampleAdds
        upsample_add = UpsamplingAdd(in_channels=bev_proj_dim, out_channels=bev_proj_dim)
        self.upsample_adds = _get_clones(upsample_add, self.n_future_blocks)

        # Decoder
        self.dense_decoder = CVT_Decoder(
            dim=bev_proj_dim,
            blocks=[bev_proj_dim, bev_proj_dim],
        )

        # Query
        self.mode_fuser = nn.Sequential(
                nn.Linear(query_dim, bev_proj_dim),
                nn.LayerNorm(bev_proj_dim),
                nn.ReLU(inplace=True)
            )
        self.multi_query_fuser =  nn.Sequential(
                nn.Linear(query_dim * 3, query_dim * 2),
                nn.LayerNorm(query_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(query_dim * 2, bev_proj_dim),
            )

        self.detach_query_pos = detach_query_pos

        self.query_to_occ_feat = MLP(
            query_dim, query_dim, bev_proj_dim, num_layers=query_mlp_layers
        )
        self.temporal_mlp_for_mask = copy.deepcopy(self.query_to_occ_feat)
        
        # Loss
        # For matching
        self.sample_ignore_mode = sample_ignore_mode
        assert self.sample_ignore_mode in ['all_valid', 'past_valid', 'none']

        self.aux_loss_weight = aux_loss_weight


        self.pan_eval = pan_eval
        self.test_seg_thresh  = test_seg_thresh

        self.test_with_track_score = test_with_track_score
        self.init_weights()

    def init_weights(self):
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def get_attn_mask(self, state, ins_query):
        # state: b, c, h, w
        # ins_query: b, q, c
        ins_embed = self.temporal_mlp_for_mask(
            ins_query 
        )
        mask_pred = torch.einsum("bqc,bchw->bqhw", ins_embed, state)
        attn_mask = mask_pred.sigmoid() < self.attn_mask_thresh
        attn_mask = rearrange(attn_mask, 'b q h w -> b (h w) q').unsqueeze(1).repeat(
            1, self.num_heads, 1, 1).flatten(0, 1)
        attn_mask = attn_mask.detach()
        
        # if a mask is all True(all background), then set it all False.
        attn_mask[torch.where(
            attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        upsampled_mask_pred = F.interpolate(
            mask_pred,
            self.bev_size,
            mode='bilinear',
            align_corners=False
        )  # Supervised by gt

        return attn_mask, upsampled_mask_pred, ins_embed

    def forward(self, x, ins_query):
        base_state = rearrange(x, '(h w) b d -> b d h w', h=self.bev_size[0])

        base_state = self.bev_sampler(base_state)
        base_state = self.bev_light_proj(base_state)
        base_state = self.base_downscale(base_state)
        base_ins_query = ins_query

        last_state = base_state
        last_ins_query = base_ins_query
        future_states = []
        mask_preds = []
        temporal_query = []
        temporal_embed_for_mask_attn = []
        n_trans_layer_each_block = self.num_trans_layers // self.n_future_blocks
        assert n_trans_layer_each_block >= 1
        
        for i in range(self.n_future_blocks):
            # Downscale
            cur_state = self.downscale_convs[i](last_state)  # /4 -> /8

            # Attention
            # temporal_aware ins_query
            cur_ins_query = self.temporal_mlps[i](last_ins_query)  # [b, q, d]
            temporal_query.append(cur_ins_query)

            # Generate attn mask 
            attn_mask, mask_pred, cur_ins_emb_for_mask_attn = self.get_attn_mask(cur_state, cur_ins_query)
            attn_masks = [None, attn_mask] 

            mask_preds.append(mask_pred)  # /1
            temporal_embed_for_mask_attn.append(cur_ins_emb_for_mask_attn)

            cur_state = rearrange(cur_state, 'b c h w -> (h w) b c')
            cur_ins_query = rearrange(cur_ins_query, 'b q c -> q b c')

            for j in range(n_trans_layer_each_block):
                trans_layer_ind = i * n_trans_layer_each_block + j
                trans_layer = self.transformer_decoder.layers[trans_layer_ind]
                cur_state = trans_layer(
                    query=cur_state,  # [h'*w', b, c]
                    key=cur_ins_query,  # [nq, b, c]
                    value=cur_ins_query,  # [nq, b, c]
                    query_pos=None,  
                    key_pos=None,
                    attn_masks=attn_masks,
                    query_key_padding_mask=None,
                    key_padding_mask=None
                )  # out size: [h'*w', b, c]

            cur_state = rearrange(cur_state, '(h w) b c -> b c h w', h=self.bev_size[0]//8)
            
            # Upscale to /4
            cur_state = self.upsample_adds[i](cur_state, last_state)

            # Out
            future_states.append(cur_state)  # [b, d, h/4, w/4]
            last_state = cur_state

        future_states = torch.stack(future_states, dim=1)  # [b, t, d, h/4, w/4]
        temporal_query = torch.stack(temporal_query, dim=1)  # [b, t, q, d]
        mask_preds = torch.stack(mask_preds, dim=2)  # [b, q, t, h, w]
        ins_query = torch.stack(temporal_embed_for_mask_attn, dim=1)  # [b, t, q, d]

        # Decode future states to larger resolution
        future_states = self.dense_decoder(future_states)
        ins_occ_query = self.query_to_occ_feat(ins_query)    # [b, t, q, query_out_dim]
        
        
        ins_occ_logits = torch.einsum("btqc,btchw->bqthw", ins_occ_query, future_states)
        
        return mask_preds, ins_occ_logits

    def merge_queries(self, outs_dict, detach_query_pos=True):
        ins_query = outs_dict.get('traj_query', None)       # [n_dec, b, nq, n_modes, dim]
        track_query = outs_dict['track_query']              # [b, nq, d]
        track_query_pos = outs_dict['track_query_pos']      # [b, nq, d]

        if detach_query_pos:
            track_query_pos = track_query_pos.detach()

        ins_query = ins_query[-1]
        ins_query = self.mode_fuser(ins_query).max(2)[0]
        ins_query = self.multi_query_fuser(torch.cat([ins_query, track_query, track_query_pos], dim=-1))
        
        return ins_query

    def forward_test(self, bev_feat, outs_dict, no_query=False):
        """推理：官方 forward_test 里与 GT 相关的 seg_gt/ins_seg_gt 不再生成。
        no_query 时官方输出 zeros_like(seg_gt)（[1,5,1,200,200] long），这里直接按同形状给零。"""
        out_dict = dict()
        if no_query:
            bs = bev_feat.shape[1]
            out_dict['seg_out'] = torch.zeros((bs, 1 + self.n_future, 1, *self.bev_size), dtype=torch.long,
                                              device=bev_feat.device)
            out_dict['ins_seg_out'] = torch.zeros((bs, 1 + self.n_future, *self.bev_size), dtype=torch.long,
                                                  device=bev_feat.device)
            return out_dict

        ins_query = self.merge_queries(outs_dict, self.detach_query_pos)

        _, pred_ins_logits = self(bev_feat, ins_query=ins_query)

        out_dict['pred_ins_logits'] = pred_ins_logits

        pred_ins_logits = pred_ins_logits[:, :, :1 + self.n_future]  # [b, q, t, h, w]
        pred_ins_sigmoid = pred_ins_logits.sigmoid()  # [b, q, t, h, w]

        if self.test_with_track_score:
            track_scores = outs_dict['track_scores'].to(pred_ins_sigmoid)  # [b, q]
            track_scores = track_scores[:, :, None, None, None]
            pred_ins_sigmoid = pred_ins_sigmoid * track_scores  # [b, q, t, h, w]

        out_dict['pred_ins_sigmoid'] = pred_ins_sigmoid
        pred_seg_scores = pred_ins_sigmoid.max(1)[0]
        seg_out = (pred_seg_scores > self.test_seg_thresh).long().unsqueeze(2)  # [b, t, 1, h, w]
        out_dict['seg_out'] = seg_out
        if self.pan_eval:
            pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(seg_out, pred_ins_sigmoid)
            out_dict['ins_seg_out'] = pred_consistent_instance_seg  # [1, 5, 200, 200]

        return out_dict
