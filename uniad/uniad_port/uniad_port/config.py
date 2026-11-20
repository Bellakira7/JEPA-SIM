"""UniAD base_e2e 的模型配置（照抄官方 `projects/configs/stage2_e2e/base_e2e.py` 的 `model` 部分，仅保留推理需要的键）。

与官方 config 的差异（仅 API 迁移，不改网络结构/超参）：
  - registry type 名保持不变（ResNet/FPN 走 mmdet 3.3 的实现；其余走本包自带实现）；
  - `anchor_info_path`(pkl) 换成 `anchor_npz_path`(npz，内容相同)；
  - 去掉 loss/train_cfg/assigner 等训练项。
"""
import copy

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
patch_size = [102.4, 102.4]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
class_names = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
               "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
group_id_list = [[0, 1, 2, 3, 4], [6, 7], [8], [5, 9]]
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
_feed_dim_ = _ffn_dim_
canvas_size = (bev_h_, bev_w_)
predict_steps = 12
predict_modes = 6
fut_steps = 4
past_steps = 4
use_nonlinear_optimizer = True
planning_steps = 6
occflow_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
}

import os
_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
# k-means 运动 anchors：v1.0.1 权重对应 GitHub release v1.0 的 motion_anchor_infos_mode6.pkl；
# 2.0 权重对应 HF UniAD2.0_R101_nuScenes/data/motion_anchor_infos_mode6.pkl（= v1.0 版 anchors 绕 z 轴旋转 -90°：(x,y)->(y,-x)）。
# 两个 pkl 均为外部下载物，已用限制型 unpickler 读出并转存 npz（内容相同）。
ANCHOR_NPZ = {'v1.0.1': os.path.join(_ASSETS, 'motion_anchors_mode6_v1.0.1.npz'),
              'v2.0': os.path.join(_ASSETS, 'motion_anchors_mode6_v2.0.npz')}
DEFAULT_ANCHOR_NPZ = ANCHOR_NPZ['v2.0']


def get_model_cfg(use_col_optim=True, anchor_npz_path=DEFAULT_ANCHOR_NPZ):
    model = dict(
        type="UniAD",
        queue_length=3,
        use_grid_mask=True,
        video_test_mode=True,
        num_query=900,
        num_classes=10,
        vehicle_id_list=vehicle_id_list,
        pc_range=point_cloud_range,
        img_backbone=dict(
            type="ResNet", depth=101, num_stages=4, out_indices=(1, 2, 3), frozen_stages=4,
            norm_cfg=dict(type="BN2d", requires_grad=False), norm_eval=True, style="caffe",
            dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True)),
        img_neck=dict(
            type="FPN", in_channels=[512, 1024, 2048], out_channels=_dim_, start_level=0,
            add_extra_convs="on_output", num_outs=4, relu_before_extra_convs=True),
        score_thresh=0.4,
        filter_score_thresh=0.35,
        qim_args=dict(qim_type="QIMBase", merger_dropout=0, update_query_pos=True, fp_ratio=0.3, random_drop=0.1),
        mem_args=dict(memory_bank_type="MemoryBank", memory_bank_score_thresh=0.0, memory_bank_len=4),
        pts_bbox_head=dict(
            type="BEVFormerTrackHead", bev_h=bev_h_, bev_w=bev_w_, num_query=900, num_classes=10, in_channels=_dim_,
            with_box_refine=True, as_two_stage=False, past_steps=past_steps, fut_steps=fut_steps,
            transformer=dict(
                type="PerceptionTransformer", rotate_prev_bev=True, use_shift=True, use_can_bus=True,
                embed_dims=_dim_,
                encoder=dict(
                    type="BEVFormerEncoder", num_layers=6, pc_range=point_cloud_range, num_points_in_pillar=4,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type="BEVFormerLayer",
                        attn_cfgs=[
                            dict(type="TemporalSelfAttention", embed_dims=_dim_, num_levels=1),
                            dict(type="SpatialCrossAttention", pc_range=point_cloud_range,
                                 deformable_attention=dict(type="MSDeformableAttention3D", embed_dims=_dim_,
                                                           num_points=8, num_levels=_num_levels_),
                                 embed_dims=_dim_)],
                        feedforward_channels=_ffn_dim_, ffn_dropout=0.1,
                        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"))),
                decoder=dict(
                    type="DetectionTransformerDecoder", num_layers=6, return_intermediate=True,
                    transformerlayers=dict(
                        type="DetrTransformerDecoderLayer",
                        attn_cfgs=[
                            dict(type="MultiheadAttention", embed_dims=_dim_, num_heads=8, dropout=0.1),
                            dict(type="CustomMSDeformableAttention", embed_dims=_dim_, num_levels=1)],
                        feedforward_channels=_ffn_dim_, ffn_dropout=0.1,
                        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm")))),
            bbox_coder=dict(type="NMSFreeCoder", post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                            pc_range=point_cloud_range, max_num=300, voxel_size=voxel_size, num_classes=10),
            positional_encoding=dict(type="LearnedPositionalEncoding", num_feats=_pos_dim_,
                                     row_num_embed=bev_h_, col_num_embed=bev_w_)),
        seg_head=dict(
            type='PansegformerHead', bev_h=bev_h_, bev_w=bev_w_, canvas_size=canvas_size, pc_range=point_cloud_range,
            num_query=300, num_classes=4, num_things_classes=3, num_stuff_classes=1, in_channels=2048,
            as_two_stage=False, with_box_refine=True,
            transformer=dict(
                type='SegDeformableTransformer',
                encoder=dict(
                    type='DetrTransformerEncoder', num_layers=6,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(type='MultiScaleDeformableAttention', embed_dims=_dim_, num_levels=_num_levels_),
                        feedforward_channels=_feed_dim_, ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
                decoder=dict(
                    type='DeformableDetrTransformerDecoder', num_layers=6, return_intermediate=True,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(type='MultiheadAttention', embed_dims=_dim_, num_heads=8, dropout=0.1),
                            dict(type='MultiScaleDeformableAttention', embed_dims=_dim_, num_levels=_num_levels_)],
                        feedforward_channels=_feed_dim_, ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')))),
            positional_encoding=dict(type='SinePositionalEncoding', num_feats=_dim_ // 2, normalize=True, offset=-0.5),
            thing_transformer_head=dict(type='SegMaskHead', d_model=_dim_, nhead=8, num_decoder_layers=4),
            stuff_transformer_head=dict(type='SegMaskHead', d_model=_dim_, nhead=8, num_decoder_layers=6,
                                        self_attn=True)),
        occ_head=dict(
            type='OccHead', grid_conf=occflow_grid_conf, ignore_index=255, bev_proj_dim=256, bev_proj_nlayers=4,
            attn_mask_thresh=0.3,
            transformer_decoder=dict(
                type='DetrTransformerDecoder', return_intermediate=True, num_layers=5,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=dict(type='MultiheadAttention', embed_dims=256, num_heads=8, attn_drop=0.0,
                                   proj_drop=0.0, dropout_layer=None, batch_first=False),
                    ffn_cfgs=dict(embed_dims=256, feedforward_channels=2048, num_fcs=2,
                                  act_cfg=dict(type='ReLU', inplace=True), ffn_drop=0.0, dropout_layer=None,
                                  add_identity=True),
                    feedforward_channels=2048,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            query_dim=256, query_mlp_layers=3, aux_loss_weight=1.,
            pan_eval=True, test_seg_thresh=0.1, test_with_track_score=True),
        motion_head=dict(
            type='MotionHead', bev_h=bev_h_, bev_w=bev_w_, num_query=300, num_classes=10,
            predict_steps=predict_steps, predict_modes=predict_modes, embed_dims=_dim_, num_cls_fcs=3,
            pc_range=point_cloud_range, group_id_list=group_id_list, num_anchor=6,
            use_nonlinear_optimizer=use_nonlinear_optimizer, anchor_npz_path=anchor_npz_path,
            vehicle_id_list=vehicle_id_list,
            transformerlayers=dict(
                type='MotionTransformerDecoder', pc_range=point_cloud_range, embed_dims=_dim_, num_layers=3,
                transformerlayers=dict(
                    type='MotionTransformerAttentionLayer', batch_first=True,
                    attn_cfgs=[dict(type='MotionDeformableAttention', num_steps=predict_steps, embed_dims=_dim_,
                                    num_levels=1, num_heads=8, num_points=4, sample_index=-1)],
                    feedforward_channels=_ffn_dim_, ffn_dropout=0.1,
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm')))),
        planning_head=dict(
            type='PlanningHeadSingleMode', embed_dims=256, planning_steps=planning_steps,
            use_col_optim=use_col_optim, planning_eval=True, with_adapter=True),
    )
    return copy.deepcopy(model)
