"""UniAD（base_e2e）推理模型：UniADTrack + UniAD 的合并移植。

搬自 UniAD：
  uniad/detectors/uniad_track.py   (UniADTrack：track 部分：_generate_empty_tracks / velo_update / get_bevs /
                                    _forward_single_frame_inference / simple_test_track / _track_instances2results)
  uniad/detectors/uniad_e2e.py     (UniAD.forward_test：把 track/seg/motion/occ/planning 串起来)
只移植推理；训练 / loss / criterion 不移植（但 ckpt 里的 `criterion.code_weights` 需要一个占位参数才能 strict 加载）。

与官方的**刻意差异**（均在 PORTING_NOTES.md 里逐条记录）：
  1. 官方把「流式状态」散落在 self.prev_frame_info / self.prev_bev / self.test_track_instances / self.timestamp / self.l2g_* /
     self.track_base.max_obj_id 等属性上；这里集中到 `StreamState`，由 runner 的 get_state/set_state 深拷贝/恢复。
  2. 把 backbone+neck（extract_feats）与「从 FPN 特征开始」（forward_from_feats）拆开，便于注入 FPN 特征。
  3. GridMask 在 eval 下恒为恒等映射（官方 `not self.training` 直接 return x），这里直接省略。
"""
import copy
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..utils import Instances, inverse_sigmoid
from ..compat import COMPAT
from ..heads.track_plugin import (MemoryBank, QueryInteractionModule, RuntimeTrackerBase, DETRTrack3DCoder)


def _box_cls():
    from mmdet3d.structures import LiDARInstance3DBoxes
    return LiDARInstance3DBoxes


class _Criterion(nn.Module):
    """占位：ckpt 里有 `criterion.code_weights`（ClipMatcher 的 code_weights，训练用），不参与推理。"""

    def __init__(self, code_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2)):
        super().__init__()
        self.code_weights = nn.Parameter(torch.tensor(list(code_weights)), requires_grad=False)


@dataclass
class StreamState:
    """UniAD 推理的全部流式状态（跨帧）。"""
    scene_token: Optional[str] = None            # 上一帧的 scene_token（用于检测场景切换）
    prev_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))   # 上一帧 can_bus[:3]（global 位置）
    prev_angle: float = 0.0                      # 上一帧 can_bus[-1]（yaw, 度）
    prev_bev: Optional[torch.Tensor] = None      # [bev_h*bev_w, 1, C]，上一帧 bev_embed
    track_instances: Optional[Instances] = None  # 上一帧输出的 track queries（含 memory bank）
    timestamp: Optional[float] = None
    l2g_r_mat: Optional[torch.Tensor] = None     # [1,3,3]
    l2g_t: Optional[torch.Tensor] = None         # [1,3]
    max_obj_id: int = 0                          # RuntimeTrackerBase.max_obj_id
    n_frames: int = 0

    def clone(self):
        return StreamState(
            scene_token=self.scene_token,
            prev_pos=np.array(self.prev_pos, copy=True),
            prev_angle=float(self.prev_angle),
            prev_bev=None if self.prev_bev is None else self.prev_bev.clone(),
            track_instances=None if self.track_instances is None else self.track_instances.clone(),
            timestamp=self.timestamp,
            l2g_r_mat=None if self.l2g_r_mat is None else self.l2g_r_mat.clone(),
            l2g_t=None if self.l2g_t is None else self.l2g_t.clone(),
            max_obj_id=self.max_obj_id,
            n_frames=self.n_frames,
        )


class _Timer:
    """可选的分段计时（cuda 同步），结果累计在 timing dict 里。"""

    def __init__(self, store, enabled, device):
        self.store, self.enabled, self.device = store, enabled, device

    def __call__(self, name):
        return _TimerCtx(self, name)


class _TimerCtx:
    def __init__(self, timer, name):
        self.t, self.name = timer, name

    def __enter__(self):
        if self.t.enabled:
            if self.t.device.type == 'cuda':
                torch.cuda.synchronize(self.t.device)
            self.t0 = time.perf_counter()

    def __exit__(self, *a):
        if self.t.enabled:
            if self.t.device.type == 'cuda':
                torch.cuda.synchronize(self.t.device)
            self.t.store.setdefault(self.name, []).append(time.perf_counter() - self.t0)


class UniADModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.timing = {}
        self.profile = True
        from mmengine.config import ConfigDict
        from mmdet.registry import MODELS as MMDET_MODELS
        from ..heads.track_head import BEVFormerTrackHead
        from ..heads.seg_head import PansegformerHead
        from ..heads.motion_head import MotionHead
        from ..heads.occ_head import OccHead
        from ..heads.planning_head import PlanningHeadSingleMode
        from ..utils import normalize_bbox  # noqa: F401

        cfg = copy.deepcopy(cfg)
        self.embed_dims = 256
        self.num_query = cfg['num_query']
        self.num_classes = cfg['num_classes']
        self.pc_range = cfg['pc_range']
        self.vehicle_id_list = cfg['vehicle_id_list']

        # --- backbone / neck：用 mmdet 3.3 的 ResNet / FPN（与 2.14 结构/参数命名一致，已逐 key 核对）---
        self.img_backbone = MMDET_MODELS.build(cfg['img_backbone'])
        self.img_neck = MMDET_MODELS.build(cfg['img_neck'])

        # --- head ---
        self.pts_bbox_head = BEVFormerTrackHead(**cfg['pts_bbox_head'])
        self.seg_head = PansegformerHead(**cfg['seg_head'])
        self.motion_head = MotionHead(**cfg['motion_head'])
        self.occ_head = OccHead(**self._cfgd(cfg['occ_head']))
        ph = cfg['planning_head']
        self.planning_head = PlanningHeadSingleMode(**{k: v for k, v in ph.items() if k != 'type'})

        self.bev_h, self.bev_w = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
        self.query_embedding = nn.Embedding(self.num_query + 1, self.embed_dims * 2)  # last one = ego query
        self.reference_points = nn.Linear(self.embed_dims, 3)

        self.track_base = RuntimeTrackerBase(
            score_thresh=cfg['score_thresh'], filter_score_thresh=cfg['filter_score_thresh'], miss_tolerance=5)
        self.query_interact = QueryInteractionModule(
            cfg['qim_args'], dim_in=self.embed_dims, hidden_dim=self.embed_dims, dim_out=self.embed_dims)
        self.memory_bank = MemoryBank(cfg['mem_args'], dim_in=self.embed_dims, hidden_dim=self.embed_dims,
                                      dim_out=self.embed_dims)
        self.mem_bank_len = self.memory_bank.max_his_length
        self.bbox_coder = DETRTrack3DCoder(
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], max_num=300, num_classes=10,
            score_threshold=0.0, with_nms=False, iou_thres=0.3)
        self.criterion = _Criterion()
        self._T = _Timer(self.timing, True, torch.device('cpu'))

    @staticmethod
    def _cfgd(d):
        from mmengine.config import ConfigDict
        d = {k: v for k, v in d.items() if k != 'type'}
        return {k: (ConfigDict(v) if isinstance(v, dict) else v) for k, v in d.items()}

    # ------------------------------------------------------------------
    # 图像特征
    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_img_feat(self, img):
        """img: [B*N, 3, H, W]（B=1, N=6）。返回 4 层 FPN 特征 list，每层 [B*N, C, h, w]（未 reshape）。"""
        feats = self.img_backbone(img)
        if isinstance(feats, dict):
            feats = list(feats.values())
        feats = self.img_neck(feats)
        return list(feats)

    # ------------------------------------------------------------------
    # track 部分
    # ------------------------------------------------------------------
    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embedding.weight.shape  # (901, 512)
        device = self.query_embedding.weight.device
        query = self.query_embedding.weight
        track_instances.ref_pts = self.reference_points(query[..., : dim // 2])

        pred_boxes_init = torch.zeros((len(track_instances), 10), dtype=torch.float, device=device)
        track_instances.query = query

        track_instances.output_embedding = torch.zeros((num_queries, dim >> 1), device=device)
        track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((len(track_instances),), dtype=torch.long, device=device)
        track_instances.iou = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = pred_boxes_init
        track_instances.pred_logits = torch.zeros((len(track_instances), self.num_classes), dtype=torch.float,
                                                  device=device)

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros((len(track_instances), mem_bank_len, dim // 2), dtype=torch.float32,
                                               device=device)
        track_instances.mem_padding_mask = torch.ones((len(track_instances), mem_bank_len), dtype=torch.bool,
                                                      device=device)
        track_instances.save_period = torch.zeros((len(track_instances),), dtype=torch.float32, device=device)

        return track_instances.to(self.query_embedding.weight.device)

    def velo_update(self, ref_pts, velocity, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta):
        """ref_pts (num_query,3) inverse-sigmoid 空间；velocity (num_query,2) m/s（lidar 系）。逐字搬运。"""
        time_delta = time_delta.type(torch.float)
        num_query = ref_pts.size(0)
        velo_pad_ = velocity.new_zeros((num_query, 1))
        velo_pad = torch.cat((velocity, velo_pad_), dim=-1)

        reference_points = ref_pts.sigmoid().clone()
        pc_range = self.pc_range
        reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = reference_points + velo_pad * time_delta

        ref_pts = reference_points @ l2g_r1 + l2g_t1 - l2g_t2

        g2l_r = torch.linalg.inv(l2g_r2).type(torch.float)

        ref_pts = ref_pts @ g2l_r

        ref_pts[..., 0:1] = (ref_pts[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
        ref_pts[..., 1:2] = (ref_pts[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
        ref_pts[..., 2:3] = (ref_pts[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

        ref_pts = inverse_sigmoid(ref_pts)
        return ref_pts

    def get_bevs(self, img_feats, img_metas, prev_bev=None):
        bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
            mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)

        if bev_embed.shape[1] == self.bev_h * self.bev_w:
            bev_embed = bev_embed.permute(1, 0, 2)

        assert bev_embed.shape[0] == self.bev_h * self.bev_w
        return bev_embed, bev_pos

    def select_active_track_query(self, track_instances, active_index, img_metas, with_mask=True):
        result_dict = self._track_instances2results(track_instances[active_index], img_metas, with_mask=with_mask)
        result_dict["track_query_embeddings"] = track_instances.output_embedding[active_index][
            result_dict['bbox_index']][result_dict['mask']]
        result_dict["track_query_matched_idxes"] = track_instances.matched_gt_idxes[active_index][
            result_dict['bbox_index']][result_dict['mask']]
        return result_dict

    def select_sdc_track_query(self, sdc_instance, img_metas):
        out = dict()
        result_dict = self._track_instances2results(sdc_instance, img_metas, with_mask=False)
        out["sdc_boxes_3d"] = result_dict['boxes_3d']
        out["sdc_scores_3d"] = result_dict['scores_3d']
        out["sdc_track_scores"] = result_dict['track_scores']
        out["sdc_track_bbox_results"] = result_dict['track_bbox_results']
        out["sdc_embedding"] = sdc_instance.output_embedding[0]
        return out

    def _track_instances2results(self, track_instances, img_metas, with_mask=True):
        bbox_dict = dict(
            cls_scores=track_instances.pred_logits,
            bbox_preds=track_instances.pred_boxes,
            track_scores=track_instances.scores,
            obj_idxes=track_instances.obj_idxes,
        )
        bboxes_dict = self.bbox_coder.decode(bbox_dict, with_mask=with_mask, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        BoxCls = _box_cls()
        bboxes = BoxCls(bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]
        bbox_index = bboxes_dict["bbox_index"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = dict(
            boxes_3d=bboxes.to("cpu"),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            track_scores=track_scores.cpu(),
            bbox_index=bbox_index.cpu(),
            track_ids=obj_idxes.cpu(),
            mask=bboxes_dict["mask"].cpu(),
            track_bbox_results=[[bboxes.to("cpu"), scores.cpu(), labels.cpu(), bbox_index.cpu(),
                                 bboxes_dict["mask"].cpu()]],
        )
        return result_dict

    def bev_from_override(self, bev_override):
        """[UniAD-port] BEV 层注入：外部给定的 bev_embed（[bev_h*bev_w,1,C]，与 get_bevs 输出同布局，
        flatten 下标 = row*bev_w + col，row↔LiDAR y、col↔LiDAR x）+ 与 get_bev_features 相同的 bev_pos。
        clone 一份，避免流式状态与调用方共享张量。"""
        bev = bev_override.to(device=self.query_embedding.weight.device, dtype=torch.float32)
        if bev.dim() == 2:
            bev = bev[:, None]
        assert tuple(bev.shape) == (self.bev_h * self.bev_w, 1, self.embed_dims), tuple(bev.shape)
        head = self.pts_bbox_head
        bev_mask = torch.zeros((1, self.bev_h, self.bev_w), device=bev.device).to(bev.dtype)
        bev_pos = head.positional_encoding(bev_mask).to(bev.dtype)
        return bev.clone(), bev_pos

    def _forward_single_frame_inference(self, img_feats, img_metas, track_instances, prev_bev=None,
                                        l2g_r1=None, l2g_t1=None, l2g_r2=None, l2g_t2=None, time_delta=None,
                                        bev_override=None):
        """从 FPN 特征 img_feats(list of [1,6,C,h,w]) 开始的单帧 track 前向。
        bev_override 非 None 时跳过 BEV encoder（img_feats/prev_bev 不再使用），直接用给定 BEV 跑 track 及后续 head。"""
        # velo update
        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]

        if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
            ref_pts = active_inst.ref_pts
            velo = active_inst.pred_boxes[:, -2:]
            ref_pts = self.velo_update(ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta=time_delta)
            ref_pts = ref_pts.squeeze(0)
            dim = active_inst.query.shape[-1]
            active_inst.ref_pts = self.reference_points(active_inst.query[..., :dim // 2])
            active_inst.ref_pts[..., :2] = ref_pts[..., :2]

        track_instances = Instances.cat([other_inst, active_inst])

        with self._T('bev_encoder'):
            if bev_override is None:
                bev_embed, bev_pos = self.get_bevs(img_feats, img_metas, prev_bev=prev_bev)
            else:
                bev_embed, bev_pos = self.bev_from_override(bev_override)
        with self._T('track_decoder'):
            det_output = self.pts_bbox_head.get_detections(
                bev_embed,
                object_query_embeds=track_instances.query,
                ref_points=track_instances.ref_pts,
                img_metas=img_metas,
            )
        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes,
            "pred_boxes": output_coords,
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "query_embeddings": query_feats,
            "all_past_traj_preds": det_output["all_past_traj_preds"],
            "bev_pos": bev_pos,
        }

        # update track instances with predict results
        track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values
        track_instances.scores = track_scores
        track_instances.pred_logits = output_classes[-1, 0]  # [300, num_cls]
        track_instances.pred_boxes = output_coords[-1, 0]  # [300, box_dim]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        track_instances.ref_pts = last_ref_pts[0]
        # hard_code: assume the 901 query is sdc query
        track_instances.obj_idxes[900] = -2
        self.track_base.update(track_instances, None)

        active_index = (track_instances.obj_idxes >= 0) & (track_instances.scores >= self.track_base.filter_score_thresh)
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))
        out.update(self.select_sdc_track_query(track_instances[track_instances.obj_idxes == -2], img_metas))

        # update with memory_bank
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        # Update track instances using matcher
        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances_fordet"] = track_instances
        out["track_instances"] = out_track_instances
        out["track_obj_idxes"] = track_instances.obj_idxes
        return out

    # ------------------------------------------------------------------
    # 整帧前向：从 FPN 特征开始
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_from_feats(self, mlvl_feats, img_metas, state: StreamState, command, l2g_t, l2g_r_mat, timestamp,
                           scene_start_canbus='zero', bev_override=None):
        """
        mlvl_feats : list of 4 tensors [1, 6, C, h, w]（FPN 输出，已 reshape 成 B,N）
        img_metas  : 单帧 dict（会被浅拷贝，can_bus 会 copy 后再改）
        command    : LongTensor [1]
        l2g_t      : [1,3] float32 tensor ； l2g_r_mat：[1,3,3] float32 tensor ； timestamp：python float（秒）
        scene_start_canbus : 'zero'（场景首帧 can_bus 位置/角度差置 0，BEVFormer 的正确做法，默认）
                             | 'official'（复刻 UniAD 官方 forward_test：用上一帧(可能来自上个场景)的差，首个场景用绝对值）
        bev_override : None（默认，走 BEV encoder）| [bev_h*bev_w,1,C] 外部 BEV（BEV 层注入，mlvl_feats 可为 None；
                       can_bus/track 状态照常推进，state.prev_bev 记为注入的 BEV）
        返回 dict：result（含 track/seg/motion/occ/planning 的原始输出，供 runner 转成规范格式）。
        """
        img_metas = dict(img_metas)
        can_bus = np.array(img_metas['can_bus'], dtype=np.float64, copy=True)
        if COMPAT.legacy_canbus_quat:
            # v1.0.1 权重训练时 can_bus[3:7] 是 [w,w,w,w]（Quaternion 对象被 numpy 当标量广播），见 compat.py
            can_bus[3:7] = can_bus[3]
        img_metas['can_bus'] = can_bus

        new_scene = (state.scene_token is None) or (img_metas['scene_token'] != state.scene_token)
        if new_scene:
            # 场景切换：清空 track / prev_bev
            state.track_instances = None
            state.prev_bev = None
            state.timestamp = None
            state.l2g_r_mat = None
            state.l2g_t = None
            state.max_obj_id = 0
            state.n_frames = 0
        state.scene_token = img_metas['scene_token']

        # ---- can_bus 帧间差（官方 forward_test）----
        tmp_pos = can_bus[:3].copy()
        tmp_angle = float(can_bus[-1])
        if scene_start_canbus == 'zero' and new_scene:
            can_bus[:3] = 0
            can_bus[-1] = 0
        else:
            can_bus[:3] -= state.prev_pos
            can_bus[-1] -= state.prev_angle
        state.prev_pos = tmp_pos
        state.prev_angle = tmp_angle

        # ---- simple_test_track ----
        if state.track_instances is None:
            track_instances = self._generate_empty_tracks()
            time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2 = None, None, None, None, None
            prev_bev = None
            self.track_base.max_obj_id = 0
        else:
            track_instances = state.track_instances
            time_delta = torch.tensor([timestamp - state.timestamp], dtype=torch.float64,
                                      device=l2g_t.device)
            l2g_r1 = state.l2g_r_mat
            l2g_t1 = state.l2g_t
            l2g_r2 = l2g_r_mat
            l2g_t2 = l2g_t
            prev_bev = state.prev_bev
            self.track_base.max_obj_id = state.max_obj_id

        state.timestamp = timestamp
        state.l2g_t = l2g_t
        state.l2g_r_mat = l2g_r_mat

        frame_res = self._forward_single_frame_inference(
            mlvl_feats, [img_metas], track_instances, prev_bev, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta,
            bev_override=bev_override)

        state.prev_bev = frame_res["bev_embed"]
        state.track_instances = frame_res["track_instances"]
        state.max_obj_id = self.track_base.max_obj_id
        state.n_frames += 1
        track_instances_fordet = frame_res["track_instances_fordet"]

        result_track = dict()
        for k in ["bev_embed", "bev_pos", "track_query_embeddings", "track_bbox_results", "boxes_3d", "scores_3d",
                  "labels_3d", "track_scores", "track_ids",
                  "sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]:
            result_track[k] = frame_res[k]

        bev_embed = result_track["bev_embed"]
        self._T.device = bev_embed.device
        self._T.enabled = self.profile
        with self._T('seg_head'):
            result_seg = self.seg_head.forward_test(bev_embed, [img_metas], rescale=True)

        # NOTE: motion_head.forward_test 会原地修改 outs_track['track_bbox_results']（拼上 SDC 框、labels[-1]=0），
        #       而 boxes_3d/labels_3d/scores_3d 是独立对象（官方同样如此），所以这里先把需要输出的拷出来。
        out_boxes = result_track['boxes_3d'].tensor.clone()
        out_scores = result_track['scores_3d'].clone()
        out_labels = result_track['labels_3d'].clone()
        out_ids = result_track['track_ids'].clone()

        with self._T('motion_head'):
            result_motion, outs_motion = self.motion_head.forward_test(
                bev_embed, outs_track=result_track, outs_seg=result_seg[0])
        outs_motion['bev_pos'] = result_track['bev_pos']

        outs_occ = None
        if self.occ_head is not None:
            occ_no_query = outs_motion['track_query'].shape[1] == 0
            with self._T('occ_head'):
                outs_occ = self.occ_head.forward_test(bev_embed, outs_motion, no_query=occ_no_query)

        with self._T('planning_head'):
            result_planning = self.planning_head.forward_test(bev_embed, outs_motion, outs_occ, command)

        return dict(
            boxes=out_boxes, scores=out_scores, labels=out_labels, track_ids=out_ids,
            boxes_3d_det=None,
            seg=result_seg[0]['pts_bbox'],
            motion=result_motion[0],
            occ=outs_occ,
            planning=result_planning,
            sdc_boxes=result_track['sdc_boxes_3d'].tensor.clone() if result_track['sdc_boxes_3d'] is not None else None,
            n_tracks=int(len(out_scores)),
            bev_embed=bev_embed,
        )
