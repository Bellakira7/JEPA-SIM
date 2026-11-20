"""UniADRunner：可注入 FPN 特征、可保存/恢复流式状态的 UniAD 推理器。

典型用法::

    runner = UniADRunner(ckpt='.../uniad_base_e2e.pth', device='cuda:0')
    data = UniADValData()
    for frame in data.iter_scene(0):                 # 一个场景
        out = runner.step_frame_images(frame)        # 真图 -> reference
    # 注入特征：
    runner.reset_scene()
    for frame in data.iter_scene(0):
        feats = runner.extract_feats(frame['img'])   # 4 层 FPN，list of [6,256,h,w]
        feats2 = my_world_model(feats)               # 被测来源的特征
        out = runner.step_frame(feats2, frame)
    # 锚帧处分叉：
    st = runner.get_state()                          # 深拷贝全部流式状态
    ... 分支 A ...; runner.set_state(st); ... 分支 B ...

输出为规范化 numpy 结构（见 `step` 的文档）。
"""
import copy
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import get_model_cfg, ANCHOR_NPZ
from .models.uniad import UniADModel, StreamState
from .compat import COMPAT

# ckpt 中存在、但官方模型代码里没有对应参数的 key（官方 mmcv load_checkpoint 非 strict，所以静默忽略）。
# 本移植显式列出并逐个说明，其余任何不匹配都会报错（strict）。
KNOWN_UNUSED_CKPT_KEYS = {
    'bbox_size_fc.weight': 'UniAD 早期版本 UniADTrack 里的 nn.Linear(256,3)（预测框尺寸基准），后续版本代码已删除，forward 里从未使用',
    'bbox_size_fc.bias': '同上',
    'pts_bbox_head.query_embedding.weight': '早期 BEVFormerTrackHead 里的 nn.Embedding(900,512)；当前 head 的 _init_layers 不再创建它'
                                            '（object query 用的是 UniADTrack.query_embedding），forward 里从未使用',
}


def load_uniad_ckpt(model: torch.nn.Module, ck, verbose=True, path=''):
    """strict 加载：只允许 KNOWN_UNUSED_CKPT_KEYS 里列出的 key 在 ckpt 多出，model 不得缺任何 key，shape 必须全部一致。
    ck 可以是路径或已加载的 ckpt dict。"""
    if isinstance(ck, str):
        path = ck
        ck = torch.load(ck, map_location='cpu', weights_only=False)
    sd = ck['state_dict'] if 'state_dict' in ck else ck
    unused = {k: v for k, v in sd.items() if k in KNOWN_UNUSED_CKPT_KEYS and k not in model.state_dict()}
    sd = {k: v for k, v in sd.items() if k not in unused}
    msg = model.load_state_dict(sd, strict=True)   # 有任何 missing/unexpected 都会抛错
    if verbose:
        print(f'[load_uniad_ckpt] {path}: loaded {len(sd)} tensors strict=True; '
              f'ignored {len(unused)} known-unused ckpt keys: {sorted(unused)}')
    return dict(meta={k: v for k, v in ck.get('meta', {}).items() if k not in ('config', 'env_info')},
                ignored=sorted(unused))


class UniADRunner:
    def __init__(self, ckpt: str, device: str = 'cuda:0', use_col_optim: bool = True,
                 scene_start_canbus: str = 'zero', convention: str = 'auto', dtype=torch.float32, verbose=True):
        """
        use_col_optim : planning head 是否用 occ head 的预测做碰撞优化（官方 base_e2e 为 True，依赖 casadi/ipopt）。
        scene_start_canbus : 'zero' | 'official'，见 UniADModel.forward_from_feats。
        convention : 'auto' | 'v1.0.1' | 'v2.0'。权重与代码版本的数值约定（见 compat.py）。auto = 按 ckpt meta 的
                     mmdet3d_version 判定（0.17.x -> v1.0.1，1.0.0rc6 -> v2.0）。
        """
        assert dtype == torch.float32, '目前只验证了 fp32'
        self.device = torch.device(device)
        self.scene_start_canbus = scene_start_canbus
        self.use_col_optim = use_col_optim
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        meta = ck.get('meta', {})
        if convention == 'auto':
            ver = str(meta.get('mmdet3d_version', ''))
            convention = 'v1.0.1' if ver.startswith('0.') else 'v2.0'
        COMPAT.set(convention)
        self.convention = convention
        if verbose:
            print(f'[UniADRunner] convention={convention} (ckpt mmdet3d_version={meta.get("mmdet3d_version")})')
        self.model = UniADModel(get_model_cfg(use_col_optim=use_col_optim, anchor_npz_path=ANCHOR_NPZ[convention]))
        info = load_uniad_ckpt(self.model, ck, verbose=verbose, path=ckpt)
        self.ckpt_info = info
        self.model.eval().to(self.device)
        self.state = StreamState()
        self.timing: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------ 状态
    def reset_scene(self):
        """清空全部流式状态（prev_bev / track queries / can_bus 基准 / 时间戳等）。"""
        self.state = StreamState()

    def get_state(self) -> StreamState:
        """深拷贝全部流式状态（张量 clone），用于在锚帧处分叉。"""
        return self.state.clone()

    def set_state(self, state: StreamState):
        """恢复流式状态（再 clone 一份，避免多个分支共享同一份张量）。"""
        self.state = state.clone()

    # ------------------------------------------------------------------ 时间统计
    def _tick(self, name, t0):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.timing.setdefault(name, []).append(time.perf_counter() - t0)

    # ------------------------------------------------------------------ 特征
    @torch.no_grad()
    def extract_feats(self, img) -> List[torch.Tensor]:
        """img: [6,3,H,W]（或 [1,6,3,H,W]）CPU/GPU 张量 -> 4 层 FPN 特征 list，每层 [6,256,h,w]（img_neck 输出）。"""
        if img.dim() == 5:
            assert img.shape[0] == 1
            img = img[0]
        img = img.to(self.device, non_blocking=True)
        t0 = time.perf_counter()
        feats = self.model.img_backbone(img)
        if isinstance(feats, dict):
            feats = list(feats.values())
        self._tick('backbone', t0)
        t0 = time.perf_counter()
        feats = list(self.model.img_neck(feats))
        self._tick('neck', t0)
        return feats

    # ------------------------------------------------------------------ 单步
    @torch.no_grad()
    def step(self, feats, img_metas: dict, command: int, l2g_t=None, l2g_r_mat=None, timestamp=None,
             return_raw=False, return_bev=False):
        """从给定 FPN 特征开始跑 BEV encoder + track/seg/motion/occ/planning，并推进流式状态。

        feats      : 4 层 FPN 特征 list，每层 [6,256,h,w] 或 [1,6,256,h,w]（须与 extract_feats 同形状/同分布）
        img_metas  : 单帧 dict（UniADValData.get_frame()['img_metas']），含 lidar2img / can_bus / scene_token /
                     img_shape / l2g_r_mat / sample_idx ...
        command    : 0=RIGHT 1=LEFT 2=FORWARD（官方来自 GT 未来自车轨迹）
        l2g_t / l2g_r_mat / timestamp：缺省时从 img_metas 取（UniADValData.get_frame 已放入）

        返回规范化输出（numpy）::
          boxes    float32 [N,9]  [x,y,z_bottom,l,w,h,yaw,vx,vy]，mmdet3d>=1.0 LiDAR 约定（l 沿朝向，yaw 为 LiDAR_TOP 系下绕 +z 逆时针、
                                    与 nuScenes GT 在 LiDAR 系下的 yaw 同义）；LiDAR_TOP 系（x 右 y 前）。v1.0.1 权重的原始输出
                                    (w,l,yaw_old) 已换算成该约定；原始列保留在 boxes_native。
          scores   float32 [N]    = track score（max sigmoid cls；官方 bbox_results 的 scores_3d，已含官方的 track 过滤 score>=0.35）
          labels   int64   [N]    0..9（nuScenes 10 类，顺序见 meta['class_names']）
          track_ids int64  [N]
          seg      dict(lane=float32[200,200] (divider∪crossing∪contour 的 0/1 二值并集), drivable=float32[200,200] 0/1,
                        divider/crossing/contour=0/1, lane_score/drivable_score=float(mask 概率,仅参考))
                   —— UniAD 的 seg 是二值栅格(0/1)，不是概率/logit；栅格约定见 meta['seg_grid']
          motion   dict(traj=float32[N,6,12,2]  相对各 agent 当前框中心的累计位移(LiDAR 轴)，已剥掉 SDC 行，与 boxes 对齐,
                        traj_abs=float32[N,6,12,2] = traj + 框中心 xy (LiDAR_TOP 系绝对位置),
                        score=float32[N,6]  log_softmax(不是概率),
                        traj_raw=float32[N+1,6,12,5] / score_raw=[N+1,6]  原始输出, 最后一行是 SDC(自车))
          plan     float32 [6,2]  = 规划路点(LiDAR_TOP 系 x 右 y 前, 0.5s 间隔共 3s)
          occ      dict(seg_out=uint8[5,200,200]) occ head 预测（0.5m 栅格, ±50m, 当前+未来4帧）；planning 碰撞优化用
          meta     dict(sample_token, scene_token, timestamp, seg_grid, occ_grid, motion_traj_kind, box_format, class_names, ...)
          bev_embed torch [bev_h*bev_w,1,256]（仅 return_bev=True 时；本帧 BEV encoder 输出，留在 GPU，供 BEV 层注入/warp）
        """
        feats = [f if f.dim() == 5 else f.unsqueeze(0) for f in feats]
        feats = [f.to(self.device) for f in feats]
        return self._step(feats, None, img_metas, command, l2g_t, l2g_r_mat, timestamp, return_raw, return_bev)

    @torch.no_grad()
    def step_bev(self, bev_embed, img_metas: dict, command: int, l2g_t=None, l2g_r_mat=None, timestamp=None,
                 return_raw=False, return_bev=False):
        """BEV 层注入：跳过 backbone/neck/BEV encoder，直接用给定 bev_embed（[bev_h*bev_w,1,256] 或 [bev_h*bev_w,256]，
        布局同 step(return_bev=True) 返回的 bev_embed：flatten 下标 = row*200+col，row↔LiDAR y、col↔LiDAR x，±51.2 m）
        跑 track/seg/motion/occ/planning，并照常推进流式状态（can_bus 基准、track queries、prev_bev:=注入的 BEV）。
        其余参数与返回同 step。"""
        return self._step(None, bev_embed, img_metas, command, l2g_t, l2g_r_mat, timestamp, return_raw, return_bev)

    def _step(self, feats, bev_embed, img_metas, command, l2g_t, l2g_r_mat, timestamp, return_raw, return_bev):
        m = self.model
        dev = self.device
        if l2g_r_mat is None:
            l2g_r_mat = img_metas['l2g_r_mat']
        l2g_r_mat = torch.as_tensor(np.asarray(l2g_r_mat), dtype=torch.float32, device=dev).reshape(1, 3, 3)
        if l2g_t is None:
            l2g_t = img_metas.get('l2g_t')
        if timestamp is None:
            timestamp = img_metas.get('timestamp')
        assert l2g_t is not None and timestamp is not None, 'step() 需要 l2g_t 与 timestamp（或 img_metas 里带 l2g_t/timestamp）'
        l2g_t = torch.as_tensor(np.asarray(l2g_t), dtype=torch.float32, device=dev).reshape(1, 3)
        timestamp = float(timestamp)
        cmd = torch.tensor([int(command)], dtype=torch.long, device=dev)

        t0 = time.perf_counter()
        raw = m.forward_from_feats(feats, img_metas, self.state, cmd, l2g_t, l2g_r_mat, timestamp,
                                   scene_start_canbus=self.scene_start_canbus, bev_override=bev_embed)
        self._tick('heads_total', t0)
        out = self._to_canonical(raw, img_metas, timestamp)
        if return_raw:
            out['_raw'] = raw
        if return_bev:
            out['bev_embed'] = raw['bev_embed'].clone()   # 与 state.prev_bev 不共享存储
        return out

    def step_images(self, img, img_metas: dict, command: int, l2g_t, l2g_r_mat, timestamp, **kw):
        """= step(extract_feats(img), ...)"""
        return self.step(self.extract_feats(img), img_metas, command, l2g_t, l2g_r_mat, timestamp, **kw)

    # 便捷：直接吃 UniADValData.get_frame() 的 dict
    def step_frame(self, feats, frame: dict, **kw):
        return self.step(feats, frame['img_metas'], frame['command'], frame['l2g_t'], frame['l2g_r_mat'],
                         frame['timestamp'], **kw)

    def step_frame_images(self, frame: dict, **kw):
        return self.step_frame(self.extract_feats(frame['img']), frame, **kw)

    def step_frame_bev(self, bev_embed, frame: dict, **kw):
        return self.step_bev(bev_embed, frame['img_metas'], frame['command'], frame['l2g_t'], frame['l2g_r_mat'],
                             frame['timestamp'], **kw)

    # ------------------------------------------------------------------ 输出规范化
    SEG_GRID = dict(
        shape_hw=(200, 200), x_range=(-51.2, 51.2), y_range=(-51.2, 51.2), resolution_m=0.512,
        frame='LiDAR_TOP of current frame (x right, y forward, z up)',
        index='arr[row, col]: row = floor((y + 51.2) / 0.512), col = floor((x + 51.2) / 0.512)  '
              '(row 0 = y -51.2 (后方), col 0 = x -51.2 (左侧)); 经验证：车辆中心落在 drivable 内的比例 '
              'row=y,col=x 为 0.51，其余 7 种行列/翻转假设 <=0.30',
        dtype='binary 0/1 float32 (UniAD seg head 本身输出二值栅格)')
    OCC_GRID = dict(shape_hw=(200, 200), x_range=(-50.0, 50.0), y_range=(-50.0, 50.0), resolution_m=0.5,
                    frame='LiDAR_TOP of current frame', index='arr[t, row, col]: row↔y, col↔x (同 seg)',
                    t='5 帧: 当前帧 + 未来 4 帧(0.5s 间隔)', dtype='uint8 0/1 (occ head, thresh 0.1 * track score)')

    @torch.no_grad()
    def _to_canonical(self, raw, img_metas, timestamp):
        boxes_native = raw['boxes'].float().cpu().numpy().astype(np.float32)
        boxes = boxes_native.copy()
        if self.convention == 'v1.0.1':
            # v1.0.1 权重输出 [x,y,z,w,l,h,yaw_old,vx,vy]，yaw_old = -yaw - pi/2（经 GT 验证，见 PORTING_NOTES）
            boxes[:, 3], boxes[:, 4] = boxes_native[:, 4], boxes_native[:, 3]
            yaw = -boxes_native[:, 6] - np.pi / 2
            boxes[:, 6] = (yaw + np.pi) % (2 * np.pi) - np.pi
        scores = raw['scores'].float().cpu().numpy().astype(np.float32)
        labels = raw['labels'].long().cpu().numpy().astype(np.int64)
        ids = raw['track_ids'].long().cpu().numpy().astype(np.int64)
        N = boxes.shape[0]

        seg = raw['seg']
        lane_cls = seg['lane'].cpu().numpy().astype(np.float32)          # [3,200,200] divider/crossing/contour 的 0/1
        lane_score = seg['lane_score'].float().cpu().numpy().astype(np.float32)
        drivable = seg['drivable'].cpu().numpy().astype(np.float32)
        seg_out = dict(
            lane=(lane_cls.sum(0) > 0).astype(np.float32),                # divider ∪ crossing ∪ contour
            drivable=drivable,
            divider=lane_cls[0], crossing=lane_cls[1], contour=lane_cls[2],
            lane_score=lane_score.max(0), drivable_score=seg['score_list'][-1].float().cpu().numpy().astype(np.float32),
            kind='binary_mask(0/1); *_score = mask probability (float), 仅供参考')

        mo = raw['motion']
        traj_raw = mo['traj'].float().cpu().numpy().astype(np.float32)       # [N+1,6,12,5]
        score_raw = mo['traj_scores'].float().cpu().numpy().astype(np.float32)  # [N+1,6] log_softmax
        assert traj_raw.shape[0] == N + 1, (traj_raw.shape, N)
        off = traj_raw[:N, :, :, :2].copy()
        # 官方 sample 的坐标约定(经 GT 验证): traj[...,:2] 为相对该 agent 当前框中心的累计位移，轴向 = LiDAR_TOP 轴(不随 agent 旋转)
        center = boxes_native[:, None, None, :2]
        motion = dict(traj=off, traj_abs=(off + center).astype(np.float32), score=score_raw[:N].copy(),
                      traj_raw=traj_raw, score_raw=score_raw)

        plan = raw['planning']['sdc_traj'].float().cpu().numpy().reshape(-1, 2).astype(np.float32)
        occ = raw['occ']
        occ_out = None
        if occ is not None and 'seg_out' in occ:
            occ_out = dict(seg_out=occ['seg_out'][0, :, 0].to(torch.uint8).cpu().numpy())

        meta = dict(sample_token=img_metas.get('sample_idx'), scene_token=img_metas.get('scene_token'),
                    timestamp=timestamp, seg_grid=self.SEG_GRID, occ_grid=self.OCC_GRID,
                    convention=self.convention,
                    motion_traj_kind='offset: traj/traj_raw[...,:2] = displacement from the agent own box center at t0 '
                                     '(LiDAR_TOP axes, cumulative, 12 steps x 0.5 s, K=6 modes); traj is [N,6,12,2] '
                                     'aligned with boxes (SDC row stripped); traj_abs = traj + box center xy; '
                                     'traj_raw is the native [N+1,6,12,5] (last row = SDC; ch2-4 = sigma_x,sigma_y,rho)',
                    motion_score_kind='log_softmax over the 6 modes (NOT probability; exp() gives probability)',
                    box_format='[x,y,z_bottom,l,w,h,yaw,vx,vy]; mmdet3d>=1.0 LiDAR_TOP convention (l along heading, '
                               'yaw = CCW about +z from +x of LiDAR_TOP = nuScenes GT yaw in the LiDAR frame); '
                               'z is bottom-center (gravity center z = z + h/2); vx,vy in m/s (LiDAR axes). '
                               'boxes_native keeps the raw network columns.',
                    class_names=['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                                 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'],
                    frame='LiDAR_TOP (x right, y forward, z up), current frame',
                    plan_kind='native UniAD sdc_traj (6 waypoints, 0.5 s apart, LiDAR_TOP axes, x right / y forward)',
                    plan_dt_s=0.5, n_tracks=N)
        return dict(boxes=boxes, boxes_native=boxes_native, scores=scores, labels=labels, track_ids=ids, seg=seg_out,
                    motion=motion, plan=plan, occ=occ_out, meta=meta)
