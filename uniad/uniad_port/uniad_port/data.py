"""UniAD base_e2e 在 nuScenes val 上做流式推理所需的数据读取 + 规划 L2 统计器。

只复现「推理路径」需要的字段，数值上对齐官方 test pipeline：

  LoadMultiViewImageFromFilesInCeph(to_float32=True)      -> mmcv.imread(path,'unchanged') (BGR uint8) -> float32
  [可选] RandomScaleImageMultiViewImage(scales=[s])        -> mmcv.imresize + lidar2img 左乘 diag(s,s,1,1)（官方 base_e2e 没用）
  NormalizeMultiviewImage(mean=[103.53,116.28,123.675], std=1, to_rgb=False)
  PadMultiViewImage(size_divisor=32)                       -> 归一化之后在右/下补 0
  DefaultFormatBundle3D                                    -> HWC->CHW, 6 相机 stack
  CustomCollect3D                                          -> img_metas（键见 get_frame）

官方代码（只读参考，没有 import）：
  projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py : get_data_info / prepare_test_data
  projects/mmdet3d_plugin/datasets/pipelines/{loading,transform_3d}.py
  projects/mmdet3d_plugin/datasets/data_utils/trajectory_api.py : get_sdc_planning_label
  projects/mmdet3d_plugin/uniad/apis/test.py + dense_heads/planning_head_plugin/planning_metrics.py

注意：
  * 模块 import 时不读 pkl（构造 UniADValData 时才读）。
  * 纯 CPU，不碰 CUDA。图像张量在 CPU。
  * mmdet3d 的 Box3DMode / LiDARInstance3DBoxes 在首次 get_frame 时才懒加载（import 较重）。
"""
from __future__ import annotations
import featsim_paths as _release_paths

import collections
import json
import math
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Union

import cv2
import mmcv
import numpy as np
import torch
from pyquaternion import Quaternion

# ----------------------------------------------------------------------------
# 默认路径 / 常量
# ----------------------------------------------------------------------------
DEFAULT_INFOS_PKL = _release_paths.INFOS_PKL
DEFAULT_IMG_ROOT = _release_paths.NUSC_ROOT

# 官方 base_e2e 的 img_norm_cfg（BGR 顺序保持不变，to_rgb=False）
IMG_NORM_MEAN = np.array([103.530, 116.280, 123.675], dtype=np.float32)
IMG_NORM_STD = np.array([1.0, 1.0, 1.0], dtype=np.float32)
SIZE_DIVISOR = 32

# 官方 test 里 6 相机的顺序（= infos['cams'] 的字典顺序）
CAM_ORDER = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
ORI_H, ORI_W = 900, 1600          # nuScenes 相机原图分辨率

PLANNING_STEPS = 6                 # 3 s / 0.5 s
# 官方 pseudo-sdc box：[x,y,z,l,w,h,yaw]=[0,0,0,4.08,1.73,1.56,0.5pi]，origin=(0.5,0.5,0.5)。
# LiDARInstance3DBoxes 构造时会把 origin 换算成 (0.5,0.5,0)（底面中心），
# 于是 box 的 z 变成 0 + h*(0-0.5) = -0.78（float32）。官方后续的 rotate/translate 都作用在这个点上，
# 而不是 lidar 原点；因为 lidar2ego/ego2global 带有微小的 roll/pitch，会对 xy 有 ~1e-3 m 量级的影响，这里照搬。
_SDC_BOX_H_F32 = np.float32(1.56)
SDC_POINT_LIDAR = np.array([0.0, 0.0, float(-np.float32(0.5) * _SDC_BOX_H_F32)], dtype=np.float64)

_BOX_TYPES = None


def _get_box_types():
    """懒加载 mmdet3d 1.4 的 Box3DMode / LiDARInstance3DBoxes。"""
    global _BOX_TYPES
    if _BOX_TYPES is None:
        from mmdet3d.structures import Box3DMode, LiDARInstance3DBoxes
        _BOX_TYPES = (Box3DMode.LIDAR, LiDARInstance3DBoxes)
    return _BOX_TYPES


def _quaternion_yaw(q: Quaternion) -> float:
    # 与 nuscenes.eval.common.utils.quaternion_yaw 一致（避免 import 整个 eval 模块）
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
    return float(np.arctan2(v[1], v[0]))


# ----------------------------------------------------------------------------
# 数据集
# ----------------------------------------------------------------------------
class UniADValData:
    """nuScenes val（官方 150 场景）流式输入。

    scenes: list[dict(scene_name, scene_token, frame_ids)]，frame_ids 为全局 info 下标，场景内按时间序。
    """

    def __init__(self,
                 infos_pkl: str = DEFAULT_INFOS_PKL,
                 img_root: str = DEFAULT_IMG_ROOT,
                 scale: float = 1.0,
                 scene_names: Optional[Sequence[str]] = None,
                 num_prefetch: int = 4,
                 scene_json: Optional[str] = None):
        self.infos_pkl = infos_pkl
        self.img_root = img_root.rstrip('/')
        self.scale = float(scale)
        self.num_prefetch = int(num_prefetch)
        # 每个 worker 线程内 cv2 单线程，避免多线程互相抢核
        cv2.setNumThreads(1)
        _get_box_types()   # 构造时就加载 mmdet3d box 类（import 较慢，避免在 worker 线程里首次 import）

        with open(infos_pkl, 'rb') as f:   # 内部可信文件
            data = pickle.load(f)
        self.infos: List[dict] = data['infos'] if isinstance(data, dict) else data
        self.token2idx: Dict[str, int] = {i['token']: k for k, i in enumerate(self.infos)}
        assert len(self.token2idx) == len(self.infos), 'info token 重复'

        # 场景名 <-> token
        scene_json = scene_json or os.path.join(self.img_root, 'v1.0-trainval', 'scene.json')
        with open(scene_json) as f:
            self.scene_token2name = {s['token']: s['name'] for s in json.load(f)}

        # 校验相机顺序（官方依赖 dict 顺序）
        for i in self.infos:
            assert tuple(i['cams'].keys()) == CAM_ORDER, f"cams 顺序异常: {tuple(i['cams'].keys())}"

        # 按出现顺序分组，场景内按 (frame_idx, timestamp) 排序
        groups: Dict[str, List[int]] = collections.OrderedDict()
        for k, i in enumerate(self.infos):
            groups.setdefault(i['scene_token'], []).append(k)
        scenes = []
        for tok, ids in groups.items():
            ids = sorted(ids, key=lambda k: (self.infos[k]['frame_idx'], self.infos[k]['timestamp']))
            scenes.append(dict(scene_name=self.scene_token2name[tok], scene_token=tok, frame_ids=ids))
        if scene_names is not None:
            want = set(scene_names)
            scenes = [s for s in scenes if s['scene_name'] in want]
            missing = want - {s['scene_name'] for s in scenes}
            assert not missing, f'infos 里没有这些场景: {sorted(missing)}'
        self.scenes: List[dict] = scenes
        self._name2scene = {s['scene_name']: n for n, s in enumerate(scenes)}
        self._info2scene = {}
        for n, s in enumerate(scenes):
            for k in s['frame_ids']:
                self._info2scene[k] = n

    # ------------------------------------------------------------------ 基本信息
    def __len__(self):
        return sum(len(s['frame_ids']) for s in self.scenes)

    def _remap_path(self, p: str) -> str:
        """'./data/nuscenes/samples/CAM_FRONT/x.jpg' -> '<img_root>/samples/CAM_FRONT/x.jpg'"""
        j = p.find('samples/')
        assert j >= 0, p
        return os.path.join(self.img_root, p[j:])

    def _resolve_scene(self, scene_idx_or_name: Union[int, str]) -> dict:
        if isinstance(scene_idx_or_name, str):
            return self.scenes[self._name2scene[scene_idx_or_name]]
        return self.scenes[int(scene_idx_or_name)]

    # ------------------------------------------------------------------ 几何
    @staticmethod
    def _lidar2img(info: dict, scale: float) -> List[np.ndarray]:
        """官方 get_data_info 的 lidar2img（float64 4x4）；scale!=1 时左乘 diag(s,s,1,1)。"""
        out = []
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= scale
        scale_factor[1, 1] *= scale
        for cam_info in info['cams'].values():
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            l2i = viewpad @ lidar2cam_rt.T
            if scale != 1.0:
                l2i = scale_factor @ l2i       # RandomScaleImageMultiViewImage
            out.append(l2i)
        return out

    @staticmethod
    def _transforms(info: dict):
        """lidar->ego, ego->global 的 (R, t)，与 get_l2g_transform 一致（float64）。"""
        l2e_r_mat = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        l2e_t = np.array(info['lidar2ego_translation'], dtype=np.float64)
        e2g_r_mat = Quaternion(info['ego2global_rotation']).rotation_matrix
        e2g_t = np.array(info['ego2global_translation'], dtype=np.float64)
        return l2e_r_mat, l2e_t, e2g_r_mat, e2g_t

    def _l2g(self, info: dict):
        l2e_r_mat, l2e_t, e2g_r_mat, e2g_t = self._transforms(info)
        l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T              # 与官方 get_data_info 逐字一致
        l2g_t = l2e_t @ e2g_r_mat.T + e2g_t
        return l2g_r_mat.astype(np.float32), l2g_t.astype(np.float32)

    @staticmethod
    def _can_bus(info: dict) -> np.ndarray:
        """官方 get_data_info 末段对 can_bus 的改写（绝对位姿，帧间差由模型侧做）。拷贝，不改 info。"""
        can_bus = np.array(info['can_bus'], dtype=np.float64).copy()
        rotation = Quaternion(info['ego2global_rotation'])
        translation = info['ego2global_translation']
        can_bus[:3] = translation
        can_bus[3:7] = rotation.elements
        patch_angle = _quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        return can_bus

    # ------------------------------------------------------------------ GT 规划
    def planning_label(self, info_idx: int):
        """照官方 get_sdc_planning_label，沿 infos 的 next 链，不依赖 NuScenes 对象。

        返回 (planning [6,3] float32 (x,y,yaw；yaw 列补 0), mask [6] bool, command int)。
        坐标系：当前帧 LiDAR 系（x 向右，y 向前）。
        """
        info0 = self.infos[info_idx]
        l2e_r_i, l2e_t_i, e2g_r_i, e2g_t_i = self._transforms(info0)
        planning = np.zeros((PLANNING_STEPS, 3), dtype=np.float64)
        mask = np.zeros(PLANNING_STEPS, dtype=bool)
        info = info0
        for k in range(PLANNING_STEPS):
            nxt = info['next']
            if nxt == '':
                break
            info = self.infos[self.token2idx[nxt]]
            l2e_r, l2e_t, e2g_r, e2g_t = self._transforms(info)
            p = SDC_POINT_LIDAR
            p = l2e_r @ p + l2e_t              # lidar_next -> ego_next
            p = e2g_r @ p + e2g_t              # ego_next -> global
            p = e2g_r_i.T @ (p - e2g_t_i)      # global -> ego_init
            p = l2e_r_i.T @ (p - l2e_t_i)      # ego_init -> lidar_init
            planning[k, :2] = p[:2]
            mask[k] = True
        if not mask.any():
            command = 2                        # FORWARD
        else:
            x_last = planning[mask][-1][0]     # 最后一个有效步的 x
            if x_last >= 2:
                command = 0                    # RIGHT
            elif x_last <= -2:
                command = 1                    # LEFT
            else:
                command = 2                    # FORWARD
        return planning.astype(np.float32), mask, command

    # ------------------------------------------------------------------ 图像
    def _load_images(self, paths: Sequence[str]):
        """读 6 张图 -> ([6,3,Hp,Wp] float32 torch, ori_shapes, pad_shapes)。

        scale==1.0（官方）走融合快路径 _load_images_fast；scale!=1.0 走 _load_images_ref（保持官方
        “先 float32 resize、再归一化”的顺序）。两者在 scale==1.0 时逐位相同（tests/test_data.py 校验）。
        """
        if self.scale == 1.0:
            return self._load_images_fast(paths)
        return self._load_images_ref(paths)

    def _load_images_fast(self, paths: Sequence[str]):
        """融合实现：uint8 -> 逐通道 float32(x) - float32(mean) 直接写进预先置 0 的 [6,3,Hp,Wp] 输出。

        与官方算子链数值逐位相同：mmcv.imnormalize 的 cv2.subtract 在 float32 上就是 float32(x)-float32(mean)
        （单次 IEEE 舍入）；std==1 时 cv2.multiply(·,1.0) 是恒等；补边是常数 0 的右/下补边，即输出的初始 0。
        省掉 astype / copy / copyMakeBorder / HWC->CHW stack 四次整图拷贝，单帧计算 ~0.3s -> ~0.1s。
        """
        assert np.all(IMG_NORM_STD == 1.0), '快路径假设 std==1'
        out = None
        ori_shapes, pad_shapes = [], []
        for i, p in enumerate(paths):
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)           # BGR uint8，同 mmcv.imread(p,'unchanged')
            assert im is not None and im.dtype == np.uint8 and im.ndim == 3 and im.shape[2] == 3, p
            h, w = im.shape[:2]
            if out is None:
                ph = int(math.ceil(h / SIZE_DIVISOR)) * SIZE_DIVISOR
                pw = int(math.ceil(w / SIZE_DIVISOR)) * SIZE_DIVISOR
                out = np.zeros((len(paths), 3, ph, pw), dtype=np.float32)
            ph, pw = out.shape[2:]
            assert (ph, pw) == (int(math.ceil(h / SIZE_DIVISOR)) * SIZE_DIVISOR, int(math.ceil(w / SIZE_DIVISOR)) * SIZE_DIVISOR), \
                '6 个相机分辨率应一致'
            for c in range(3):
                np.subtract(im[:, :, c], IMG_NORM_MEAN[c], out=out[i, c, :h, :w], dtype=np.float32)
            ori_shapes.append((h, w, 3))
            pad_shapes.append((ph, pw, 3))
        return torch.from_numpy(out), ori_shapes, pad_shapes

    def _load_images_ref(self, paths: Sequence[str]):
        """参考实现：逐算子调用 mmcv（与官方 pipeline 一一对应），也用于 scale!=1.0。"""
        chw, ori_shapes, pad_shapes = [], [], []
        for p in paths:
            img = mmcv.imread(p, 'unchanged')                 # BGR uint8
            img = img.astype(np.float32)                      # to_float32
            if self.scale != 1.0:                             # RandomScaleImageMultiViewImage（可选）
                h, w = img.shape[:2]
                img = mmcv.imresize(img, (int(w * self.scale), int(h * self.scale)), return_scale=False)
            img = mmcv.imnormalize(img, IMG_NORM_MEAN, IMG_NORM_STD, to_rgb=False)
            ori_shapes.append(tuple(img.shape))               # PadMultiViewImage: ori_shape=补边前
            img = mmcv.impad_to_multiple(img, SIZE_DIVISOR, pad_val=0)   # 归一化之后补 0
            pad_shapes.append(tuple(img.shape))
            chw.append(img.transpose(2, 0, 1))
        arr = np.ascontiguousarray(np.stack(chw, axis=0))
        return torch.from_numpy(arr), ori_shapes, pad_shapes

    def _expected_shapes(self):
        """不读图时的 (ori_shape, pad_shape)——原图分辨率是常数 900x1600。"""
        if self.scale != 1.0:
            h, w = int(ORI_H * self.scale), int(ORI_W * self.scale)
        else:
            h, w = ORI_H, ORI_W
        ph = int(math.ceil(h / SIZE_DIVISOR)) * SIZE_DIVISOR
        pw = int(math.ceil(w / SIZE_DIVISOR)) * SIZE_DIVISOR
        return [(h, w, 3)] * 6, [(ph, pw, 3)] * 6

    # ------------------------------------------------------------------ 取帧
    def get_frame(self, info_idx: int, load_images: bool = True) -> dict:
        """返回一帧的输入。img 为 CPU float32 torch [6,3,H,W]；load_images=False 时 img=None。"""
        info = self.infos[info_idx]
        box_mode_3d, box_type_3d = _get_box_types()
        paths = [self._remap_path(c['data_path']) for c in info['cams'].values()]

        if load_images:
            img, ori_shapes, pad_shapes = self._load_images(paths)
        else:
            img = None
            ori_shapes, pad_shapes = self._expected_shapes()

        l2g_r_mat, l2g_t = self._l2g(info)
        scene_name = self.scene_token2name[info['scene_token']]
        sdc_planning, sdc_planning_mask, command = self.planning_label(info_idx)

        img_metas = dict(
            filename=paths,
            ori_shape=ori_shapes,
            img_shape=list(pad_shapes),
            pad_shape=list(pad_shapes),
            lidar2img=self._lidar2img(info, self.scale),
            scale_factor=1.0,
            flip=False,
            pcd_horizontal_flip=False,
            pcd_vertical_flip=False,
            pcd_scale_factor=1.0,
            box_mode_3d=box_mode_3d,
            box_type_3d=box_type_3d,
            img_norm_cfg=dict(mean=IMG_NORM_MEAN.copy(), std=IMG_NORM_STD.copy(), to_rgb=False),
            sample_idx=info['token'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            pts_filename=self._remap_path(info['lidar_path']),
            scene_token=info['scene_token'],
            can_bus=self._can_bus(info),
            l2g_r_mat=l2g_r_mat.copy(),
            l2g_t=l2g_t.copy(),                            # [UniAD-port] 额外放进 img_metas，便于 runner.step(feats, img_metas, command) 直接调用
            timestamp=info['timestamp'] / 1e6,
        )
        return dict(
            img=img,
            img_metas=img_metas,
            l2g_r_mat=torch.from_numpy(l2g_r_mat),         # 官方 prepare_test_data 对 l2g* 做 to_tensor
            l2g_t=torch.from_numpy(l2g_t),
            timestamp=info['timestamp'] / 1e6,             # 秒，python float
            sample_token=info['token'],
            scene_token=info['scene_token'],
            scene_name=scene_name,
            frame_idx=info['frame_idx'],
            info_idx=info_idx,
            img_scale=self.scale,
            # --- GT 规划（仅评测用） ---
            sdc_planning=sdc_planning,                     # float32 [6,3]
            sdc_planning_mask=sdc_planning_mask,           # bool [6]
            command=command,                               # 0 RIGHT / 1 LEFT / 2 FORWARD
        )

    def iter_scene(self, scene_idx_or_name: Union[int, str], load_images: bool = True):
        """按时间序 yield get_frame 的结果；用线程池预取 num_prefetch 帧（cv2 解码释放 GIL）。"""
        sc = self._resolve_scene(scene_idx_or_name)
        ids = sc['frame_ids']
        if self.num_prefetch <= 0 or not load_images:
            for k in ids:
                yield self.get_frame(k, load_images=load_images)
            return
        pool = ThreadPoolExecutor(max_workers=self.num_prefetch)
        q = collections.deque()
        try:
            nxt = 0
            while nxt < len(ids) and len(q) < self.num_prefetch:
                q.append(pool.submit(self.get_frame, ids[nxt], load_images))
                nxt += 1
            while q:
                fut = q.popleft()
                if nxt < len(ids):
                    q.append(pool.submit(self.get_frame, ids[nxt], load_images))
                    nxt += 1
                yield fut.result()
        finally:
            for fut in q:
                fut.cancel()
            pool.shutdown(wait=True)


# ----------------------------------------------------------------------------
# 规划 L2 统计器
# ----------------------------------------------------------------------------
class PlanningL2Acc:
    """对照官方 PlanningMetric.update/compute_L2 + NuScenesE2EDataset.evaluate 的两种口径。

    官方逐帧：
        pred[...,0] = -pred[...,0]; gt[...,0] = -gt[...,0]        （对 L2 无影响，保留）
        L2[k] = sqrt( sum_xy( (pred-gt)^2 * mask[k] ) )            （mask 无效步 L2 记 0）
        self.L2 += L2;  self.total += 1                            （total 含 GT 不足 6 步的帧）
        compute() = L2_sum / total                                 -> 6 个值（0.5s..3.0s）

    两种口径（compute() 返回的顶层键）：
      * 'official'   : 分母 = 总帧数 total（GT 不足 6 步的帧在无效步贡献 0，会把均值拉低）。
      * 'valid_only' : 分母 = 该时间步 mask 有效的帧数（每步各自归一化）。
    每种口径再给两种 planning_evaluation_strategy（官方 evaluate()）：
      * 'uniad': 时间步 i 的值本身（0.5s..3.0s 共 6 个）。报告里 1s=index1, 2s=index3, 3s=index5。
      * 'stp3' : 前 i+1 步 per-step 值的均值，即 value[:i+1].mean()。
    'avg' 的定义（两种 strategy 都一样）：6 个 per-step 值的算术平均 = stp3 的 3.0s 值。
    另外额外给 'avg_123' = (1.0s + 2.0s + 3.0s)/3（UniAD 论文表格里 Avg 的常见口径，仅 uniad strategy 有意义，
    stp3 里同样按 strategy 取值后再平均 1s/2s/3s 三列）。

    精度说明：官方的累加器是 float32 torch buffer，这里用 float64，差异 ~1e-7 量级。
    """

    N = PLANNING_STEPS

    def __init__(self):
        self.reset()

    def reset(self):
        self.l2_sum = np.zeros(self.N, dtype=np.float64)
        self.n_valid = np.zeros(self.N, dtype=np.int64)
        self.n_frames = 0

    def update(self, pred_xy: np.ndarray, gt_xy: np.ndarray, gt_mask: np.ndarray):
        """逐帧。pred_xy/gt_xy: [6,2]（当前帧 LiDAR 系，x 右 y 前）；gt_mask: [6] bool。"""
        pred = np.array(pred_xy, dtype=np.float64)[:self.N, :2].copy()
        gt = np.array(gt_xy, dtype=np.float64)[:self.N, :2].copy()
        mask = np.asarray(gt_mask).astype(bool)[:self.N]
        assert pred.shape == (self.N, 2) and gt.shape == (self.N, 2) and mask.shape == (self.N,)
        pred[:, 0] = -pred[:, 0]
        gt[:, 0] = -gt[:, 0]
        l2 = np.sqrt((((pred - gt) ** 2) * mask[:, None]).sum(-1))      # [6]，无效步 = 0
        self.l2_sum += l2
        self.n_valid += mask.astype(np.int64)
        self.n_frames += 1

    def merge(self, other: 'PlanningL2Acc') -> 'PlanningL2Acc':
        """多进程累加：把 other 的累计量并入 self（原地），返回 self。"""
        self.l2_sum += other.l2_sum
        self.n_valid += other.n_valid
        self.n_frames += other.n_frames
        return self

    @staticmethod
    def _table(per_step: np.ndarray) -> dict:
        """per_step: 6 个 per-step 值 -> {'0.5s'..'3.0s','avg','avg_123'}"""
        out = {f'{0.5 * (i + 1):.1f}s': float(per_step[i]) for i in range(len(per_step))}
        out['avg'] = float(np.mean(per_step))
        out['avg_123'] = float(np.mean([per_step[1], per_step[3], per_step[5]]))
        return out

    def _strategies(self, per_step: np.ndarray) -> dict:
        # avg 两种 strategy 都定义为 6 个 per-step 值的算术平均（= stp3 的 3.0s 值）
        uniad = per_step
        stp3 = np.array([per_step[:i + 1].mean() for i in range(self.N)])
        t_uniad = self._table(uniad)
        t_stp3 = self._table(stp3)
        t_uniad['avg'] = t_stp3['avg'] = float(np.mean(per_step))
        t_stp3['avg_123'] = float(np.mean([stp3[1], stp3[3], stp3[5]]))
        return dict(uniad=t_uniad, stp3=t_stp3)

    def compute(self) -> dict:
        total = max(self.n_frames, 1)
        official = self.l2_sum / total
        with np.errstate(invalid='ignore', divide='ignore'):
            valid_only = np.where(self.n_valid > 0, self.l2_sum / np.maximum(self.n_valid, 1), np.nan)
        return dict(
            official=self._strategies(official),
            valid_only=self._strategies(valid_only),
            n_frames=int(self.n_frames),
            n_valid_per_step=[int(v) for v in self.n_valid],
        )
