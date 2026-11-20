"""nuScenes 连续 T 帧 clip：图像（uint8 BGR，1.0× 原图 900×1600，补到 928×1600）、动作、layout。

- 图像预处理照官方 base_e2e test pipeline（**不 resize**，见 `uniad_port/PORTING_NOTES.md` §2），但**均值相减放到 GPU**
  （见 ``normalize_on_gpu``），worker 只做解码 + 补边（uint8，省传输）。与 `UniADValData` 的输入逐位相同（`extractor.py --selfcheck`）。
- actions[t] = 帧 t→t+1 自车相对位姿（帧 t 自车系）[dx, dy, dyaw]；最后一帧置 0。
- layout[t] = **目标帧 t+1** 的 6 路 filled layout（uint8，token 对齐画布 120×200，见 ``nusc_data.render_layout``）；最后一帧全黑（同主线约定）。

可选（默认全关，关掉时与加这两项之前逐位相同）：
- ``map_layout``：layout 里先画矢量地图（车道线 / 人行横道 / 道路边界，``nusc_map``），再叠框 = 设计稿 §3.3 的"完整条件"变体。
- ``sweeps``：相位增广（``nusc_sweeps``）。一个样本 = (场景, 起点 a, 相位 p)；p=0 是原关键帧链，p=1..5 是 12 Hz 相机 sweeps 组成的
  相位错开的 2 Hz 链。``__getitem__`` 里 p 在每次取样时**重新随机**（``phase_mode='random'``，训练用，样本数仍 = 关键帧 clip 数，
  每个 epoch 看到的相位不同），或固定成某个相位（``phase_mode=int``，调试/验证用）。
  ``phase_mode='all'`` 把 6 个相位全部展开成样本（约 ×6，仅供"数据量"口径核对，不建议训练用）。
- ``assoc_layout``：layout 变 6 通道，后 3 通道 = 关联通道（帧 t 的框画进目标帧 t+1 的相机，只画两帧都在场的实例，``nusc_data.render_assoc_layout``），
  FlowSim 逐 agent 关联描述的相机空间对应物；前 3 通道与不开时逐位相同。
"""
import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nusc_data as nd

MEAN_BGR = torch.tensor([103.530, 116.280, 123.675]).view(1, 3, 1, 1)
VALID_H, VALID_W, PAD_H = 900, 1600, 928
ACT_MEAN = np.array([2.9365, -0.0091, 0.0055], np.float32)
ACT_STD = np.array([1.7260, 0.0661, 0.0539], np.float32)


def load_u8(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    assert im is not None and im.shape == (VALID_H, VALID_W, 3), (path, None if im is None else im.shape)
    out = np.zeros((PAD_H, VALID_W, 3), np.uint8)
    out[:VALID_H] = im
    return np.ascontiguousarray(out.transpose(2, 0, 1))  # CHW 连续存储（DCNv2 对非 contiguous 输入静默算错，见 extractor.py）


def normalize_on_gpu(img_u8):
    """[...,3,928,1600] uint8 BGR → float32，(x-mean)/1，pad 行置 0（与官方"先归一化再补零"一致）。"""
    x = img_u8.float()
    shp = x.shape
    x = x.reshape(-1, 3, PAD_H, VALID_W) - MEAN_BGR.to(x.device)
    x[:, :, VALID_H:] = 0
    return x.reshape(shp).contiguous()


class NuscClips(Dataset):
    def __init__(self, split='train', T=5, step=1, max_clips=None, seed=0, with_layout=True, map_layout=False, sweeps=False,
                 phase_mode='random', assoc_layout=False):
        idx = nd.build_index()
        names = nd.splits()[split]
        self.T, self.with_layout, self.map_layout, self.sweeps = T, with_layout, map_layout, sweeps
        self.phase_mode = phase_mode
        self.assoc_layout = assoc_layout
        assert not (map_layout and not with_layout), 'map_layout 需要 layout'
        assert not (assoc_layout and not with_layout), 'assoc_layout 需要 layout'
        self.sf = None
        if sweeps:
            import nusc_sweeps as ns
            self.sf = ns.SweepFrames(idx)
        self.mv = None  # 每个 worker 懒加载（MapVectors 带 LRU，不跨进程共享）
        self.clips = []  # (scene, start)；相位另算
        self.idx = idx
        for s in names:
            n = len(idx['scenes'][s])
            for a in range(0, n - T + 1, step):
                self.clips.append((s, a))
        if max_clips is not None and len(self.clips) > max_clips:
            rng = np.random.RandomState(seed)
            sel = rng.choice(len(self.clips), max_clips, replace=False)
            self.clips = [self.clips[i] for i in sorted(sel)]
        self.n_phase = 6 if (sweeps and phase_mode == 'all') else 1

    def __len__(self):
        return len(self.clips) * self.n_phase

    def _phase(self, scene, a):
        if not self.sweeps:
            return 0
        if self.phase_mode == 'random':
            # p≥1 的链要关键帧 i+1，clip 末帧 a+T-1 之后必须还有一个关键帧
            ok = a + self.T < len(self.idx['scenes'][scene])
            return int(np.random.randint(0, 6)) if ok else 0
        p = int(self.phase_mode)
        return p if (p == 0 or a + self.T < len(self.idx['scenes'][scene])) else 0

    def _frames(self, scene, a, p):
        if p == 0:
            return list(self.idx['scenes'][scene][a:a + self.T])
        return [self.sf.frame(scene, a + t, p) for t in range(self.T)]

    def __getitem__(self, i):
        if self.phase_mode == 'all' and self.sweeps:
            i, p = divmod(i, 6)
            scene, a = self.clips[i]
            if p and a + self.T >= len(self.idx['scenes'][scene]):
                p = 0
        else:
            scene, a = self.clips[i]
            p = self._phase(scene, a)
        fr = self._frames(scene, a, p)
        T = self.T
        imgs = np.stack([np.stack([load_u8(f['cams'][c]['path']) for c in nd.CAMS]) for f in fr])  # [T,6,3,928,1600]
        act = np.zeros((T, 3), np.float32)
        for t in range(T - 1):
            act[t] = nd.relative_action(fr[t], fr[t + 1])
        act = (act - ACT_MEAN) / ACT_STD
        act[T - 1] = 0
        out = dict(imgs=torch.from_numpy(imgs), actions=torch.from_numpy(act),
                   tokens=[f['token'] for f in fr], phase=p)
        if self.with_layout:
            if self.map_layout and self.mv is None:
                from nusc_map import MapVectors
                self.mv = MapVectors()
            H, W = nd.LAYOUT_HW
            lay = np.zeros((T, 6, 3, H, W), np.uint8)
            for t in range(T - 1):
                f = fr[t + 1]
                lay[t] = nd.render_layout(f, map_vec=self.mv.for_frame(f) if self.map_layout else None)
            if self.assoc_layout:
                # 关联通道（6 通道 = 主 3 + 关联 3）：layout[t] 的关联部分 = 帧 t 的框画进目标帧 t+1 的相机（nd.render_layout_cond）
                lay = np.concatenate([lay, np.zeros_like(lay)], axis=2)
                for t in range(T - 1):
                    lay[t, :, 3:] = nd.render_assoc_layout(fr[t], fr[t + 1])
            out['layout'] = torch.from_numpy(lay)
        return out


def collate(b):
    return dict(imgs=torch.stack([x['imgs'] for x in b]), actions=torch.stack([x['actions'] for x in b]),
                layout=torch.stack([x['layout'] for x in b]) if 'layout' in b[0] else None,
                tokens=[x['tokens'] for x in b], phase=[x['phase'] for x in b])
