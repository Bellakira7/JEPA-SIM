"""UniAD 图像前端（ResNet101-DCN + FPN）的独立抽取器，权重取自完整 UniAD ckpt 的 img_backbone./img_neck.。

预处理照官方 base_e2e test pipeline（`uniad_port/PORTING_NOTES.md` §2 第 1 条）：BGR 读图 → (x - [103.53,116.28,123.675]) / 1
→ **不 resize（1.0×）** → 右/下补 0 到 32 倍数：900×1600 → 928×1600。
输出 4 层 FPN 特征，每层 [N_cam, 256, h, w]：116×200 / 58×100 / 29×50 / 15×25（stride 8/16/32/64）。
与 `uniad_port` 的 `UniADRunner.extract_feats` 逐位一致（自检：`python extractor.py --selfcheck`）。

两份 ckpt 的主干权重不同（GitHub v1.0.1 与 HF 2.0 是两次独立训练），抽特征必须用和下游 UniAD 同一份 ckpt。
"""
import featsim_paths as _release_paths
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from mmdet.models.backbones import ResNet
from mmdet.models.necks import FPN

MEAN = np.array([103.530, 116.280, 123.675], dtype=np.float32)
CAMS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']  # nuScenes/BEVFormer 顺序
SCALE = 1.0
CKPTS = {
    'gh': _release_paths.CKPT_GH,            # GitHub v1.0.1
    'hf': _release_paths.CKPT_HF,  # HF 2.0
}
DEFAULT_CKPT = 'hf'


def load_image(path, scale=SCALE):
    """BGR float32，(scale≠1 时 cv2 bilinear resize)，减均值，补 0 到 32 倍数 → ([3,H,W], (nh, nw))。"""
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w = im.shape[:2]
    if scale != 1.0:
        nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)
        im = cv2.resize(im.astype(np.float32), (nw, nh), interpolation=cv2.INTER_LINEAR)
    nh, nw = im.shape[:2]
    H, W = int(np.ceil(nh / 32) * 32), int(np.ceil(nw / 32) * 32)
    out = np.zeros((3, H, W), np.float32)
    for c in range(3):  # float32(x) - float32(mean)，与 UniADValData._load_images_fast 同一次 IEEE 舍入
        np.subtract(im[:, :, c], MEAN[c], out=out[c, :nh, :nw], dtype=np.float32)
    return torch.from_numpy(out), (nh, nw)


def _backbone_neck_state(ckpt):
    """ckpt: 'gh' / 'hf' / 完整 UniAD ckpt 路径 / 旧的 {'backbone','neck'} 拆分文件（feat/uniad_bb_neck.pth，= GitHub v1.0.1）。"""
    path = CKPTS.get(ckpt, ckpt)
    sd = torch.load(path, map_location='cpu', weights_only=False)
    if 'backbone' in sd and 'neck' in sd:
        return sd['backbone'], sd['neck']
    sd = sd.get('state_dict', sd)
    bb = {k[len('img_backbone.'):]: v for k, v in sd.items() if k.startswith('img_backbone.')}
    nk = {k[len('img_neck.'):]: v for k, v in sd.items() if k.startswith('img_neck.')}
    assert bb and nk, f'{path} 里没有 img_backbone./img_neck.'
    return bb, nk


class UniADFrontend(torch.nn.Module):
    def __init__(self, ckpt=DEFAULT_CKPT):
        super().__init__()
        self.backbone = ResNet(depth=101, num_stages=4, out_indices=(1, 2, 3), frozen_stages=4,
                               norm_cfg=dict(type='BN', requires_grad=False), norm_eval=True, style='caffe',
                               dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
                               stage_with_dcn=(False, False, True, True))
        self.neck = FPN(in_channels=[512, 1024, 2048], out_channels=256, start_level=0,
                        add_extra_convs='on_output', num_outs=4, relu_before_extra_convs=True)
        bb, nk = _backbone_neck_state(ckpt)
        self.backbone.load_state_dict(bb, strict=True)
        self.neck.load_state_dict(nk, strict=True)
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, img):  # img [N,3,H,W]
        
        # 11-03 实测）。clip_dataset 的 uint8 [T,6,3,H,W] 来自 HWC→CHW 转置后 np.stack，正是 channels_last 步长。
        return list(self.neck(self.backbone(img.contiguous())))


def selfcheck(ckpt, n_frames=2):
    """与 UniADRunner.extract_feats 逐位对拍（同一 GPU、同一 ckpt、val 场景 0 的前 n_frames 帧）；
    同时核对 clip_dataset 的 uint8 读图 + GPU 归一化路径与 UniADValData 的输入逐位相同。"""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(here), 'uniad_port'))
    sys.path.insert(0, here)
    from uniad_port.runner import UniADRunner
    from uniad_port.data import UniADValData
    from clip_dataset import load_u8, normalize_on_gpu
    dev = 'cuda'
    runner = UniADRunner(CKPTS.get(ckpt, ckpt), device=dev, verbose=False)
    fe = UniADFrontend(ckpt).to(dev)
    data = UniADValData(num_prefetch=0)
    ok = True
    for i, fr in enumerate(data.iter_scene(0)):
        if i >= n_frames:
            break
        x_ref = fr['img']                                               # [6,3,928,1600] float32 CPU
        x_u8 = torch.from_numpy(np.stack([load_u8(p) for p in fr['img_metas']['filename']]))
        x_gpu = normalize_on_gpu(x_u8.to(dev))
        d_in = (x_gpu.cpu() - x_ref).abs().max().item()
        x_li = torch.stack([load_image(p)[0] for p in fr['img_metas']['filename']])
        d_li = (x_li - x_ref).abs().max().item()
        f_ref = runner.extract_feats(x_ref)
        f_fe = fe(x_gpu)
        d = [(a - b).abs().max().item() for a, b in zip(f_ref, f_fe)]
        shapes = [tuple(f.shape) for f in f_fe]
        print(f'frame {i}: input maxdiff(clip_dataset)={d_in} (load_image)={d_li}  feats maxdiff per level={d}  shapes={shapes}')
        ok &= d_in == 0 and d_li == 0 and all(v == 0 for v in d)
        ok &= shapes == [(6, 256, 116, 200), (6, 256, 58, 100), (6, 256, 29, 50), (6, 256, 15, 25)]
    print(f'[selfcheck {ckpt}]', 'PASS (逐位一致)' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selfcheck', action='store_true')
    ap.add_argument('--ckpt', default=DEFAULT_CKPT, help="'gh' | 'hf' | ckpt 路径")
    a = ap.parse_args()
    if a.selfcheck:
        sys.exit(0 if selfcheck(a.ckpt) else 1)
