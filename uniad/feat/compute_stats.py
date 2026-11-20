"""各层各通道均值/方差（train 随机抽样帧）+ copy-last 参照（标准化空间 L1）。1.0×（928×1600）输入。

用法：
  CUDA_VISIBLE_DEVICES=2 ../venv/bin/python compute_stats.py --n 50  --ckpt hf --out level_stats_1x_hf_smoke.pt   # 冒烟（~1–2 min）

两份 ckpt 的主干不同，统计量必须和训练/评测用的 ckpt 一致（文件名带 ckpt 标签）。

copy-last 与"预测通道均值"两个参照（标准化空间、所有位置等权）：
  copy  = mean_c (Σ|f_{t+1}-f_t|_c / cnt) / σ_c             —— 逐通道精确累加，不依赖 μ
  mean  = mean_c E|f_{t+1}-μ_c| / σ_c                       —— 在每帧随机抽 ``--sub`` 个位置的子样本上算（μ 要等全部帧算完才知道）
1.0× 下一帧 4 层 × 6 相机约 18.5 万位置 × 256 ch（fp32 每帧 ~190 MB），不再把整帧特征存到 CPU。
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_dataset import NuscClips, collate, normalize_on_gpu
from extractor import UniADFrontend

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=2000, help='train 中随机抽的 (t, t+1) 帧对数')
ap.add_argument('--ckpt', default='hf', help="'hf' | 'gh' | 完整 UniAD ckpt 路径")
ap.add_argument('--out', required=True)
ap.add_argument('--sub', type=int, default=2048, help='每帧每层抽多少个 (相机,位置) 进"预测均值"子样本')
ap.add_argument('--workers', type=int, default=12)
a = ap.parse_args()

ds = NuscClips('train', T=2, step=1, max_clips=a.n, with_layout=False)
dl = DataLoader(ds, batch_size=1, num_workers=a.workers, collate_fn=collate)
fe = UniADFrontend(a.ckpt).cuda()
L = 4
s1 = [torch.zeros(256, dtype=torch.float64, device='cuda') for _ in range(L)]
s2 = [torch.zeros(256, dtype=torch.float64, device='cuda') for _ in range(L)]
sc = [torch.zeros(256, dtype=torch.float64, device='cuda') for _ in range(L)]  # Σ|f1-f0| 逐通道
cnt = [0] * L
sub = [[] for _ in range(L)]
g = torch.Generator(device='cuda').manual_seed(0)
t0 = time.time()
for i, b in enumerate(dl):
    x = normalize_on_gpu(b['imgs'][0].cuda())  # [2,6,3,928,1600]
    fs = fe(x.flatten(0, 1))  # list of [12,256,h,w]
    for l, f in enumerate(fs):
        f = f.double()
        s1[l] += f.sum((0, 2, 3)); s2[l] += (f * f).sum((0, 2, 3)); cnt[l] += f.shape[0] * f.shape[2] * f.shape[3]
        f0, f1 = f[:6], f[6:]
        sc[l] += (f1 - f0).abs().sum((0, 2, 3))
        flat = f1.permute(0, 2, 3, 1).reshape(-1, 256)
        sub[l].append(flat[torch.randint(flat.shape[0], (a.sub,), generator=g, device='cuda')].float().cpu())
    if i % 50 == 0:
        print(i, f'{time.time() - t0:.0f}s', flush=True)
mean = [(s1[l] / cnt[l]).float().cpu() for l in range(L)]
std = [((s2[l] / cnt[l]) - (s1[l] / cnt[l]) ** 2).clamp_min(1e-6).sqrt().float().cpu() for l in range(L)]
n_pos_pair = [cnt[l] // 2 for l in range(L)]
copy_l1 = [float(((sc[l].cpu() / n_pos_pair[l]).float() / std[l]).mean()) for l in range(L)]
mean_l1 = [float(((torch.cat(sub[l]) - mean[l]).abs() / std[l]).mean()) for l in range(L)]
torch.save(dict(mean=mean, std=std, n_frames=2 * len(ds), ckpt=a.ckpt, scale=1.0,
                level_grids=[tuple(f.shape[-2:]) for f in fs], copy_l1=copy_l1, mean_l1=mean_l1), a.out)
print(f'saved {a.out}  ({2 * len(ds)} frames, {time.time() - t0:.0f}s)')
print('mean|std per level:', [(round(float(m.mean()), 4), round(float(s.mean()), 4)) for m, s in zip(mean, std)])
for l in range(L):
    print(f'level P{l + 3}: copy-last L1 {copy_l1[l]:.3f}   predict-mean L1 {mean_l1[l]:.3f}   ratio {copy_l1[l] / mean_l1[l]:.3f}')
