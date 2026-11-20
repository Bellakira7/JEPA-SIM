"""PyramidPredictor：UniAD 4 层 FPN 特征的动作条件化多相机预测器（V-JEPA2-AC 式，复用主线 ac_modules / layout_adapter）。

与主线 ``MultiCamACPredictor`` 的差别（只为适配 UniAD 的多尺度金字塔，每条写明）：
1. **每个 (相机, 金字塔层) 是一路 token 流**。UniAD 1.0× 输入（928×1600，`uniad_port/PORTING_NOTES.md` §2）每相机 4 层
   [116×200, 58×100, 29×50, 15×25]（stride 8/16/32/64），各层用不同 ``pool``（默认 8/4/2/1）切成**同一个 15×25 token 网格**。
   116/8、58/4、29/2 不是整数：**右/下补零到 pool 整数倍（120×200 / 60×100 / 30×50）再整池化，输出 pixel_shuffle 后裁回原网格**
   （``pad_to_tokens`` / ``crop_from_tokens``，裁剪是补零的精确逆）。不用 adaptive pool 的理由：补零保持各层左上角对齐，
   token (i,j) 在 P3/P4/P5 上**恰好**覆盖同一块 64×64 px 图像区域 [64i,64i+64)×[64j,64j+64)（P6 = P5 上 stride-2 卷积，
   中心落在同一块内），且每层每个 token 内的子格数固定（PatchEmbed/proj 的 p×p 排布处处同义）；adaptive pool 的窗口
   边界逐层不同（116→15 与 58→15 的切分点不重合），层间同一 token 对应的图像区域不一致，逆变换也不唯一。
   代价：最后一行 token 有部分是补零（P3 116→120 补 4 行，原图本身 900→928 也是补边），与 FPN 在补边处的行为一致。
   RoPE 用同一套 (h,w) 坐标，每相机 1500 token、每帧 9000 token。
2. **PatchEmbed / 输出 proj 按层各一份、跨相机共享**（主线是每相机一份；6 个 nuScenes 相机同型号，再各配一份会让
   P3 层的 8×8 patch 嵌入就有 6×6.3M 参数）。相机身份靠 ``cam_embed``，层身份靠 ``level_embed``。
3. layout 条件：**一个** ``CamLayoutAdapter``（replace_main，pixel_unshuffle 因子 = P3 pool=8）吃 120×200 的 token 对齐画布
   （``nusc_data.render_layout``：原图 ÷8，与 P3 栅格同一几何），输出 15×25 网格上的 [B,T,375,D]，**同一份加到四层的 token 上**
   （层间靠 level_embed 区分；layout 是语义量，不需要每层单独学）。
4. action：3 维 [Δx, Δy, Δyaw]（自车系），``Linear(3,D)`` → 每个 ACBlock 的 AdaLN（与主线新口径相同，无 state）。
5. 可选 ``residual``：输出 = 输入（copy-last）+ Δ，proj 零初始化。
6. 可选 ``warp_levels``（G2 之后的方案，设计稿 §3.2）：这些层的输出 = **流场 warp 后的上一帧同层特征** + Δ。G2 显示细层（P3/P4）只靠 token 级信息
   还原不了（P4 降到 token 网格 Det IoU 只剩 0.78），补上一帧细层又因自车运动而错位（检测反而更差），所以让 token 同时输出一个
   每格 2 维的流场（``flow_proj``：384 → 2·p²，pixel_shuffle 展开到该层网格，单位 = ``flow_px`` 像素 / 该层 stride），
   用 ``grid_sample`` 把上一帧细层搬到当前位置，再加 Δ（原来的 ``proj``）。流场与 Δ 均零初始化 → 初始输出 ≈ copy-last（warp 恒等），训练只学偏离。
   零流场时输出 ≈ copy-last（``grid_sample`` 的归一化坐标有 fp32 舍入，P3 最大偏差约 1e-4，特征标准差约 1，可忽略；不是逐位相同）。
   没有深度，流场是 2D 图像平面位移；这是"带运动补偿的可学上采样器"的最小实现，不依赖相机参数。
7. 可选 ``assoc_layout``（FlowSim 逐 agent 关联描述 [s_t, s_{t-1}, Δs, valid] 的相机空间对应物）：layout 变 6 通道，后 3 通道 = 上下文末帧 t 的框
   画进目标帧 t+1 的相机（``nusc_data.render_assoc_layout``）。它与主通道走**同一个** detail 栈：第一层 3×3 卷积拆成
   ``detail[0](主通道) + layout_assoc(关联通道)``（数学上等于一个 6 通道输入的卷积，两组框在 token 内子格分辨率上直接可比），
   ``layout_assoc`` 零初始化 → 初始输出与不带它的 ckpt **逐位相同**（主通道那一项原样计算，再加精确的 0），可从旧 layout 权重热启动。
   只支持 replace_main 的 adapter（本文件的默认）。不改 date 仓库的 ``CamLayoutAdapter``。

输入/输出都是**标准化后**的特征（每层每通道 (x-μ)/σ，stats 由调用方管理）：``feats: list[4] of [B,T,6,C,h_l,w_l]``，
输出同形状；位置 t 的输出是对帧 t+1 的预测（V-JEPA2-AC：无 target query）。
"""
import importlib.util
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# 避免训练/评测中途别人改了活代码；本地默认读活目录。
import os
_JEPA = str(Path(__file__).parent)


def _load(name):
    spec = importlib.util.spec_from_file_location(f'_jepa_{name}', f'{_JEPA}/{name}.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_acm = _load('ac_modules')
_lay = _load('layout_adapter')
ACBlock, PatchEmbed = _acm.ACBlock, _acm.PatchEmbed
CamLayoutAdapter = _lay.CamLayoutAdapter

LEVEL_GRIDS = ((116, 200), (58, 100), (29, 50), (15, 25))  # 1.0×（928×1600 输入）的 P3..P6
LEVEL_POOLS = (8, 4, 2, 1)
LEVEL_STRIDES = (8, 16, 32, 64)  # 各层相对原图的 stride（px / 格）


def pad_to_tokens(x, pool):
    """[..., h, w] → 右/下补零到 pool 整数倍 [..., ceil(h/p)*p, ceil(w/p)*p]。"""
    h, w = x.shape[-2:]
    Ph, Pw = math.ceil(h / pool) * pool, math.ceil(w / pool) * pool
    return F.pad(x, (0, Pw - w, 0, Ph - h)) if (Ph, Pw) != (h, w) else x


def crop_from_tokens(y, grid):
    """pad_to_tokens 的逆：[..., Ph, Pw] → [..., h, w]（取左上角）。"""
    h, w = grid
    return y[..., :h, :w]


def warp_feat(x, flow):
    """反向 warp：out(i,j) = x(i + flow_y, j + flow_x)（双线性，边界复制）。x [M,C,h,w]；flow [M,2,h,w]（x 在前，单位 = 格）。
    flow = 0 时采样点恰在格心，恒等（fp32 舍入误差 ≤ 1e-4）。"""
    M, C, h, w = x.shape
    ys = (torch.arange(h, device=x.device, dtype=torch.float32) + 0.5).view(1, h, 1)
    xs = (torch.arange(w, device=x.device, dtype=torch.float32) + 0.5).view(1, 1, w)
    gx = (xs + flow[:, 0].float()) / w * 2 - 1
    gy = (ys + flow[:, 1].float()) / h * 2 - 1
    return F.grid_sample(x.float(), torch.stack([gx, gy], -1), mode='bilinear', padding_mode='border', align_corners=False)


class PyramidPredictor(nn.Module):
    def __init__(
        self,
        level_grids=LEVEL_GRIDS,
        level_pools=LEVEL_POOLS,
        n_cams=6,
        in_chans=256,
        embed_dim=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0,
        action_dim=3,
        enable_layout=True,
        layout_detail=None,
        residual=False,
        warp_levels=(),
        flow_px=32.0,
        assoc_layout=False,
        init_std=0.02,
        use_activation_checkpointing=False,
        level_strides=None,
    ):
        super().__init__()
        self.level_grids = [tuple(g) for g in level_grids]
        self.level_pools = [int(p) for p in level_pools]
        self.L = len(self.level_grids)
        self.n_cams, self.D = n_cams, embed_dim
        self.channels = (in_chans,) * self.L if isinstance(in_chans, int) else tuple(in_chans)
        assert len(self.channels) == self.L
        self.residual, self.init_std = residual, init_std
        self.warp_levels = tuple(sorted(int(l) for l in warp_levels))
        assert all(0 <= l < self.L for l in self.warp_levels), self.warp_levels
        self.flow_px = float(flow_px)
        self.level_strides = tuple(LEVEL_STRIDES if level_strides is None else level_strides)
        assert len(self.level_strides) == self.L and all(s > 0 for s in self.level_strides)
        self.use_activation_checkpointing = use_activation_checkpointing
        # 每层补零后的网格与 token 网格
        self.pad_grids, self.tok_grids = [], []
        for (h, w), p in zip(self.level_grids, self.level_pools):
            Hp, Wp = math.ceil(h / p), math.ceil(w / p)
            self.pad_grids.append((Hp * p, Wp * p))
            self.tok_grids.append((Hp, Wp))
        assert len(set(self.tok_grids)) == 1, f'四层 token 网格必须一致: {self.tok_grids}'
        self.Hp, self.Wp = self.tok_grids[0]
        self.n_tok = self.Hp * self.Wp  # 每 (相机, 层) 的 token 数
        self.tokens_per_frame = n_cams * self.L * self.n_tok

        self.patch_embed = nn.ModuleList([PatchEmbed(patch_size=p, in_chans=c, embed_dim=embed_dim) for p,c in zip(self.level_pools,self.channels)])
        self.cam_embed = nn.Parameter(torch.zeros(n_cams, 1, embed_dim))
        self.level_embed = nn.Parameter(torch.zeros(self.L, 1, embed_dim))
        self.action_encoder = nn.Linear(action_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [ACBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None, drop=0.0,
                     act_layer=nn.GELU, wide_silu=True, attn_drop=0.0, drop_path=0.0,
                     norm_layer=lambda d: nn.LayerNorm(d, eps=1e-6), ada_cond_dim=embed_dim)
             for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.proj = nn.ModuleList([nn.Linear(embed_dim, c * p * p) for p,c in zip(self.level_pools,self.channels)])
        self.flow_proj = nn.ModuleDict({str(l): nn.Linear(embed_dim, 2 * self.level_pools[l] ** 2) for l in self.warp_levels})

        self.apply(self._init_weights)
        for i, blk in enumerate(self.blocks):
            blk.attn.proj.weight.data.div_(math.sqrt(2.0 * (i + 1)))
            blk.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (i + 1)))
        nn.init.trunc_normal_(self.cam_embed, std=init_std)
        nn.init.trunc_normal_(self.level_embed, std=init_std)
        if residual:
            for pj in self.proj:
                nn.init.zeros_(pj.weight)
                nn.init.zeros_(pj.bias)
        for l in self.warp_levels:  # 流场与 Δ 都零初始化 → 初始输出 = copy-last
            for pj in (self.flow_proj[str(l)], self.proj[l]):
                nn.init.zeros_(pj.weight)
                nn.init.zeros_(pj.bias)
        # layout adapter 必须在 apply(_init_weights) 之后构造（zero-init 不能被重新随机化）
        self.layout_adapter = None
        if enable_layout:
            self.layout_adapter = CamLayoutAdapter(
                (self.Hp, self.Wp), embed_dim, 3,
                detail=layout_detail or dict(hidden=64, dilations=(1, 2, 4), replace_main=True),
                token_pool=self.level_pools[0])
        # 关联通道（第 7 条）：与 detail 第一层同形状的卷积，输入 = 关联通道 pixel_unshuffle 后的 3·p² 通道；零初始化，构造在 adapter 之后
        self.layout_assoc = None
        if assoc_layout:
            ad = self.layout_adapter
            assert ad is not None and ad.replace_main, 'assoc_layout 需要 enable_layout 且 adapter 为 replace_main'
            c0 = ad.detail[0]
            with torch.random.fork_rng(devices=[]):  # 不消耗全局 CPU RNG：之后 DataLoader 的 worker 种子（相位采样）与不带关联通道的对照臂相同
                self.layout_assoc = nn.Conv2d(3 * ad.token_pool ** 2, c0.out_channels, kernel_size=c0.kernel_size, padding=c0.padding,
                                              dilation=c0.dilation)
            nn.init.zeros_(self.layout_assoc.weight)
            nn.init.zeros_(self.layout_assoc.bias)

        hh, ww = torch.meshgrid(torch.arange(self.Hp), torch.arange(self.Wp), indexing='ij')
        # 同一帧内 token 顺序: [cam][level][token]；RoPE 的 (h,w) 在四层、六相机间重复同一套 12×20 坐标
        self.register_buffer('frame_pos_h', hh.flatten().float().repeat(n_cams * self.L), persistent=False)
        self.register_buffer('frame_pos_w', ww.flatten().float().repeat(n_cams * self.L), persistent=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _assoc_layout_tokens(self, layout):
        """layout [B,T,N,6,H,W] → [B, T*N, n_tok, D]。主通道逐步复刻 ``CamLayoutAdapter.forward``（replace_main、画布 = cam_grid 不池化）的同一串算子，
        只在 detail 第一层卷积之后加上关联通道的 ``layout_assoc``（零初始化时加的是精确的 0）。"""
        ad = self.layout_adapter
        B, T, N, C, H, W = layout.shape
        p = ad.token_pool
        assert C == 6, f'assoc_layout 需要 6 通道 layout，收到 {C}'
        assert (H, W) == (ad.pooled_grid[0] * p, ad.pooled_grid[1] * p), (H, W, ad.pooled_grid, p)
        x = layout.reshape(B * T * N, C, H, W)
        h = ad.detail[0](F.pixel_unshuffle(x[:, :3].contiguous(), p)) + self.layout_assoc(F.pixel_unshuffle(x[:, 3:].contiguous(), p))
        for m in ad.detail[1:]:
            h = m(h)
        out = ad.zero_proj(h).flatten(2).transpose(1, 2)
        return out.reshape(B, T * N, -1, out.shape[-1])

    def _embed(self, feats, layout, B, T):
        """feats list[L] of [B,T,N,C,h_l,w_l]、layout [B,T,N,3|6,H,W] → token [B, T*tokens_per_frame, D]（forward / forward_step 共用）。"""
        N, D = self.n_cams, self.D
        lay = None
        if self.layout_adapter is not None:
            assert layout is not None
            if self.layout_assoc is None:
                # adapter 输入 [B, T', 3, H, W]：把 (T,N) 合成 T'，输出 [B, T*N, n_tok, D]
                lay = self.layout_adapter(layout.reshape(B, T * N, 3, *layout.shape[-2:]))
            else:
                lay = self._assoc_layout_tokens(layout)
            lay = lay.view(B, T, N, self.n_tok, D)

        toks = []  # list over level: [B,T,N,n_tok,D]
        for l, (x, p) in enumerate(zip(feats, self.level_pools)):
            h, w = self.level_grids[l]
            x = pad_to_tokens(x.reshape(B * T * N, self.channels[l], h, w), p)
            assert tuple(x.shape[-2:]) == self.pad_grids[l], (tuple(x.shape), self.pad_grids[l])
            t = self.patch_embed[l](x).view(B, T, N, self.n_tok, D)
            t = t + self.cam_embed.view(1, 1, N, 1, D) + self.level_embed[l].view(1, 1, 1, 1, D)
            if lay is not None:
                t = t + lay
            toks.append(t)
        x = torch.stack(toks, dim=3)  # [B,T,N,L,n_tok,D]
        return x.reshape(B, T * self.tokens_per_frame, D)

    def _decode(self, x, feats, B, T):
        """transformer 输出 token [B, T*tokens_per_frame, D] → list[L] of [B,T,N,C,h_l,w_l]（feats 只给 residual / warp 层当上一帧，同样 T 帧）。"""
        N, D = self.n_cams, self.D
        x = self.norm(x).view(B, T, N, self.L, self.n_tok, D)

        outs = []
        for l, p in enumerate(self.level_pools):
            h, w = self.level_grids[l]
            y = self.proj[l](x[:, :, :, l])  # [B,T,N,n_tok,C*p*p]
            y = y.reshape(B * T * N, self.Hp, self.Wp, -1).permute(0, 3, 1, 2)
            y = F.pixel_shuffle(y, p) if p > 1 else y
            y = crop_from_tokens(y, (h, w))
            if l in self.warp_levels:
                f = self.flow_proj[str(l)](x[:, :, :, l]).reshape(B * T * N, self.Hp, self.Wp, -1).permute(0, 3, 1, 2)
                f = F.pixel_shuffle(f, p) if p > 1 else f
                f = crop_from_tokens(f.float(), (h, w)) * (self.flow_px / self.level_strides[l])  # 单位：该层的格
                y = y + warp_feat(feats[l].reshape(B * T * N, self.channels[l], h, w), f)
            elif self.residual:
                y = y + feats[l].reshape(B * T * N, self.channels[l], h, w).to(y.dtype)
            outs.append(y.reshape(B, T, N, self.channels[l], h, w))
        return outs

    def forward(self, feats, actions, layout=None, return_kv_cache=False):
        """feats: list[L] of [B,T,N,C,h_l,w_l]；actions [B,T,A]；layout [B,T,N,3,H,W]（float 0..1，N 路相机；assoc_layout 时 6 通道）。
        返回 list[L] of [B,T,N,C,h_l,w_l]。

        ``return_kv_cache=True``：额外返回这次稠密前向顺带算出的逐层 post-RoPE K/V，``(outs, kv_cache)``，``kv_cache`` 是长 depth 的 list，
        每项 ``(k, v)`` 各 ``[B, H, T, tokens_per_frame, head_dim]``，可直接喂给 ``forward_step`` 续接（见其 docstring 的语义说明）。
        不传时与加这个参数之前逐位相同。"""
        B, T = actions.shape[:2]
        cond = self.action_encoder(actions)  # [B,T,D]
        x = self._embed(feats, layout, B, T)

        pos_d = torch.arange(T, device=x.device, dtype=torch.float32).repeat_interleave(self.tokens_per_frame)
        pos_h, pos_w = self.frame_pos_h.repeat(T), self.frame_pos_w.repeat(T)
        block_causal = (T, self.tokens_per_frame)
        kv_cache = [] if return_kv_cache else None
        for blk in self.blocks:
            if return_kv_cache:
                x, k, v = blk(x, pos_d, pos_h, pos_w, attn_mask=None, block_causal=block_causal, cond=cond, return_kv=True)
                H = k.shape[1]
                kv_cache.append((k.view(B, H, T, self.tokens_per_frame, -1), v.view(B, H, T, self.tokens_per_frame, -1)))
            elif self.use_activation_checkpointing and self.training:
                x = checkpoint(blk, x, pos_d, pos_h, pos_w, None, block_causal, cond, use_reentrant=False)
            else:
                x = blk(x, pos_d, pos_h, pos_w, attn_mask=None, block_causal=block_causal, cond=cond)
        outs = self._decode(x, feats, B, T)
        return (outs, kv_cache) if return_kv_cache else outs

    def forward_step(self, feats, actions, layout, pos, kv_cache):
        """单帧增量前向（KV cache）：吃一帧新特征，对 ``kv_cache`` 里的历史帧做注意力，返回这一帧的预测（= 下一帧）和推进后的缓存。

        feats list[L] of [B,1,N,C,h_l,w_l]；actions [B,1,A]；layout [B,1,N,3|6,H,W]；pos：新帧的绝对 RoPE 时间位置（标量）——与缓存里各帧
        当初被赋予的位置同一套（RoPE 只看相对位置，所以不必从 0 起，但同一条缓存里必须连续递增）；kv_cache：``None``（空缓存）或
        ``forward(..., return_kv_cache=True)`` / 上一次 ``forward_step`` 返回的缓存。返回 ``(outs, new_kv_cache)``，outs 形状 [B,1,N,C,h_l,w_l]，
        new_kv_cache 每层在时间维多一帧；**淘汰旧帧由调用方做**（``trim_kv_cache``）。

        **与稠密滑窗的关系（调用方必须知道）**：缓存里每一帧的 K/V 是它**入缓存那一刻**算的——当时它的 query 看见的是入缓存时的全部历史。
        稠密滑窗每步丢掉最旧帧后会把窗口里所有帧**按新窗口重算**，所以最旧帧被淘汰之后，两者就不再等价（缓存里的帧"多看了"已淘汰的更早历史）。
        缓存没淘汰过时（窗口还在增长）二者只差浮点误差（test_pyramid_kv.py 检查 2）。ckpt 若是按稠密滑窗训练的（本基准的 rollout 臂），淘汰之后的
        偏差只能用下游指标实测；主线的 ``sliding_xl`` 训练口径才是这个推理方式的无偏对应。"""
        B, T = actions.shape[:2]
        assert T == 1 and all(f.shape[1] == 1 for f in feats), f'forward_step 只吃单帧，收到 T={T}'
        tpf = self.tokens_per_frame
        cond = self.action_encoder(actions)  # [B,1,D]
        x = self._embed(feats, layout, B, 1)
        pos_d = torch.full((tpf,), float(pos), device=x.device, dtype=torch.float32)
        new_cache = []
        for li, blk in enumerate(self.blocks):
            lc = kv_cache[li] if kv_cache is not None else None
            x, k, v = blk(x, pos_d, self.frame_pos_h, self.frame_pos_w, cond=cond, return_kv=True, kv_cache=lc)
            H = k.shape[1]
            k, v = k.view(B, H, 1, tpf, -1), v.view(B, H, 1, tpf, -1)
            new_cache.append((k, v) if lc is None else (torch.cat([lc[0].to(k.dtype), k], dim=2), torch.cat([lc[1].to(v.dtype), v], dim=2)))
        return self._decode(x, feats, B, 1), new_cache


def trim_kv_cache(kv_cache, keep):
    """只留最近 ``keep`` 帧的缓存（时间维是 dim 2）；``keep`` ≥ 现有帧数时原样返回。"""
    if kv_cache is None or kv_cache[0][0].shape[2] <= keep:
        return kv_cache
    return [(k[:, :, -keep:], v[:, :, -keep:]) for k, v in kv_cache]


def predictor_kwargs_from_args(a):
    """train_pred 的命令行参数 → PyramidPredictor 构造参数（评测加载 ckpt 时重建网络用）。"""
    kw = dict(level_pools=list(a['pools']), depth=a['depth'], enable_layout=not a['no_layout'],
              residual=bool(a.get('residual', False)), warp_levels=tuple(a.get('warp_levels') or ()))
    if a.get('assoc_layout'):
        kw['assoc_layout'] = True
    return kw


def uses_assoc_layout(ck):
    """ckpt（``load_predictor`` 返回的 dict）是否要 6 通道 layout（关联通道）。评测侧据此调 ``nusc_data.render_layout_cond(..., assoc=...)``。"""
    kw = ck.get('model_kwargs') or predictor_kwargs_from_args(ck['args'])
    return bool(kw.get('assoc_layout', False))


def sliding_rollout(predictor, feats, actions, layout, L, F):
    """全 horizon 滑动窗口 rollout（主线 ``JEPAWorldModel._rollout_sliding``，论文 AD-E2E-JEPA 附录 A.2；标准化空间）。

    ``feats``  list[4] of [B,T,N,C,h,w]，**只用前 L 帧作真实上下文**（后面的真实帧只在 loss 里当目标，从不进上下文）；
    ``actions`` [B,≥L+F-1,A]、``layout`` [B,≥L+F-1,N,3,H,W] 或 None：位置 t 的条件 = 帧 t→t+1 的动作 / 目标帧 t+1 的 layout，随窗口右移。
    - **窗口 0**：真实帧 0..L-1 一次稠密前向 → ``z_tf``（位置 p ↔ 帧 p+1，逐位置监督）。
    - **第 1 步**：复用窗口 0 最后一个位置的输出（帧 L），不额外前向。
    - **第 j≥2 步**：窗口 = 拼接序列 [真实 0..L-1, 预测 L..L+j-2] 的**最后 L 帧**（丢最旧、只追加最新预测，长度恒为 L），
      **喂给 predictor 前对整个上下文 detach**（TBPTT：各步计算图独立，梯度只流过这一步自己的前向），取最后位置 = 帧 L+j-1 的预测。
      与主线的差别：主线要求 F≤L（窗口里至少留一帧真实帧）；这里取"拼接序列的最后 L 帧"，F>L 时后几步的窗口全是预测帧
      （= 闭环部署的纯自回归口径），不需要这个限制。
    返回 (z_tf, z_ar)：z_tf list[4] of [B,L,...]；z_ar list[4] of [B,F-1,...]（第 2..F 步，z_ar[:, i] ↔ 帧 L+1+i；F=1 时 0 帧）。
    预测拼回上下文时转回 feats 的 dtype（autocast 下 residual 层的输出是 bf16）。"""
    assert F >= 1 and actions.shape[1] >= L + F - 1, (F, L, tuple(actions.shape))
    lay = lambda a, b: layout[:, a:b] if layout is not None else None
    z_tf = predictor([f[:, :L] for f in feats], actions[:, :L], lay(0, L))
    seq = [torch.cat([f[:, :L], z[:, L - 1:L].to(f.dtype)], 1) for f, z in zip(feats, z_tf)]  # 真实 0..L-1 + 预测帧 L
    ar = []
    for j in range(2, F + 1):
        ctx = [s[:, j - 1:].detach() for s in seq]  # 最后 L 帧
        assert ctx[0].shape[1] == L, (ctx[0].shape, L, j)
        z = predictor(ctx, actions[:, j - 1:j - 1 + L], lay(j - 1, j - 1 + L))
        new = [x[:, -1:].to(s.dtype) for x, s in zip(z, seq)]
        ar.append(new)
        seq = [torch.cat([s, n], 1) for s, n in zip(seq, new)]
    z_ar = [torch.cat([a[l] for a in ar], 1) if ar else z_tf[l][:, :0].to(feats[l].dtype) for l in range(len(feats))]
    return z_tf, z_ar


def load_predictor(path, device='cuda'):
    """读 train_pred.py 存的 ckpt（``model`` / ``args`` / 可选 ``model_kwargs``）→ (eval 模式的 PyramidPredictor, ckpt dict)。
    ``model_kwargs`` 缺省时从 ``args`` 还原（旧 ckpt 没有 residual 这一项，按 False）。"""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    kw = ck.get('model_kwargs') or predictor_kwargs_from_args(ck['args'])
    m = PyramidPredictor(**kw)
    m.load_state_dict(ck['model'], strict=True)
    return m.to(device).eval(), ck


if __name__ == '__main__':
    # 单测（因果性、layout zero-init 不扰动、1.0× 网格/补零-裁剪精确逆、层间 token 几何对齐）见 test_pyramid_predictor.py
    import subprocess, sys
    sys.exit(subprocess.call([sys.executable, __file__.replace('pyramid_predictor.py', 'test_pyramid_predictor.py')]))
