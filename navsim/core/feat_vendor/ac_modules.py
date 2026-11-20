"""V-JEPA 2-AC predictor 基础模块（移植自官方源码, 不自创结构）。

出处: facebookresearch/vjepa2 ``src/models/utils/modules.py`` + ``src/models/utils/patch_embed.py``


多步(单次前向吃整个 rollout 窗口)变体应直接复用官方 ``build_action_block_causal_attention_mask``
与 ``ACRoPEAttention``("动作 token 只转时间维、frame token 转 d/h/w 三维"), 而非重新推导。

与官方源码的差别(仅为适配 7 路异分辨率相机, 数学不变):
- ``ACRoPEAttention.forward`` 不再从"均匀 T×H×W 网格 + 前置 action_tokens"的 token 下标反推
  (d,h,w) 位置, 而是由 predictor 显式传入逐 token 的 ``pos_d/pos_h/pos_w``。条件 token 的
  ``pos_h = pos_w = 0`` ⇒ 旋转角为 0 ⇒ 恰好等价于官方"动作 token 仅施加时间维 RoPE"的写法;
  frame token 的 h/w 仍按官方 ``h * grid_size / H`` 归一到公共网格。
- ``build_action_block_causal_attention_mask`` 的每帧 token 数由 ``add_tokens + H*W`` 泛化为
  ``tokens_per_frame``(多相机 token 之和 + 条件 token 数), 块内双向、跨块单向的结构逐字不变。
- RoPE 采用官方注释里给出的修正版展开(``repeat_interleave``); 官方保留 bug 版仅为兼容其预训练权重,
  本 predictor 从零训练, 无兼容负担。
- RoPE 的 sin/cos 角度在 fp32 下计算再 cast 回 x.dtype(官方直接用 x.dtype; 多相机公共网格最宽 60,
  bf16 下 60 rad 的分辩率约 0.25 rad, 见 ``rotate_queries_or_keys`` 内注释)。

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath


def build_action_block_causal_attention_mask(
    T: int, tokens_per_frame: int, local_window_time: int | None = None
) -> torch.Tensor:
    """块因果 mask: 帧 t1 的全部 token(条件 + frame)可见帧 t2<=t1 的全部 token; 块内双向。

    官方: ``N_T = add_tokens + H*W``; 此处 ``N_T = tokens_per_frame``。True = 参与注意力
    (与 ``F.scaled_dot_product_attention`` 的 bool ``attn_mask`` 语义一致)。

    ``local_window_time``: None(默认, 旧行为逐位不变)= 无界历史(纯因果, 看得到 0..t1 全部); 给定正整数
    W 时额外加一条"只看最近 W 帧"的滑窗约束(SWA), 即 ``t2 in [t1-W+1, t1]``。只用于离线核对 KV cache 的
    环形淘汰是否等价于有界滑窗注意力(``jepa_test_kv_cache.py``, 见 jepa推理加速.md §3.3); 生产 forward 不传。
    """
    N_T = tokens_per_frame
    N = T * N_T
    mask = torch.zeros(N, N).bool()
    mask_block = torch.ones(N_T, N_T).bool()
    window = T if local_window_time is None else int(local_window_time)

    for t1 in range(T):
        for t2 in range(max(0, t1 - window + 1), t1 + 1):
            mask[t1 * N_T : (t1 + 1) * N_T, t2 * N_T : (t2 + 1) * N_T] = mask_block

    return mask



def rotate_queries_or_keys(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """对 x[..., N, D] 的最后一维做 RoPE 旋转, pos 为每个 token 的标量位置([N] 或可广播到 x[..., N])。"""
    D = x.size(-1)
    assert D % 2 == 0, 'Embedding dimension must be a multiple of 2 for block matrix rotation'

    # -- compute angle for each position
    # 与官方唯一的数值差别: 角度在 fp32 下计算再 cast 回 x.dtype。官方用 x.dtype(bf16 autocast 下即 bf16),
    # 官方网格仅 16×16 尚可; 本任务 RoPE 公共网格宽达 60, pos·omega≈60 rad 在 bf16 下分辩率约 0.25 rad,
    # 会破坏相对位置性质。结构/公式不变。
    omega = torch.arange(D // 2, dtype=torch.float32, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    freq = torch.einsum('..., f -> ... f', pos.to(torch.float32), omega)  # (..., N, D/2), outer product

    # -- build rotation matrix and apply (官方注释给出的修正版: 频率按相邻对 interleave)
    emb_sin = freq.sin().repeat_interleave(2, dim=-1).to(x.dtype)  # (..., N, D)
    emb_cos = freq.cos().repeat_interleave(2, dim=-1).to(x.dtype)  # (..., N, D)

    # --
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.0, wide_silu=True
    ):
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden_features = hidden_features = hidden_features or in_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        # 官方命名保持不变(fc2 为 gate 分支, fc3 为输出投影); ``_rescale_blocks`` 对 fc2 缩放与官方一致
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)


class ACRoPEAttention(nn.Module):
    """Action-conditioned RoPE attention(官方 ``ACRoPEAttention``, 位置改为显式传入)。

    head_dim 按官方切成 d/h/w 三段(各 ``2*((head_dim//3)//2)`` 维)分别旋转, 余下维度不旋转。
    条件 token(action/state/navi)由 predictor 传入 ``pos_h = pos_w = 0``, 只有 d(时间)维实际被旋转。
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        # --
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.is_causal = is_causal

    def _rope_qk(self, q, k, pos_d, pos_h, pos_w):
        """对 q/k 施加 d/h/w 三维 RoPE(整段前向与单帧增量前向共用, 保证两条路的旋转逐位一致)。"""
        s = 0
        # Rotate depth (time)
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=pos_d)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=pos_d)
        s += self.d_dim
        # Rotate height dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=pos_h)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=pos_h)
        s += self.h_dim
        # Rotate width dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=pos_w)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=pos_w)
        s += self.w_dim

        # Combine rotated dimension
        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)
        return q, k

    def forward(self, x, pos_d, pos_h, pos_w, attn_mask=None, block_causal=None, return_kv=False, kv_cache=None):
        """
        Args:
            x: [B, N, C]; pos_*: [N] 逐 token 位置(frame token 的 h/w 已归一到公共网格, 条件 token 为 0)。
            attn_mask: 官方路径, 稠密 bool mask [N, N](``build_action_block_causal_attention_mask``)。
            block_causal: ``(T, tokens_per_frame)``; 给定时用【逐帧分块】实现同一个块因果 mask:
                帧 t 的 query 只对帧 ≤t 的 key/value 做无 mask 的 SDPA。与稠密 mask 数学上逐元素等价
                (块内双向、跨块单向), 但可走 flash 内核: 实测 27.8k token 时 fwd+bwd 1.62 s → 0.35 s,
                且不用物化 N×N mask。二者同时给定时优先 block_causal。
            return_kv: True 时额外返回本次算出的 post-RoPE ``(k, v)``(各 ``[B, H, N, head_dim]``), 给 KV cache 建缓存用。
            kv_cache: ``(k_old, v_old)`` 各 ``[B, H, n, N_T, head_dim]``(post-RoPE, 更早 n 帧); 给定时是**单帧增量前向**:
                ``x`` 是新的一帧(``N = N_T``), 其 query 对 ``n`` 帧缓存 + 自己做无 mask SDPA(块内双向、对旧帧全可见,
                与块因果 mask 下该帧的一行等价), 并返回新帧的 ``(k, v)`` 由调用方推进缓存。此时忽略 attn_mask/block_causal。
        Returns:
            ``x``; ``return_kv`` 或 ``kv_cache`` 非空时是 ``(x, k, v)``。默认(两者皆无)与旧版逐位相同。
        """
        B, N, C = x.size()

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        q, k = self._rope_qk(q, k, pos_d, pos_h, pos_w)

        dropout_p = self.proj_drop_prob if self.training else 0.0
        if kv_cache is not None:
            k_old, v_old = kv_cache
            K = torch.cat([k_old.flatten(2, 3).to(k.dtype), k], dim=2)  # [B, H, (n+1)*N_T, D]
            V = torch.cat([v_old.flatten(2, 3).to(v.dtype), v], dim=2)
            x = F.scaled_dot_product_attention(q, K, V, dropout_p=dropout_p)
        elif block_causal is not None:
            T, N_T = block_causal
            assert T * N_T == N, f'block_causal {block_causal} does not match N={N}'
            outs = []
            for t in range(T):
                qs = slice(t * N_T, (t + 1) * N_T)
                ks = slice(0, (t + 1) * N_T)
                outs.append(F.scaled_dot_product_attention(q[:, :, qs], k[:, :, ks], v[:, :, ks], dropout_p=dropout_p))
            x = torch.cat(outs, dim=2)
        elif attn_mask is not None or self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=self.is_causal, attn_mask=attn_mask
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_kv or kv_cache is not None:
            return x, k, v
        return x


class ACBlock(nn.Module):
    """官方 ``ACBlock``(pre-norm; attn + MLP/SwiGLU 残差, 可选 DropPath)。

    ``ada_cond_dim > 0`` 时加 AdaLN 动作条件(论文 AD-E2E-JEPA / JEPA-WM 的 "AdaLN + RoPE"):
    两处 pre-norm 之后各做一次 ``y·(1+γ)+β``, ``(γ1, β1, γ2, β2) = Linear(SiLU(cond))``。
    **不加 gate**(不是 DiT 的 adaLN-Zero 6 参数形式): JEPA-WM 消融里 AdaLN-zero 在信号最可靠的
    DROID / PushT / Maze 上不如 AdaLN, 论文最终选 AdaLN。初始化不在这里做, 由外层 ``_init_weights``
    统一(trunc_normal); 想要 AdaLN-zero 式零初始化由外层在 ``apply`` 之后显式置零。
    ``ada_cond_dim=0``(默认)不建 ``ada_ln``, state_dict 与旧版逐键相同。
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        ada_cond_dim=0,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ACRoPEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            use_sdpa=use_sdpa,
            is_causal=is_causal,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.ada_ln = (
            nn.Sequential(nn.SiLU(), nn.Linear(int(ada_cond_dim), 4 * dim, bias=True)) if ada_cond_dim > 0 else None
        )

    def forward(
        self, x, pos_d, pos_h, pos_w, attn_mask=None, block_causal=None, cond=None, return_kv=False, kv_cache=None
    ):
        """``cond``: ``[B, T, ada_cond_dim]``, 逐帧一份; ``x`` 是 ``[B, T*N_T, D]``(帧 major 展平),
        同一帧的全部 token(含 7 路相机)共用该帧的 γ/β。调制先在 ``[B,T,·]`` 上算完再广播, 不对 N_T 个 token 重复算。
        None = 不调制。

        ``return_kv`` / ``kv_cache``: 透传给 ``ACRoPEAttention``(见其 docstring); 任一非空时返回 ``(x, k, v)``,
        默认(都不传)与旧版逐位相同。单帧增量时 ``x`` 是 ``[B, N_T, D]``、``cond`` 是 ``[B, 1, ada_cond_dim]``。"""
        if cond is not None:
            assert self.ada_ln is not None, 'cond 非空但 ada_cond_dim=0(没建 ada_ln)'
            B, T = cond.shape[:2]
            assert x.shape[1] % T == 0, f'token 数 {x.shape[1]} 不是帧数 {T} 的整数倍'
            g1, b1, g2, b2 = (m.unsqueeze(2) for m in self.ada_ln(cond).chunk(4, dim=-1))  # 各 [B, T, 1, D]
        y = self.norm1(x)
        if cond is not None:
            y = (y.view(B, T, -1, y.shape[-1]) * (1 + g1) + b1).view_as(y)
        want_kv = return_kv or kv_cache is not None
        y = self.attn(
            y, pos_d, pos_h, pos_w, attn_mask=attn_mask, block_causal=block_causal, return_kv=return_kv,
            kv_cache=kv_cache,
        )
        if want_kv:
            y, k, v = y
        x = x + self.drop_path(y)
        y = self.norm2(x)
        if cond is not None:
            y = (y.view(B, T, -1, y.shape[-1]) * (1 + g2) + b2).view_as(y)
        x = x + self.drop_path(self.mlp(y))
        return (x, k, v) if want_kv else x


class PatchEmbed(nn.Module):
    """官方 ``PatchEmbed``(``src/models/utils/patch_embed.py``): Conv2d(k=s=patch) → [B, N, D]。

    这里作用在 backbone 的特征图而非像素上: patch_size=token_pool(默认 2) 即提案 §六.1 "predictor
    内部做 2×2 pooling" 的实现载体(可学习 pooling, 等价于把 2×2 特征块拼成一个 token 再线性投影);
    patch_size=1 退化为逐 token 线性投影(官方 ``predictor_embed``)。
    """

    def __init__(self, patch_size=2, in_chans=384, embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def _pad_to_stride(hw, stride: int):
    """``(H, W)`` → ``(pad_h, pad_w, H_pad, W_pad)``: 补到 ``stride`` 的整数倍。已整除时全 0。"""
    H, W = int(hw[0]), int(hw[1])
    ph = (-H) % stride
    pw = (-W) % stride
    return ph, pw, H + ph, W + pw


class Projector(nn.Module):
    """压缩投影器: 两层 stride-2 卷积, 把 DINOv3 特征图压成低分辨率、低维的 latent(论文 Eq.7)。

    结构出处与自定部分(重要, 见 ``docs/todo/jepa_three_directions.md`` §3 与本地计划文件):

    - **论文给定的**: 两层卷积、``stride 2×2``、对 history / future 分别逐帧作用、context 与 target
      **共用同一份权重**; 净效果是降维 `D_enc → D_proj`(论文 1024→256, **4×**)。
    - **论文没给、由本实现自定的**: kernel / padding / 激活 / 每层通道数。论文这些只写在图 2 里,

      ``Conv2d(k=3, s=2, p=1) → GELU → Conv2d(k=3, s=2, p=1)``, 通道按 ``out_dim`` 折半递降
      (384 → 192 → 96)。``k=3,s=2,p=1`` 时输出边长 = ``ceil(H/2)``, 两层 = ``ceil(H/4)``。

    **norm_type(官方实现对齐)**: 论文图 2 没画 norm, 但官方代码(github.com/HaoranZhuExplorer/AD-E2E-JEPA,
    ``navsim/agents/jepa_wm_v3/jepa_wm_projector.py``, config ``norm_type: layer``)的 ``ChannelProjector`` 是
    ``Conv → LayerNorm → GELU → Conv → LayerNorm``(LN 沿通道维、带 affine), **输出端有 LN**。这一点对 SIGReg 是承重的:
    SIGReg 的 Epps-Pulley 统计量积分区间只有 t∈[0,3], 输出尺度 σ≫3 后统计量饱和(N=384 时上限≈484)、没有回拉梯度,
    所以只靠 SIGReg 管不住 projector 输出放大; 又因为 ``target_sg`` 截断了 MSE 对 target 的梯度, 预测 loss 也管不住。
    无 norm 版(``norm_type=None``, 默认)在 ``abl_proj_sigreg_128gpu`` 上 iter 8640~8800 ``tgt_chvar_mean``(逐通道方差×D)从 1.07 涨到 3e5 后训崩。
    ``norm_type='layer'`` 让每个 token 的输出逐通道均值 0 / 方差 1(γ=1, β=0 初始), 与 SIGReg 不带 √D 的目标(各向同性
    N(0,I), 即每通道方差 1)在初始就自洽。``None`` 时不建任何 norm 层, 参数键与前向与加这个参数之前**逐位相同**(旧 payload 照常加载)。

    **为什么需要 pad**: ``ac_predictor.py`` 对 token 网格有整除断言, 而现有 7 个相机里 ``0``/``9``(30×40)、
    ``11``(38×60) 在 stride=4 下除不尽。这里在**模块内部**把特征图补到 stride 的整数倍、token 出来后再
    crop 回真实网格(见 :meth:`forward` 的返回值与 :class:`ProjDecoder`), 于是 config 的 ``CAM_GRIDS``
    语义完全不变(仍是 loss 分辨率 / head 网格 / layout 画布), 下游无需知道 pad 的存在。
    pad 用 **replicate**(``F.pad(mode='replicate'``) 而非补 0 —— 补 0 会在边界造出一圈与真实特征无关的
    输入, 而这一圈恰好落在最外侧 token 里。
    """

    def __init__(
        self,
        in_chans: int = 384,
        out_dim: int = 96,
        stride: int = 4,
        act_layer=nn.GELU,
        hidden: int | None = None,
        norm_type: str | None = None,
    ):
        super().__init__()
        assert stride == 4, f'Projector 目前只实现了两层 stride-2(净 stride 4), 收到 stride={stride}'
        assert norm_type in (None, 'layer'), f'norm_type 只支持 None / "layer", 收到 {norm_type!r}'
        hidden = int(hidden) if hidden is not None else max(out_dim, in_chans // 2)
        self.stride = stride
        self.in_chans = in_chans
        self.out_dim = out_dim
        self.norm_type = norm_type
        # ``net`` 的键(net.0 / net.2)不随 norm_type 变; norm 层另放 ``norms``, ``None`` 时不建 → 旧 state_dict 逐键兼容。
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, hidden, kernel_size=3, stride=2, padding=1),
            act_layer(),
            nn.Conv2d(hidden, out_dim, kernel_size=3, stride=2, padding=1),
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden), nn.LayerNorm(out_dim)]) if norm_type == 'layer' else None

    def pad_for(self, hw) -> tuple[int, int, int, int]:
        """给定特征图 ``(H, W)``, 返回 ``(pad_h, pad_w, H_pad, W_pad)``; 供 decoder / 下游裁剪对账。"""
        return _pad_to_stride(hw, self.stride)

    def forward(self, x: torch.Tensor):
        """``[B, C, H, W]`` → ``(tokens [B, N, out_dim], meta)``。

        ``meta = (H, W, pad_h, pad_w, H_pad, W_pad)``, 供 :class:`ProjDecoder` 还原到原始网格。
        token 网格为 ``(H_pad/stride, W_pad/stride)``, 展平顺序是行主序(H 再 W), 与
        :func:`PatchEmbed.forward` / ``MultiCamACPredictor._tokens_to_maps`` 一致 —— 否则 token 与
        layout 特征相加时会错位。
        """
        H, W = int(x.shape[-2]), int(x.shape[-1])
        pad_h, pad_w, h_pad, w_pad = self.pad_for((H, W))
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')  # (left, right, top, bottom)
        if self.norms is None:
            y = self.net(x)  # [B, out_dim, h_pad/4, w_pad/4]
            tokens = y.flatten(2).transpose(1, 2)  # [B, N, out_dim]
        else:
            # Conv → LN → GELU → Conv → LN(官方 ChannelProjector); LN 沿通道维, 逐 token 独立, 所以 replicate pad 的边不受影响。
            conv1, act, conv2 = self.net[0], self.net[1], self.net[2]
            y = conv1(x)
            y = self.norms[0](y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            y = conv2(act(y))
            tokens = self.norms[1](y.flatten(2).transpose(1, 2))  # [B, N, out_dim], 逐 token 通道维归一
        return tokens, (H, W, pad_h, pad_w, h_pad, w_pad)


class ProjDecoder(nn.Module):
    """解压回原始特征空间: ``[B, N, latent]`` → ``[B, C, H, W]``(与 :class:`Projector` 对称)。

    **这是我们相对论文新增的一环**(论文世界模型侧没有 decoder: 它下游只用 latent 空间的最近邻代价,
    见 AD-E2E-JEPA Eq.13)。我们保留了冻结的 A5 检测/规划 head, 而 head 吃的是原始特征空间
    ``[B, C, H_c, W_c]``, 所以必须自己训一个解码器把 latent 解回去, 供 ``head_loss`` 用。

    实现是"线性投影 + 参数无关的 ``pixel_shuffle``": ``Linear(latent → C·s·s)`` 再 ``pixel_shuffle(s)``
    (s = 投影器的净 stride), 与 predictor 现有输出头 ``predictor_proj + _tokens_to_maps`` 的做法同构,
    只是出口网格不同。``in_grid`` 给出 token 网格 ``(H', W')``, ``meta`` 给出原始网格与 pad 量,
    解码后按 pad 裁掉多出来的边, 保证输出与 ``Projector`` 的输入**逐位同形**。
    """

    def __init__(self, latent_dim: int = 96, out_chans: int = 384, stride: int = 4):
        super().__init__()
        self.stride = stride
        self.out_chans = out_chans
        self.latent_dim = latent_dim
        self.proj = nn.Linear(latent_dim, out_chans * stride * stride, bias=True)

    def forward(self, tokens: torch.Tensor, in_grid, meta=None) -> torch.Tensor:
        """
        Args:
            tokens: ``[B, N, latent_dim]``(N = ``in_grid[0] * in_grid[1]``)。
            in_grid: token 网格 ``(H', W')``。
            meta: :class:`Projector.forward` 返回的第二项; 给了就裁掉 pad 边, 返回 ``[B, C, H, W]``。
                不给则返回 pad 后的整张图 ``[B, C, H', W']``(自用时才这么用)。

        Returns:
            ``[B, C, H, W]``。
        """
        B = tokens.shape[0]
        hp, wp = int(in_grid[0]), int(in_grid[1])
        s = self.stride
        y = self.proj(tokens)  # [B, N, C·s·s]
        y = y.reshape(B, hp, wp, self.out_chans, s, s).permute(0, 3, 1, 4, 2, 5)
        y = y.reshape(B, self.out_chans, hp * s, wp * s)  # [B, C, H_pad, W_pad]
        if meta is None:
            return y
        H, W, pad_h, pad_w, h_pad, w_pad = meta
        assert (hp * s, wp * s) == (h_pad, w_pad), (
            f'token 网格 {in_grid} 与 meta 不一致: {(hp * s, wp * s)} vs {(h_pad, w_pad)}'
        )
        if pad_h or pad_w:
            y = y[:, :, :H, :W]
        return y
