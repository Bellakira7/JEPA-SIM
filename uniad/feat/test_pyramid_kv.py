"""PyramidPredictor 的 KV cache（forward_step / return_kv_cache / trim_kv_cache）单测。
用法：CUDA_VISIBLE_DEVICES=7 ../venv/bin/python test_pyramid_kv.py

1. return_kv_cache=True 的输出与默认 forward 逐位相同；缓存是 depth 项 (k, v)，形状 [B,H,T,tokens_per_frame,head_dim]。
2. 缓存没淘汰过：逐帧 forward_step（空缓存起）、稠密前缀 + 逐帧续接，与一次稠密前向（T 帧）在 fp32 下只差浮点误差。
3. 缓存 + 淘汰（trim 到 W-1）与"逐层有界滑窗注意力 SWA(W)"的稠密前向（主线 ``build_action_block_causal_attention_mask(local_window_time=W)``）
   逐位置只差浮点误差——证明实现正确；它**不等于**稠密滑窗（每步按新窗口重算整窗，见 forward_step 的 docstring），差多少是第 4 项报告的。
4. （报告项，不判对错）淘汰后 forward_step 与"稠密滑窗（丢最旧帧、整窗重算）取最后位置"的差：随机权重下只量级参考，真 ckpt 的偏差看 kv_bench.py / A 表。
5. forward_step 拒绝多帧输入；trim_kv_cache 的语义（keep ≥ 现有帧数时原样返回，否则只留最近 keep 帧）。
6. fp16 权重（model.half()）：forward_step 链全程有限，与 fp32 链的最大差在合理量级（随机权重，宽松界）。

随机初始化的 residual / flow / layout 投影层是零，输出恒为 copy-last，测试会空转，所以先把零层随机化（check_nontrivial 核对输出确实偏离输入）。
n_cams=2（每帧 3000 token）只为让第 3 项的稠密 mask（N×N bool）装得下；注意力、RoPE、缓存逻辑与 6 相机相同。
"""
import importlib.util
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyramid_predictor as pp
from pyramid_predictor import LEVEL_GRIDS, PyramidPredictor, trim_kv_cache

fails = []


def check(name, cond, msg=''):
    print(f'[{"PASS" if cond else "FAIL"}] {name} {msg}')
    if not cond:
        fails.append(name)


def mk(depth=2, n_cams=2, **kw):
    m = PyramidPredictor(depth=depth, n_cams=n_cams, residual=True, warp_levels=(0, 1), **kw)
    for p in m.parameters():
        if float(p.detach().abs().max()) == 0:
            p.data.normal_(0, 0.02)
    return m.cuda().eval()


def inputs(T, n_cams=2, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    feats = [torch.randn(1, T, n_cams, 256, *grid, device='cuda', generator=g) for grid in LEVEL_GRIDS]
    act = torch.randn(1, T, 3, device='cuda', generator=g)
    lay = torch.rand(1, T, n_cams, 3, 120, 200, device='cuda', generator=g)
    return feats, act, lay


def frame(feats, act, lay, t):
    return [f[:, t:t + 1] for f in feats], act[:, t:t + 1], lay[:, t:t + 1]


def mad(a, b):
    return max(float((x.float() - y.float()).abs().max()) for x, y in zip(a, b))


def cat_t(outs):
    """[(list[L] of [1,1,...]), ...] → list[L] of [1,T,...]"""
    return [torch.cat([o[l] for o in outs], 1) for l in range(len(outs[0]))]


@torch.no_grad()
def swa_dense(m, feats, act, lay, W):
    """逐层有界滑窗注意力（SWA(W)）的稠密参照：与 forward 同一条 embed / blocks / decode，只把块因果 mask 换成带 local_window_time 的。"""
    spec = importlib.util.spec_from_file_location('_acm_t', f'{pp._JEPA}/ac_modules.py')
    acm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(acm)
    B, T = act.shape[:2]
    tpf = m.tokens_per_frame
    mask = acm.build_action_block_causal_attention_mask(T, tpf, W).cuda()
    cond = m.action_encoder(act)
    x = m._embed(feats, lay, B, T)
    pos_d = torch.arange(T, device='cuda', dtype=torch.float32).repeat_interleave(tpf)
    pos_h, pos_w = m.frame_pos_h.repeat(T), m.frame_pos_w.repeat(T)
    for blk in m.blocks:
        x = blk(x, pos_d, pos_h, pos_w, attn_mask=mask, block_causal=None, cond=cond)
    return m._decode(x, feats, B, T)


@torch.no_grad()
def chain(m, feats, act, lay, keep=None, cache=None, t0=0, dtype=None):
    """从 cache 起逐帧 forward_step 到最后一帧；keep=窗口淘汰（每步前 trim 到 keep 帧）。返回 (各帧输出, 缓存)。"""
    T = act.shape[1]
    outs = []
    for t in range(t0, T):
        f, a, l = frame(feats, act, lay, t)
        if dtype is not None:
            f, a, l = [x.to(dtype) for x in f], a.to(dtype), l.to(dtype)
        if keep is not None:
            cache = trim_kv_cache(cache, keep)
        o, cache = m.forward_step(f, a, l, pos=t, kv_cache=cache)
        outs.append(o)
    return outs, cache


def main():
    torch.manual_seed(0)
    m = mk()
    T = 4
    feats, act, lay = inputs(T)
    with torch.no_grad():
        dense = m(feats, act, lay)
    nz = max(float((o - f).abs().max()) for o, f in zip(dense, feats))
    check('0 non-trivial', nz > 1.0, f'输出偏离 copy-last {nz:.3g}（零层已随机化）')

    # 1. return_kv_cache
    with torch.no_grad():
        o1, kv = m(feats, act, lay, return_kv_cache=True)
    H = m.blocks[0].attn.num_heads
    hd = m.D // H
    check('1 return_kv_cache == forward', all(torch.equal(a, b) for a, b in zip(o1, dense)) and len(kv) == len(m.blocks)
          and all(k.shape == (1, H, T, m.tokens_per_frame, hd) == v.shape for k, v in kv), f'depth={len(kv)} k={tuple(kv[0][0].shape)}')

    # 2. 没淘汰：逐帧链 / 稠密前缀 + 续接 vs 稠密
    outs, _ = chain(m, feats, act, lay)
    d_chain = mad(cat_t(outs), dense)
    k2 = 2
    with torch.no_grad():
        pre, kv2 = m([f[:, :k2] for f in feats], act[:, :k2], lay[:, :k2], return_kv_cache=True)
    outs2, _ = chain(m, feats, act, lay, cache=kv2, t0=k2)
    d_mix = mad(cat_t([[p[:, t:t + 1] for p in pre] for t in range(k2)] + outs2), dense)
    tol = 2e-3
    check('2 no-eviction chain == dense', d_chain < tol and d_mix < tol, f'逐帧链 max|Δ|={d_chain:.3g}，稠密前缀+续接 max|Δ|={d_mix:.3g}（tol {tol}）')

    # 3. 淘汰 == SWA(W)；4. 报告：淘汰 vs 稠密滑窗
    W, T2 = 3, 6
    feats6, act6, lay6 = inputs(T2, seed=1)
    ref = swa_dense(m, feats6, act6, lay6, W)
    outs, _ = chain(m, feats6, act6, lay6, keep=W - 1)
    d_swa = mad(cat_t(outs), ref)
    check('3 evict(W-1) cache == SWA(W) dense', d_swa < tol, f'W={W} T={T2} max|Δ|={d_swa:.3g}')
    # 稠密滑窗：第 t 帧的预测 = 窗口 [t-W+1, t] 整窗重算后的最后位置
    last = []
    with torch.no_grad():
        for t in range(T2):
            lo = max(0, t - W + 1)
            o = m([f[:, lo:t + 1] for f in feats6], act6[:, lo:t + 1], lay6[:, lo:t + 1])
            last.append([x[:, -1:] for x in o])
    per_t = [mad(a, b) for a, b in zip(outs, last)]
    print(f'[INFO] 4 淘汰缓存 vs 稠密滑窗（逐帧 max|Δ|，随机权重只看量级）：' + ' '.join(f't{t}={d:.3g}' for t, d in enumerate(per_t))
          + f'；窗口未满的前 {W} 帧应 ≈0')
    check('4 窗口未满时缓存 == 稠密滑窗', max(per_t[:W]) < tol, f'前 {W} 帧 max {max(per_t[:W]):.3g}；淘汰后 max {max(per_t[W:]):.3g}')

    # 5. 输入校验 / trim 语义
    try:
        m.forward_step([f[:, :2] for f in feats], act[:, :2], lay[:, :2], pos=0, kv_cache=None)
        check('5a forward_step rejects T=2', False)
    except AssertionError:
        check('5a forward_step rejects T=2', True)
    c3 = trim_kv_cache(kv, 3)
    c9 = trim_kv_cache(kv, 9)
    check('5b trim_kv_cache', c9 is kv and c3[0][0].shape[2] == 3 and torch.equal(c3[0][0], kv[0][0][:, :, -3:]) and trim_kv_cache(None, 3) is None)

    # 6. fp16 权重
    m16 = mk()
    m16.load_state_dict(m.state_dict())
    m16 = m16.half()
    outs16, _ = chain(m16, feats6, act6, lay6, keep=W - 1, dtype=torch.float16)
    fin = all(bool(torch.isfinite(x).all()) for o in outs16 for x in o)
    d16 = mad(cat_t(outs16), cat_t(outs))
    scale = max(float(x.abs().max()) for x in cat_t(outs))
    check('6 fp16 chain finite & close to fp32', fin and d16 < 5e-2 * max(1.0, scale), f'finite={fin} max|Δ|={d16:.3g}（输出量级 {scale:.3g}）')

    print('ALL PASSED' if not fails else f'FAILED: {fails}')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
