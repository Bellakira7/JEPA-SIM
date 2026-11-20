"""PyramidPredictor 单测（1.0× 网格）。用法：CUDA_VISIBLE_DEVICES=2 ../venv/bin/python test_pyramid_predictor.py

1. 网格：四层 token 网格都是 15×25，补零网格 120×200 / 60×100 / 30×50 / 15×25，每帧 9000 token。
2. 补零-裁剪精确逆：crop(pixel_shuffle(pixel_unshuffle(pad(x)))) 与 x 逐位相同，补出来的位置全 0。
3. 层间 token 几何对齐：token (i,j) 在 P3/P4/P5 上覆盖同一块 [64i,64i+64)×[64j,64j+64) px；P6 中心落在同一块内。
4. 因果性：改帧 2 的输入不影响帧 0、1 的输出。
5. layout zero-init：换 layout 不改输出。
6. residual=True 初始化时输出 = 输入（proj 零初始化）。
7. forward 输出形状与 LEVEL_GRIDS 一致；layout 画布 = adapter 的 cam_grid（120×200，进 adapter 时不再池化）。
8. warp_feat：flow=0 恒等；flow_x=+1 格时 out(i,j)=x(i,j+1)（右侧边界复制）；flow_y 同理。
9. warp_levels 初始化：被选层输出 ≈ 上一帧同层特征（流场与 Δ 零初始化 → copy-last，容差 1e-3：grid_sample 坐标舍入），其余层不受影响；梯度能回到 flow_proj；因果性不变。
10. predictor_kwargs_from_args / load_predictor：保存→读回 strict 加载，输出逐位相同（含 warp_levels）。
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyramid_predictor import (LEVEL_GRIDS, LEVEL_POOLS, PyramidPredictor, crop_from_tokens, pad_to_tokens, warp_feat,
                               load_predictor, predictor_kwargs_from_args)
import nusc_data as nd

STRIDES = (8, 16, 32, 64)
fails = []


def check(name, cond, msg=''):
    print(f'[{"PASS" if cond else "FAIL"}] {name} {msg}')
    if not cond:
        fails.append(name)


def main():
    torch.manual_seed(0)
    dev = 'cuda'
    m = PyramidPredictor(depth=2, residual=False).to(dev)
    check('1 grid', (m.Hp, m.Wp) == (15, 25) and m.tokens_per_frame == 9000
          and m.pad_grids == [(120, 200), (60, 100), (30, 50), (15, 25)],
          f'tok={(m.Hp, m.Wp)} tokens/frame={m.tokens_per_frame} pad={m.pad_grids} params(M)={sum(p.numel() for p in m.parameters()) / 1e6:.2f}')

    ok = True
    for (h, w), p in zip(LEVEL_GRIDS, LEVEL_POOLS):
        x = torch.randn(3, 256, h, w, device=dev)
        xp = pad_to_tokens(x, p)
        ok &= tuple(xp.shape[-2:]) == (15 * p, 25 * p)
        ok &= bool((xp[..., h:, :] == 0).all() and (xp[..., :, w:] == 0).all())
        y = F.pixel_unshuffle(xp, p) if p > 1 else xp
        ok &= tuple(y.shape[-2:]) == (15, 25)
        z = crop_from_tokens(F.pixel_shuffle(y, p) if p > 1 else y, (h, w))
        ok &= torch.equal(z, x)
    check('2 pad/crop exact inverse', ok)

    # 名义几何：stride-s 栅格的第 r 格覆盖 [s*r, s*(r+1)) px；token i 覆盖第 [p*i, p*(i+1)) 格
    ok = True
    for i in range(15):
        spans = {(p * i * s, p * (i + 1) * s) for s, p in zip(STRIDES[:3], LEVEL_POOLS[:3])}
        ok &= spans == {(64 * i, 64 * (i + 1))}
        c6 = 64 * i + 32 - 16  # P6 第 i 格 = P5 上 3×3/stride2/pad1 卷积，中心在 P5 第 2i 格中心（32*2i+16）
        ok &= 64 * i <= c6 < 64 * (i + 1)
    check('3 cross-level token alignment', ok)

    B, T, N = 1, 3, 6
    feats = [torch.randn(B, T, N, 256, h, w, device=dev) for h, w in LEVEL_GRIDS]
    act = torch.randn(B, T, 3, device=dev)
    lay = torch.rand(B, T, N, 3, *nd.LAYOUT_HW, device=dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        out = m(feats, act, lay)
        out2 = m(feats, act, torch.zeros_like(lay))
        feats2 = [f.clone() for f in feats]
        for f in feats2:
            f[:, 2] += 1.0
        out3 = m(feats2, act, lay)
    check('7 output shapes', [tuple(o.shape) for o in out] == [(B, T, N, 256, h, w) for h, w in LEVEL_GRIDS])
    d_lay = max((a.float() - b.float()).abs().max().item() for a, b in zip(out, out2))
    check('5 layout zero-init invariance', d_lay == 0, f'max diff {d_lay}')
    d01 = max((a[:, :2].float() - b[:, :2].float()).abs().max().item() for a, b in zip(out, out3))
    d2 = max((a[:, 2].float() - b[:, 2].float()).abs().max().item() for a, b in zip(out, out3))
    check('4 causality', d01 == 0 and d2 > 0, f'diff frames 0..1 = {d01}, frame 2 = {d2:.3f}')

    full = (m.Hp * m.layout_adapter.token_pool, m.Wp * m.layout_adapter.token_pool)
    check('7 layout canvas = adapter cam_grid', full == tuple(nd.LAYOUT_HW), f'{full} vs {nd.LAYOUT_HW}')

    mr = PyramidPredictor(depth=1, residual=True, enable_layout=False).to(dev)
    with torch.no_grad():
        o = mr([f[:, :1] for f in feats], act[:, :1])
    check('6 residual init = identity', all(torch.equal(a, b[:, :1]) for a, b in zip(o, feats)))

    # 8 warp_feat
    x = torch.randn(2, 5, 7, 9, device=dev)
    z = torch.zeros(2, 2, 7, 9, device=dev)
    ok = torch.allclose(warp_feat(x, z), x, atol=1e-5)
    fx = torch.zeros_like(z); fx[:, 0] = 1.0
    o = warp_feat(x, fx)
    ok &= torch.allclose(o[..., :-1], x[..., 1:], atol=1e-5) and torch.allclose(o[..., -1], x[..., -1], atol=1e-5)
    fy = torch.zeros_like(z); fy[:, 1] = 1.0
    o = warp_feat(x, fy)
    ok &= torch.allclose(o[..., :-1, :], x[..., 1:, :], atol=1e-5) and torch.allclose(o[..., -1, :], x[..., -1, :], atol=1e-5)
    check('8 warp_feat identity/shift', bool(ok))

    # 9 warp_levels
    mw = PyramidPredictor(depth=2, warp_levels=(0, 1), residual=False).to(dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        ow = mw(feats, act, lay)
    d = [(a.float() - f.float()).abs().max().item() for a, f in zip(ow, feats)]
    check('9a warp init = copy-last on warp levels', d[0] < 1e-3 and d[1] < 1e-3 and d[2] > 1e-1 and d[3] > 1e-1,
          f'maxdiff per level {[f"{v:.2e}" for v in d]}')
    mw.train()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        ow = mw([f[:, :2] for f in feats], act[:, :2], lay[:, :2])
        loss = sum((o.float() - f[:, 1:2]).abs().mean() for o, f in zip(ow, [f[:, 1:3][:, :2] for f in feats]))
    loss.backward()
    g_flow = mw.flow_proj['0'].weight.grad
    check('9b grad reaches flow_proj', g_flow is not None and g_flow.abs().sum().item() > 0)
    mw.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        mw.flow_proj['0'].weight.data.normal_(0, 0.02); mw.proj[0].weight.data.normal_(0, 0.02)
        a1 = mw(feats, act, lay)
        a2 = mw(feats2, act, lay)
    d01 = max((a[:, :2].float() - b[:, :2].float()).abs().max().item() for a, b in zip(a1, a2))
    check('9c causality with warp', d01 == 0, f'diff frames 0..1 = {d01}')

    # 10 save / load
    import tempfile
    args = dict(pools=[8, 4, 2, 1], depth=2, no_layout=False, residual=False, warp_levels=[0, 1])
    with tempfile.TemporaryDirectory() as td:
        torch.save(dict(model=mw.state_dict(), args=args, step=1), f'{td}/m.pt')
        m2, ck = load_predictor(f'{td}/m.pt', dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            b1 = mw(feats, act, lay)
            b2 = m2(feats, act, lay)
        check('10 save/load bitwise', all(torch.equal(u, v) for u, v in zip(b1, b2)) and m2.warp_levels == (0, 1))

    print('ALL PASS' if not fails else f'FAILED: {fails}')
    return not fails


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
