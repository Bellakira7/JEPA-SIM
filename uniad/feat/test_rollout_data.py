"""滑窗 rollout / 相位增广 / 地图 layout 的单测。用法：CUDA_VISIBLE_DEVICES=2 ../venv/bin/python test_rollout_data.py

滑窗（桩预测器：输出 = 输入 + 1，位置 p 的输出就是"下一帧"，真实帧 t 的特征填常数 t，所以每个窗口里每一帧是谁一眼可对账）：
 1. 窗口内容：第 j 步的上下文 = 帧 j-1 .. L+j-2（真实 + 预测拼成的最后 L 帧，连续编号），F>L 时后几步全是预测帧；条件（action/layout）取 [j-1, j-1+L)。
 2. 预测值：帧 L+k-1 的预测 = L+k-1（链式 +1 与真实编号一致）；z_tf 位置 p ↔ 帧 p+1。
 3. TBPTT：第 j≥2 步喂给 predictor 的上下文 requires_grad=False（窗口 0 的输入本来就没梯度），而输出有梯度；z_ar 能把梯度传回参数。
 4. F=1：z_ar 为空；F>actions 可用长度 → 报错。
 5. 与真实 PyramidPredictor 一致：第 2 步 = 手工拼上下文后直接调 predictor 的最后一个位置，逐位相同；形状正确。
 6. sliding_losses：loss = (L_TF + ΣL_k)/F 与手算一致；copy_k 用最后一个真实上下文帧。
 7. F_schedule 解析/查表；最后一项必须等于 --F。
数据：
 8. 不开 sweeps/map：与旧实现逐位相同（layout = nd.render_layout(fr) 无地图版；图像、动作同直接计算）。
 9. 相位帧：每个相机选中的是离 (关键帧时刻 + p·83.3ms) 最近的 sweep（数据里相机 sweeps 偶有掉帧，间隔 >120 ms 约 1.5%，所以不断言固定容差）；
    相邻帧间隔 ≈ 关键帧间隔；p=0 返回原 dict；场景末尾回退到 p=0；随机相位 0..5 都能取到。
10. 插值框：α=0/1 回到端点，只保留两帧都有的实例；yaw 沿最短弧。
11. 地图 layout：不传 map_vec 与旧版逐位相同；传了之后框区域颜色不变（框叠在线上）、非框像素有增加；相位帧也能渲染。
"""
import featsim_paths as _release_paths
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _release_paths.DEPS)
import nusc_data as nd
import nusc_sweeps as ns
from clip_dataset import NuscClips, load_u8, ACT_MEAN, ACT_STD
from nusc_map import MapVectors
from pyramid_predictor import PyramidPredictor, sliding_rollout, LEVEL_GRIDS
from train_pred import F_at, parse_F_schedule, sliding_losses

fails = []


def check(name, cond, msg=''):
    print(f'[{"PASS" if cond else "FAIL"}] {name} {msg}')
    if not cond:
        fails.append(name)


class Stub(nn.Module):
    """输出 = 输入 + 1（每个位置、每层），记录每次调用的入参。带一个可学标量，让输出带梯度。"""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(()))
        self.calls = []

    def forward(self, feats, actions, layout=None):
        self.calls.append(dict(ctx=[f.detach().clone() for f in feats], req=[f.requires_grad for f in feats], act=actions.clone(),
                               lay=None if layout is None else layout.clone()))
        return [f * self.w + 1.0 for f in feats]


def stub_inputs(B=1, T=12, L=4, dev='cpu'):
    # 4 层，每层 [B,T,N=2,C=3,h,w]，真实帧 t 的值 = t；action[b,t,0] = 100+t；layout[b,t] = 1000+t
    feats = [torch.arange(T, dtype=torch.float32).view(1, T, 1, 1, 1, 1).expand(B, T, 2, 3, 2 + l, 3).contiguous().to(dev) for l in range(4)]
    act = (100 + torch.arange(T, dtype=torch.float32)).view(1, T, 1).expand(B, T, 3).contiguous().to(dev)
    lay = (1000 + torch.arange(T, dtype=torch.float32)).view(1, T, 1, 1, 1, 1).expand(B, T, 2, 3, 4, 5).contiguous().to(dev)
    return feats, act, lay


def test_sliding():
    L, Fn = 4, 8
    feats, act, lay = stub_inputs(T=L + Fn, L=L)
    m = Stub()
    z_tf, z_ar = sliding_rollout(m, feats, act[:, :L + Fn - 1], lay[:, :L + Fn - 1], L, Fn)
    ok = len(m.calls) == Fn  # 窗口 0 一次 + 第 2..F 步各一次
    ctx_ok, cond_ok = True, True
    for j, c in enumerate(m.calls, start=1):
        if j == 1:
            want = list(range(0, L))
            t0 = 0
        else:
            t0 = j - 1
            want = list(range(j - 1, L + j - 1))  # 帧 j-1 .. L+j-2（真实 0..L-1 + 预测 L..）
        got = c['ctx'][0][0, :, 0, 0, 0, 0].tolist()
        ctx_ok &= got == [float(x) for x in want]
        cond_ok &= c['act'][0, :, 0].tolist() == [100.0 + t for t in range(t0, t0 + L)]
        cond_ok &= c['lay'][0, :, 0, 0, 0, 0].tolist() == [1000.0 + t for t in range(t0, t0 + L)]
        ctx_ok &= all(x.shape[1] == L for x in c['ctx'])
    check('1 window content (F>L, 窗口恒长 L)', ok and ctx_ok and cond_ok, f'calls={len(m.calls)}')
    check('2a z_tf position p ↔ frame p+1', z_tf[0][0, :, 0, 0, 0, 0].tolist() == [1.0, 2.0, 3.0, 4.0])
    check('2b z_ar[:, i] = frame L+1+i', z_ar[0][0, :, 0, 0, 0, 0].tolist() == [float(L + 1 + i) for i in range(Fn - 1)],
          str(z_ar[0][0, :, 0, 0, 0, 0].tolist()))
    # 3 TBPTT
    check('3a ctx of steps>=2 detached', all(not any(c['req']) for c in m.calls[1:]))
    feats2, act2, lay2 = stub_inputs(T=L + 3)
    m2 = Stub()
    z_tf2, z_ar2 = sliding_rollout(m2, feats2, act2[:, :L + 2], lay2[:, :L + 2], L, 3)
    loss = sum(z.sum() for z in z_ar2)
    loss.backward()
    check('3b z_ar grads reach params', m2.w.grad is not None and m2.w.grad.abs().item() > 0)
    # 梯度截断：z_ar 最后一帧 = w*(w*(w*x+1)+1)+1 若不截断会有 w^3 项；截断后每步只流过自己的 w（d/dw = ctx 值）
    m3 = Stub()
    f3 = [torch.zeros(1, L + 3, 1, 1, 1, 1)] * 4
    a3 = torch.zeros(1, L + 3, 3)
    _, z3 = sliding_rollout(m3, f3, a3[:, :L + 2], None, L, 3)
    z3[0][:, -1].sum().backward()  # 第 3 步的输出：= w*ctx_last + 1，ctx_last 已 detach 成常数（第 2 步预测帧 = w*0+1... 的数值）
    ctx_last = m3.calls[2]['ctx'][0][0, -1, 0, 0, 0, 0].item()
    check('3c grad truncated at step boundary', abs(m3.w.grad.item() - ctx_last) < 1e-6, f'grad={m3.w.grad.item()} ctx_last={ctx_last}')
    # 4
    m4 = Stub()
    _, z1 = sliding_rollout(m4, feats, act[:, :L], lay[:, :L], L, 1)
    check('4a F=1 → z_ar empty & 1 call', z1[0].shape[1] == 0 and len(m4.calls) == 1)
    try:
        sliding_rollout(Stub(), feats, act[:, :L + 2], lay[:, :L + 2], L, 8)
        check('4b actions too short → assert', False)
    except AssertionError:
        check('4b actions too short → assert', True)


def test_real_predictor():
    dev = 'cuda'
    torch.manual_seed(0)
    m = PyramidPredictor(depth=1, residual=True, warp_levels=(0, 1)).to(dev).eval()
    # 打破零初始化，让输出真的依赖输入
    with torch.no_grad():
        for p in list(m.proj.parameters()) + list(m.flow_proj.parameters()):
            p.normal_(0, 0.02)
    L, Fn = 2, 3
    feats = [torch.randn(1, L + Fn, 6, 256, h, w, device=dev) for h, w in LEVEL_GRIDS]
    act = torch.randn(1, L + Fn - 1, 3, device=dev)
    lay = torch.rand(1, L + Fn - 1, 6, 3, *nd.LAYOUT_HW, device=dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        z_tf, z_ar = sliding_rollout(m, feats, act, lay, L, Fn)
        # 手工：第 2 步窗口 = [real1, pred L]
        pL = [z[:, L - 1:L].to(f.dtype) for z, f in zip(z_tf, feats)]
        ctx = [torch.cat([f[:, 1:L], p], 1) for f, p in zip(feats, pL)]
        z2 = m(ctx, act[:, 1:1 + L], lay[:, 1:1 + L])
    d = max((a[:, 0].float() - b[:, -1].float()).abs().max().item() for a, b in zip(z_ar, z2))
    check('5a step-2 = manual predictor call', d == 0, f'max diff {d}')
    check('5b shapes', [tuple(z.shape) for z in z_tf] == [(1, L, 6, 256, h, w) for h, w in LEVEL_GRIDS]
          and [tuple(z.shape) for z in z_ar] == [(1, Fn - 1, 6, 256, h, w) for h, w in LEVEL_GRIDS])


def test_losses_sched():
    L, Fn = 2, 3
    torch.manual_seed(0)
    feats = [torch.randn(1, L + Fn, 1, 2, 3, 3) for _ in range(4)]
    z_tf = [torch.randn(1, L, 1, 2, 3, 3) for _ in range(4)]
    z_ar = [torch.randn(1, Fn - 1, 1, 2, 3, 3) for _ in range(4)]
    loss, d = sliding_losses(z_tf, z_ar, feats, L, Fn)
    lv = lambda p, t: sum((x - y).abs().mean() for x, y in zip(p, t)) / 4
    want = (lv(z_tf, [f[:, 1:L + 1] for f in feats]) + lv([z[:, 0:1] for z in z_ar], [f[:, L + 1:L + 2] for f in feats])
            + lv([z[:, 1:2] for z in z_ar], [f[:, L + 2:L + 3] for f in feats])) / Fn
    check('6a loss = (L_TF + ΣL_k)/F', abs(loss.item() - want.item()) < 1e-6)
    want_copy = [lv([f[:, L - 1:L] for f in feats], [f[:, L + k - 1:L + k] for f in feats]).item() for k in range(1, Fn + 1)]
    check('6b copy_k from last real ctx frame', np.allclose([x.item() for x in d['copy']], want_copy))
    s = parse_F_schedule('0:2,1500:4,3000:8', 8)
    check('7a F schedule lookup', [F_at(s, x) for x in (0, 1499, 1500, 2999, 3000, 99999)] == [2, 2, 4, 4, 8, 8])
    check('7b default schedule', parse_F_schedule(None, 8) == [(0, 8)])
    try:
        parse_F_schedule('0:2,1500:4', 8)
        check('7c last must equal F', False)
    except AssertionError:
        check('7c last must equal F', True)


def test_data():
    idx = nd.build_index()
    sf = ns.SweepFrames(idx)
    sp = nd.splits()
    # 8：不开 sweeps/map → 与直接计算逐位相同
    ds = NuscClips('train', T=3, step=1, max_clips=50, seed=1)
    scene, a = ds.clips[7]
    b = ds[7]
    fr = idx['scenes'][scene][a:a + 3]
    ok = np.array_equal(b['imgs'][1, 2].numpy(), load_u8(fr[1]['cams'][nd.CAMS[2]]['path']))
    act = np.stack([nd.relative_action(fr[0], fr[1]), nd.relative_action(fr[1], fr[2]), np.zeros(3, np.float32)])
    ok &= np.array_equal(b['actions'].numpy(), ((act - ACT_MEAN) / ACT_STD) * np.array([[1], [1], [0]], np.float32))
    ok &= np.array_equal(b['layout'][1].numpy(), nd.render_layout(fr[2])) and not b['layout'][2].any() and b['phase'] == 0
    check('8 default = old behaviour (no sweeps/map)', bool(ok))
    # 9 相位帧
    names = [s for s in sp['train'] if s in idx['scenes']]
    s = names[10]
    sw = sf.sw['scenes'][s]
    n = len(idx['scenes'][s])
    # 数据里相机 sweeps 偶有掉帧（相邻间隔 >120 ms 约占 1.5%，最长 350 ms），所以不能断言"偏差 ≤ 半个标称间隔"，
    # 断言的是：选中的就是离目标时刻 (关键帧时刻 + p·83.3ms) 最近的那个 sweep；文件存在；p=0 返回原 dict。
    ok_t, gap, ok_p0 = True, [], sf.frame(s, 3, 0) is idx['scenes'][s][3]
    for p in range(1, 6):
        f = [sf.frame(s, i, p) for i in range(3, 7)]
        for i, x in zip(range(3, 7), f):
            for ch in nd.CAMS:
                c = sw[ch]
                want = int(c['ts'][c['kpos'][i]]) + p * ns.PHASE_US
                ok_t &= x['cams'][ch]['t'] == int(c['ts'][int(np.argmin(np.abs(c['ts'] - want)))])
                ok_t &= abs(x['cams'][ch]['t'] - want) <= 175_000  # 最长空洞 350 ms 的一半
                ok_t &= x['cams'][ch]['path'].endswith('.jpg') and os.path.exists(x['cams'][ch]['path'])
        gap += [(f[k + 1]['timestamp'] - f[k]['timestamp']) / 1e3 for k in range(3)]
    check('9a phase frames: nearest sweep & files exist & p=0 is original', ok_t and ok_p0)
    check('9b chain spacing ≈ keyframe spacing', 400 < np.mean(gap) < 600, f'mean {np.mean(gap):.0f} ms')
    ds2 = NuscClips('train', T=3, step=1, max_clips=200, seed=2, sweeps=True, phase_mode=3)
    ok = True
    for i in range(0, 200, 17):
        scene_i, a_i = ds2.clips[i]
        o = ds2[i]
        want_p = 3 if a_i + 3 < len(idx['scenes'][scene_i]) else 0
        ok &= o['phase'] == want_p and o['imgs'].shape == (3, 6, 3, 928, 1600)
    check('9c dataset fixed phase & scene-end fallback to p=0', bool(ok))
    # 末尾回退：构造起点 = n-T 的 clip
    ds3 = NuscClips('train', T=3, step=1, sweeps=True, phase_mode=5)
    j = next(k for k, (sc, st) in enumerate(ds3.clips) if st + 3 == len(idx['scenes'][sc]))
    check('9d last clip of scene falls back to p=0', ds3[j]['phase'] == 0)
    dsr = NuscClips('train', T=3, step=1, max_clips=40, seed=3, sweeps=True, phase_mode='random')
    ps = {dsr[0]['phase'] for _ in range(40)}
    check('9e random phase varies', len(ps) >= 3, str(sorted(ps)))
    # 10 插值
    b0, b1 = idx['scenes'][s][3]['boxes'], idx['scenes'][s][4]['boxes']
    m1 = {b['inst'] for b in b1}
    i0, i1 = ns.interp_boxes(b0, b1, 0.0), ns.interp_boxes(b0, b1, 1.0)
    common = [b for b in b0 if b['inst'] in m1]
    ok = len(i0) == len(common) and all(np.allclose(x['center'], y['center']) for x, y in zip(i0, common))
    bb = {b['inst']: b for b in b1}
    ok &= all(np.allclose(x['center'], bb[x['inst']]['center']) for x in i1)
    ra = {'center': np.zeros(3), 'wlh': np.ones(3), 'R': ns._rz(np.pi - 0.1), 'cls': 0, 'inst': 'x'}
    rb = dict(ra, R=ns._rz(-np.pi + 0.1))  # 最短弧：差 0.2 rad，经过 ±π
    mid = ns.interp_boxes([ra], [rb], 0.5)[0]
    yaw = np.arctan2(mid['R'][1, 0], mid['R'][0, 0])
    check('10 interp endpoints / common-only / shortest-arc', bool(ok) and abs(abs(yaw) - np.pi) < 1e-9, f'yaw {yaw:.6f}')
    # 11 地图 layout
    mv = MapVectors()
    f = idx['scenes'][s][5]
    base = nd.render_layout(f)
    mp = nd.render_layout(f, map_vec=mv.for_frame(f))
    box = (base > 0).any(1, keepdims=True)
    check('11a map layout: boxes stay on top, lines added', np.array_equal(np.where(box, mp, 0), np.where(box, base, 0)) and (mp > 0).sum() > (base > 0).sum())
    f3 = sf.frame(s, 5, 3)
    m3 = nd.render_layout(f3, map_vec=mv.for_frame(f3))
    check('11b phase frame renders with map (map_idx)', m3.shape == (6, 3, *nd.LAYOUT_HW) and (m3 > 0).any() and 'map_idx' in f3)
    dsm = NuscClips('train', T=3, step=1, max_clips=20, seed=4, map_layout=True)
    check('11c dataset map_layout != no-map', (dsm[0]['layout'] != NuscClips('train', T=3, step=1, max_clips=20, seed=4)[0]['layout']).any())


if __name__ == '__main__':
    test_sliding()
    test_losses_sched()
    test_real_predictor()
    test_data()
    print('ALL PASS' if not fails else f'FAILED: {fails}')
    sys.exit(0 if not fails else 1)
