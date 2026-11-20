"""关联通道（``--assoc_layout``，pyramid_predictor 文档第 7 条）的单测。用法：CUDA_VISIBLE_DEVICES=4 ../venv/bin/python test_assoc_layout.py

渲染（nusc_data）：
 1a. render_assoc_layout(fr, fr) == render_layout(fr)（逐字节；关键帧 + 相位帧）：关联通道与主通道是同一份投影 / 填充代码，只换了框。
 1b. render_layout_cond(assoc=False) == render_layout(fr_tgt, map)（逐字节，就是旧入口）；assoc=True → [6,6,H,W]，前 3 通道同上。
 1c. 实例筛选：关联通道 = 只保留 fr_tgt 里也在场的 fr_prev 框后按 fr_tgt 相机画（逐字节对独立构造）；删掉目标帧里的一个实例，它在关联通道里也消失。
 1d. 运动场景：关联通道 ≠ 主通道（框位移了）、≠ 在 fr_prev 自己相机里画（相机几何取目标帧）。
 1e. 相位帧：关联通道用 fr_prev 每个相机时刻的插值框（cams[ch]['boxes']），不是关键帧标注。
数据（clip_dataset）：
 2. assoc_layout=True：layout [T,6,6,H,W]；前 3 通道与 assoc_layout=False 逐位相同（同 clip、同固定相位、开地图）；
    后 3 通道 [t] == render_assoc_layout(fr[t], fr[t+1])；末帧全黑。
预测器（rofull6000 真权重热启动）：
 3a. 热启动只缺 layout_assoc.*；零初始化；构造关联卷积不消耗全局 CPU RNG（3a0：DataLoader worker 种子 / 相位采样与对照臂相同）。
 3b. 前向逐位相同：fp32 / bf16 autocast，关联通道填随机非零输入。
 3c. 训练口径（train 模式 + activation checkpointing + autocast 的 sliding_rollout，L=4 F=2）：z_tf / z_ar 逐位相同、loss 相同；

     120/146 个参数的梯度就差 ~7.6e-6，开 use_deterministic_algorithms 也一样，所以判据是"在噪声底以内"而不是逐位）；
     layout_assoc 的梯度非零（新通道可学）。
 3d. 关联权重非零后，关联输入确实改变输出。
 3e. ckpt 往返：model_kwargs 带 assoc_layout → load_predictor 重建、uses_assoc_layout=True；旧 ckpt uses_assoc_layout=False。
仿真位姿（offlog.layout_reproject.render_cond）：
 4a. 恒等位姿：render_cond(assoc=True/False) == 训练口径 nd.render_layout_cond(kf[i], kf[i+1], map)（逐字节）。
 4b. 横向 δ=2 m：关联通道 ≠ δ=0（框随仿真相机移动）。
"""
import featsim_paths as _release_paths
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import nusc_data as nd  # noqa: E402
import nusc_sweeps as ns  # noqa: E402
from clip_dataset import NuscClips  # noqa: E402
from nusc_map import MapVectors  # noqa: E402
from pyramid_predictor import PyramidPredictor, LEVEL_GRIDS, load_predictor, sliding_rollout, uses_assoc_layout  # noqa: E402
from train_pred import sliding_losses  # noqa: E402

DATA = os.environ.get('UNIAD_BENCH', _release_paths.BENCH)  # ckpt 等数据读活目录；代码根 BENCH 在快照里是 <RUN>/code
INIT = os.environ.get('ASSOC_TEST_INIT', f'{DATA}/ro/runs/full_20261105/train_full/iter_6000.pt')
fails = []


def check(name, cond, msg=''):
    print(f'[{"PASS" if cond else "FAIL"}] {name} {msg}', flush=True)
    if not cond:
        fails.append(name)


def moving_pair(idx, names, min_disp=3.0):
    """找一对相邻关键帧（i, i+1），至少有一个两帧都在场、世界系位移 ≥ min_disp 的实例。"""
    for s in names:
        kf = idx['scenes'][s]
        for i in range(len(kf) - 1):
            a, b = kf[i], kf[i + 1]
            mb = {x['inst']: x for x in b['boxes']}
            for x in a['boxes']:
                y = mb.get(x['inst'])
                if y is not None and np.linalg.norm(y['center'] - x['center']) >= min_disp and len(a['boxes']) >= 5:
                    return s, i, x['inst']
    raise RuntimeError('没找到运动实例')


def test_render():
    idx = nd.build_index()
    names = nd.splits()['val']
    mv = MapVectors()
    s, i, inst = moving_pair(idx, names)
    fp, ft = idx['scenes'][s][i], idx['scenes'][s][i + 1]
    print(f'[info] scene {s} frames {i}->{i + 1}, moving inst {inst[:8]}')
    sf = ns.SweepFrames(idx)
    gp, gt_ = sf.frame(s, i, 3), sf.frame(s, i + 1, 3)

    ok = all(np.array_equal(nd.render_assoc_layout(f, f), nd.render_layout(f)) for f in (fp, ft, gp, gt_))
    check('1a assoc(fr, fr) == render_layout(fr) (keyframe + phase frame, byte-exact)', ok)

    m = mv.for_frame(ft)
    ref = nd.render_layout(ft, map_vec=m)
    c0 = nd.render_layout_cond(fp, ft, map_vec=m, assoc=False)
    c1 = nd.render_layout_cond(fp, ft, map_vec=m, assoc=True)
    check('1b cond(assoc=False) == render_layout; assoc=True first 3 ch == render_layout',
          np.array_equal(c0, ref) and c1.shape == (6, 6, *nd.LAYOUT_HW) and np.array_equal(c1[:, :3], ref)
          and np.array_equal(c1[:, 3:], nd.render_assoc_layout(fp, ft)), f'shape {c1.shape}')

    keep = {b['inst'] for b in ft['boxes']}
    exp = nd.render_layout(dict(ft, boxes=[b for b in fp['boxes'] if b['inst'] in keep]))
    got = nd.render_assoc_layout(fp, ft)
    ft_minus = dict(ft, boxes=[b for b in ft['boxes'] if b['inst'] != inst])
    got_minus = nd.render_assoc_layout(fp, ft_minus)
    exp_minus = nd.render_layout(dict(ft, boxes=[b for b in fp['boxes'] if b['inst'] in keep and b['inst'] != inst]))
    n_drop = len([b for b in fp['boxes'] if b['inst'] not in keep])
    check('1c instance filter (both frames present) == independent construction; removed instance disappears',
          np.array_equal(got, exp) and np.array_equal(got_minus, exp_minus) and not np.array_equal(got, got_minus),
          f'prev boxes {len(fp["boxes"])}, dropped (not in tgt) {n_drop}')

    d_main = int((got != nd.render_layout(ft)).any(1).sum())
    d_prevcam = int((got != nd.render_layout(dict(fp, boxes=[b for b in fp['boxes'] if b['inst'] in keep]))).any(1).sum())
    check('1d moving scene: assoc != main channel and != prev-camera rendering', d_main > 0 and d_prevcam > 0,
          f'pixels differing: vs main {d_main}, vs prev-camera {d_prevcam}')

    a_ph = nd.assoc_frame(gp, gt_)
    use_cam = all(all(any(b is c for c in gp['cams'][ch]['boxes']) for b in a_ph['cams'][ch]['boxes']) for ch in nd.CAMS)
    kf_boxes = nd.render_layout(dict(gt_, cams={ch: dict(c, boxes=[b for b in fp['boxes'] if b['inst'] in {x['inst'] for x in gt_['cams'][ch]['boxes']}])
                                                  for ch, c in gt_['cams'].items()}))
    check('1e phase frame: assoc uses per-camera interpolated prev boxes (not keyframe annotations)',
          use_cam and not np.array_equal(nd.render_assoc_layout(gp, gt_), kf_boxes))
    return s, i


def test_dataset(scene, i):
    T = 6
    kw = dict(split='val', T=T, step=1, map_layout=True, sweeps=True, phase_mode=2)
    d0, d1 = NuscClips(**kw), NuscClips(**kw, assoc_layout=True)
    j = d0.clips.index((scene, max(0, min(i, len(d0.idx['scenes'][scene]) - T - 1))))
    x0, x1 = d0[j], d1[j]
    fr = d1._frames(*d1.clips[j], x1['phase'])
    L0, L1 = x0['layout'].numpy(), x1['layout'].numpy()
    ok_tail = all(np.array_equal(L1[t, :, 3:], nd.render_assoc_layout(fr[t], fr[t + 1])) for t in range(T - 1))
    check('2 dataset: [T,6,6,H,W]; main ch == assoc_layout=False; assoc ch == render_assoc_layout(fr[t], fr[t+1]); last slot black',
          L1.shape == (T, 6, 6, *nd.LAYOUT_HW) and np.array_equal(L1[:, :, :3], L0) and ok_tail and not L1[T - 1].any()
          and torch.equal(x0['imgs'], x1['imgs']) and torch.equal(x0['actions'], x1['actions']) and x1['phase'] == 2 and L1[:, :, 3:].any(),
          f'shape {L1.shape} phase {x1["phase"]}')


def test_predictor():
    dev = torch.device('cuda')
    ck = torch.load(INIT, map_location='cpu', weights_only=False)
    kw = dict(ck['model_kwargs'])
    torch.manual_seed(0)
    m0 = PyramidPredictor(**kw, use_activation_checkpointing=True).to(dev)
    r0 = torch.rand(4)
    m0.load_state_dict(ck['model'])
    torch.manual_seed(0)
    m1 = PyramidPredictor(**kw, assoc_layout=True, use_activation_checkpointing=True).to(dev)
    r1 = torch.rand(4)
    check('3a0 building the assoc conv does not consume the global CPU RNG (same data-loader seeds as the control arm)', torch.equal(r0, r1))
    missing, unexpected = m1.load_state_dict(ck['model'], strict=False)
    check('3a warm start: only layout_assoc.* missing, zero-init', not unexpected and sorted(missing) == ['layout_assoc.bias', 'layout_assoc.weight']
          and not m1.layout_assoc.weight.any() and not m1.layout_assoc.bias.any(), f'missing {missing} unexpected {unexpected}')

    g = torch.Generator().manual_seed(1)
    B, L, F_, N = 1, 4, 2, 6
    T = L + F_
    feats = [torch.randn(B, T, N, 256, h, w, generator=g).to(dev) for h, w in LEVEL_GRIDS]
    act = torch.randn(B, T, 3, generator=g).to(dev)
    H, W = nd.LAYOUT_HW
    lay3 = (torch.rand(B, T, N, 3, H, W, generator=g) > 0.8).float().to(dev)
    lay_a = (torch.rand(B, T, N, 3, H, W, generator=g) > 0.7).float().to(dev) * 0.6
    lay6 = torch.cat([lay3, lay_a], 3)

    m0.eval(), m1.eval()
    with torch.no_grad():
        y0 = m0([f[:, :L] for f in feats], act[:, :L], lay3[:, :L])
        y1 = m1([f[:, :L] for f in feats], act[:, :L], lay6[:, :L])
        eq32 = all(torch.equal(a, b) for a, b in zip(y0, y1))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y0 = m0([f[:, :L] for f in feats], act[:, :L], lay3[:, :L])
            y1 = m1([f[:, :L] for f in feats], act[:, :L], lay6[:, :L])
        eq16 = all(torch.equal(a, b) for a, b in zip(y0, y1))
    check('3b forward bit-identical to rofull6000 (fp32, bf16 autocast) with nonzero assoc input', eq32 and eq16, f'fp32 {eq32} bf16 {eq16}')

    m0.train(), m1.train()
    outs = []
    for m, lay in ((m0, lay3), (m1, lay6), (m0, lay3)):
        m.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z_tf, z_ar = sliding_rollout(m, feats, act[:, :T - 1], lay[:, :T - 1], L, F_)
        loss, _ = sliding_losses(z_tf, z_ar, feats, L, F_)
        loss.backward()
        outs.append((z_tf, z_ar, loss.detach(), {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}))
    (zt0, za0, l0, g0), (zt1, za1, l1, g1), (_, _, _, g0b) = outs
    eq_z = all(torch.equal(a, b) for a, b in zip(zt0 + za0, zt1 + za1))
    common = sorted(set(g0) & set(g1))
    max_gd = max((g0[n] - g1[n]).abs().max().item() for n in common)
    noise = max((g0[n] - g0b[n]).abs().max().item() for n in common)
    ga = g1.get('layout_assoc.weight')
    check('3c train-mode sliding rollout (ckpt + autocast): z / loss bit-identical, old-param grads within repeat-run noise, assoc grad nonzero',
          eq_z and torch.equal(l0, l1) and set(g0) == set(common) and max_gd <= 2 * noise + 1e-12 and ga is not None and ga.abs().max().item() > 0,
          f'loss {l0.item():.6f}/{l1.item():.6f} grads {len(common)}: max |Δ| vs rofull6000 {max_gd:.2e}, repeat-run noise {noise:.2e}; '
          f'|grad assoc|max {0 if ga is None else ga.abs().max().item():.2e}')

    m1.eval()
    with torch.no_grad():
        torch.nn.init.normal_(m1.layout_assoc.weight, std=0.02)
        ya = m1([f[:, :L] for f in feats], act[:, :L], lay6[:, :L])
        yb = m1([f[:, :L] for f in feats], act[:, :L], torch.cat([lay3, torch.zeros_like(lay_a)], 3)[:, :L])
    diff = max((a - b).abs().max().item() for a, b in zip(ya, yb))
    check('3d nonzero assoc weights: assoc input changes output', diff > 0, f'max diff {diff:.3e}')

    with tempfile.TemporaryDirectory() as tdir:
        p = os.path.join(tdir, 'x.pt')
        torch.save(dict(model=m1.state_dict(), args=dict(ck['args'], assoc_layout=True), model_kwargs=dict(kw, assoc_layout=True), step=1), p)
        m2, ck2 = load_predictor(p, device=dev)
        p2 = os.path.join(tdir, 'y.pt')  # 没有 model_kwargs 的旧式 ckpt：从 args 还原
        torch.save(dict(model=m1.state_dict(), args=dict(ck['args'], assoc_layout=True), step=1), p2)
        m3, ck3 = load_predictor(p2, device=dev)
    check('3e ckpt roundtrip: assoc rebuilt from model_kwargs / args; old ckpt -> False',
          m2.layout_assoc is not None and m3.layout_assoc is not None and uses_assoc_layout(ck2) and uses_assoc_layout(ck3)
          and not uses_assoc_layout(ck) and torch.equal(m2.layout_assoc.weight, m1.layout_assoc.weight))


def test_reproject():
    if BENCH not in sys.path:
        sys.path.insert(0, BENCH)
    from offlog.layout_reproject import SimLayoutRenderer
    from offlog.sim_pose import LogTrack
    from uniad_port.data import UniADValData
    data = UniADValData(num_prefetch=1)
    rend = SimLayoutRenderer(data.infos)
    tr = LogTrack(data, data.scenes[12])
    a = tr.anchors(stride=8, min_speed=0.0)[0]
    p0, p2 = tr.rollout(a, 'lat', 0.0), tr.rollout(a, 'lat', 2.0)
    ok, n = True, 0
    for k in range(1, 5):
        f0, f1 = rend.nd_frame(p0[k - 1].tmpl_idx), rend.nd_frame(p0[k].tmpl_idx)
        mv = rend.map_vecs.for_frame(rend.nd_frame(p0[k].cmd_idx))
        for assoc in (False, True):
            got = rend.render_cond(p0[k - 1], p0[k], with_map=True, assoc=assoc)
            ok &= np.array_equal(got, nd.render_layout_cond(f0, f1, map_vec=mv, assoc=assoc))
            n += 1
        ok &= np.array_equal(rend.render_cond(p0[k - 1], p0[k], assoc=False), rend.render(p0[k]))
    check('4a identity pose: render_cond == training render_layout_cond (byte-exact, assoc on/off)', ok, f'{n} comparisons')
    d = int((rend.render_cond(p2[5], p2[6], assoc=True)[:, 3:] != rend.render_cond(p0[5], p0[6], assoc=True)[:, 3:]).sum())
    check('4b lateral 2 m: assoc channel moves with the sim camera', d > 0, f'{d} bytes differ')


if __name__ == '__main__':
    s, i = test_render()
    test_dataset(s, i)
    test_predictor()
    test_reproject()
    print('ALL PASS' if not fails else f'FAILED: {fails}')
    sys.exit(1 if fails else 0)
