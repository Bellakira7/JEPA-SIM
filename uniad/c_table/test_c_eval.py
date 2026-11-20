"""test c eval.

  ① stream：「log」臂（位姿强制沿日志、真图、命令走 Route.command、帧走 OffLogEnv.sim_frame）与普通真图流式（data.get_frame）UniAD 输出逐位相同（每步摘要哈希）。
  ④ pred  ：StepPredictor 逐步预测 与 B 表 b_eval.SimARSource.rollout_cond（横向 δ=+2 m，仿真位姿条件）/ A 表 a_eval.ARSource.rollout（δ=0）逐位相同；
            并且 log / copy / 预测器三个臂的 warmup 段（前 W 步）UniAD 输出摘要相同。
  ② splat ：SplatAD 服务（splat_server.py）在日志位姿（D=None）的渲染与 render_eval 的 shift_+0.0 逐像素相同；一个 D≠None 的请求与 B 表 3DGS 作业的渲染图逐像素相同。

    CUDA_VISIBLE_DEVICES=2 venv/bin/python c_table/test_c_eval.py --checks stream,pred,splat
"""
import featsim_paths as _release_paths
import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

ROOT = os.environ.get('G3_CODE_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, 'c_table'), os.path.join(ROOT, 'b_table'), os.path.join(ROOT, 'a_table'), os.path.join(ROOT, 'feat'), ROOT,
          os.path.join(ROOT, 'uniad_port')):
    if p not in sys.path:
        sys.path.insert(0, p)
import c_eval as ce  # noqa: E402
import closed_loop as cl  # noqa: E402
from offlog import common  # noqa: E402

BENCH = common.UNIAD_BENCH
FAILS = []


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def setup(need_preds):
    from uniad_port.data import UniADValData, CAM_ORDER
    from uniad_port.runner import UniADRunner
    from offlog.gt_world import GTWorld
    from offlog.map_gt import MapGT
    from offlog.metas import OffLogEnv
    from offlog.layout_reproject import SimLayoutRenderer
    torch.set_grad_enabled(False)
    g = ce.G()
    g.W = 4
    g.data = UniADValData(num_prefetch=0)
    g.cams = tuple(CAM_ORDER)
    g.runner = UniADRunner(ce.CKPTS['hf'], device='cuda:0', verbose=False)
    g.gtw = GTWorld([i['token'] for i in g.data.infos], cache_path=os.path.join(common.CACHE_DIR, 'gt_world_val.pkl'))
    g.env = OffLogEnv(g.data, g.gtw, MapGT())
    g.rend = SimLayoutRenderer(g.data.infos)
    srcs = {}
    if need_preds:
        dev = torch.device('cuda:0')
        st = torch.load(ce.STATS.format(ckpt='hf'))
        g.mean, g.std = [m.to(dev) for m in st['mean']], [s.to(dev) for s in st['std']]
        for spec in ce.DEFAULT_PREDS:
            n_, p_ = spec.split('=', 1)
            srcs[n_] = ce.StepPredictor(n_, p_, g.mean, g.std, dev, 4)
    return g, srcs


def c_stream(g, si, n_steps):
    print(f'[① stream] scene {si}')
    scn = ce.Scene(g, si)
    n_end = min(scn.n, g.W + n_steps)
    ref = []
    g.runner.reset_scene()
    for j in range(n_end):
        fr = g.data.get_frame(scn.ids[j], load_images=True)
        ref.append(ce.out_digest(g.runner.step_frame(g.runner.extract_feats(fr['img']), fr)))
    imgs = ce.real_images(g.data, scn, 0, scn.n)
    warm = [g.runner.extract_feats(next(imgs)) for _ in range(g.W)]
    dig = []
    rows, summ = ce.run_arm(g, scn, 'log', 'log', warm, img_iter=imgs, max_steps=n_steps, digests=dig)
    same = sum(a == b for a, b in zip(ref, dig))
    check(len(dig) == len(ref) and same == len(ref), f'log 臂 vs 普通真图流式：{same}/{len(ref)} 步逐位相同')
    check(all(r['cmd'] == r['cmd_log'] for r in rows), 'log 臂的路线命令 = 官方命令（每步）')
    check(all(abs(r['lat']) < 1e-9 for r in rows) and abs(summ['progress'] - 1.0) < 1e-9, f'log 臂 lat=0、progress={summ["progress"]:.6f}')
    return scn, warm


def c_pred(g, srcs, si):
    import a_eval
    print(f'[④ pred] scene {si}')
    scn = ce.Scene(g, si)
    an = scn.tr.anchors(stride=8, min_hist=4, horizon=8, max_scale=1.5, min_speed=5.0) or [4]
    a = an[0]
    g.runner.reset_scene()
    feats = [g.runner.extract_feats(g.data.get_frame(scn.ids[j])['img']) for j in range(a + 1)]
    try:
        import b_eval
        sim_ok = hasattr(b_eval, 'SimARSource')
    except Exception as e:             # b_eval 在别处被改坏时不拖垮本检查：只做 δ=0 对拍
        print(f'  (b_eval 导入失败，跳过 δ=+2 对拍：{e!r})')
        sim_ok = False
    for name, s_ in srcs.items():
        wm = s_.map_layout
        C = min(s_.ctx, a + 1)
        ctx_j = list(range(a + 1 - C, a + 1))
        for val in ([0.0, 2.0] if sim_ok else [0.0]):
            poses = scn.tr.rollout(a, 'lat', val, horizon=8)
            # 逐步（c_eval 的写法）
            seq = [s_.norm(feats[j]) for j in ctx_j]
            E = [scn.ctx_E(j) for j in ctx_j]
            L = [None] + [scn.ctx_layout(j, wm) for j in ctx_j[1:]]
            mine = []
            for p in poses[1:]:
                E_new = g.rend.sim_frame(p)[0]['ego2global']
                lay_new = g.rend.render(p, with_map=wm)
                nxt, raw = s_.predict(seq, E + [E_new], L + [lay_new])
                mine.append(raw)
                seq, E, L = (seq + [nxt])[-s_.ctx:], (E + [E_new])[-s_.ctx:], (L + [lay_new])[-s_.ctx:]
            # 参照
            if val == 0.0:
                ar = a_eval.ARSource(name, s_.path, s_.mean, s_.std, s_.dev, 4)
                ctx_nd = [g.rend.nd_frame(scn.ids[j]) for j in ctx_j]
                tgt_nd = [g.rend.nd_frame(p.tmpl_idx) for p in poses[1:]]

                def layout_of(fr, with_map, _c={}):
                    k = (fr['token'], with_map)
                    if k not in _c:
                        _c[k] = __import__('nusc_data').render_layout(fr, map_vec=g.rend.map_vecs.for_frame(fr) if with_map else None)
                    return _c[k]
                ref = ar.rollout([feats[j] for j in ctx_j], ctx_nd, tgt_nd, layout_of)
            else:
                sr = b_eval.SimARSource(name, s_.path, s_.mean, s_.std, s_.dev, 4)
                ctx_nd = [g.rend.nd_frame(scn.ids[j]) for j in ctx_j]
                sim_nd = [g.rend.sim_frame(p)[0] for p in poses]
                act_seq = np.concatenate([b_eval.nd_actions(ctx_nd), b_eval.nd_actions(sim_nd)], 0).astype(np.float32)
                lay_seq = [None] + [scn.ctx_layout(j, wm) for j in ctx_j[1:]] + [g.rend.render(p, with_map=wm) for p in poses[1:]]
                ref = sr.rollout_cond([feats[j] for j in ctx_j], act_seq, lay_seq)
            md = max(float((x - y).abs().max()) for a_, b_ in zip(mine, ref) for x, y in zip(a_, b_))
            check(md == 0.0, f'{name} δ={val:+.1f} 逐步预测 vs {"A 表 ARSource.rollout" if val == 0 else "B 表 SimARSource.rollout_cond"}：8 步 × 4 层 最大差 {md:.3g}')
    # 关联通道（O5）：warmup 段（日志帧，nd.render_layout_cond）与闭环段（仿真位姿，rend.render_cond）在恒等位姿上逐字节相同；assoc=False 时 sim_layout = 旧 render
    if hasattr(g.rend, 'render_cond'):
        n_eq = n_all = 0
        for wm in (False, True):
            for j in range(1, 4):
                p0, p1 = cl.log_pose(scn.tr, j - 1), cl.log_pose(scn.tr, j)
                n_all += 2
                n_eq += int(np.array_equal(scn.ctx_layout(j, wm, True), g.rend.render_cond(p0, p1, with_map=wm, assoc=True)))
                n_eq += int(np.array_equal(scn.ctx_layout(j, wm, False), scn.sim_layout(p0, p1, wm, False)))
        pl = scn.tr.rollout(a, 'lat', 2.0, horizon=2)
        same_old = np.array_equal(scn.sim_layout(pl[0], pl[1], True, False), g.rend.render(pl[1], with_map=True))
        lay6 = scn.sim_layout(pl[0], pl[1], True, True)
        check(n_eq == n_all and same_old and lay6.shape[1] == 6 and np.array_equal(lay6[:, :3], g.rend.render(pl[1], with_map=True)),
              f'关联通道 layout：日志帧 vs 仿真位姿接口 {n_eq}/{n_all} 逐字节相同；assoc=False 仿真位姿 = 旧 render：{same_old}；assoc=True 形状 {lay6.shape}')
    # warmup 段：三个臂摘要相同
    imgs = ce.real_images(g.data, scn, 0, scn.n)
    warm = [g.runner.extract_feats(next(imgs)) for _ in range(g.W)]
    digs = {}
    for arm, kind, s_ in [('log', 'log', None), ('copy', 'copy', None)] + [(n_, 'pred', s_) for n_, s_ in srcs.items()]:
        d = []
        ce.run_arm(g, scn, arm, kind, warm, src=s_, img_iter=imgs if kind == 'log' else None, max_steps=2, digests=d)
        digs[arm] = d
    w = {arm: tuple(d[:g.W]) for arm, d in digs.items()}
    check(len(set(w.values())) == 1, f'warmup {g.W} 步：{len(w)} 个臂 UniAD 输出摘要相同')
    check(len({d[g.W] for d in digs.values()}) == len(digs), '接管第 1 步各臂输出不同（特征来源确实不同）')


def c_kv(g, srcs, si):
    """⑥ KV cache 路径（11-10；δ=0 仿真位姿条件，8 步闭环）：
       a) StepPredictor(kv=True) 逐步闭环 与 a_eval.ARSource(kv=True).rollout_kv（同一份条件）逐位相同（两条评测路径共用同一套 KV 语义）；
       b) 第 1 步与稠密 StepPredictor 逐位相同（首步 = 稠密前向 + return_kv_cache，输出不受影响）；
       c) 缓存连续性检测：用新序列（seq[-1] 不是上次返回的 nxt）再调一次，自动重建，第 1 步结果与 a) 的第 1 步逐位相同；
       d)（报告项）第 2..8 步 KV 对稠密的最大差 —— KV 淘汰近似的量级（ctx=4，淘汰从第 4 步起）。"""
    import a_eval
    import nusc_data
    print(f'[⑥ kv] scene {si}')
    scn = ce.Scene(g, si)
    an = scn.tr.anchors(stride=8, min_hist=4, horizon=8, max_scale=1.5, min_speed=5.0) or [4]
    a = an[0]
    g.runner.reset_scene()
    feats = [g.runner.extract_feats(g.data.get_frame(scn.ids[j])['img']) for j in range(a + 1)]
    poses = scn.tr.rollout(a, 'lat', 0.0, horizon=8)
    for name, s_ in srcs.items():
        wm = s_.map_layout
        C = min(s_.ctx, a + 1)
        ctx_j = list(range(a + 1 - C, a + 1))
        lay_new = [g.rend.render(p, with_map=wm) for p in poses[1:]]
        E_new = [g.rend.sim_frame(p)[0]['ego2global'] for p in poses[1:]]

        def closed_loop(src, n=len(poses) - 1):
            seq = [src.norm(feats[j]) for j in ctx_j]
            E = [scn.ctx_E(j) for j in ctx_j]
            L = [None] + [scn.ctx_layout(j, wm) for j in ctx_j[1:]]
            out = []
            for i in range(n):
                nxt, raw = src.predict(seq, E + [E_new[i]], L + [lay_new[i]])
                out.append(raw)
                seq, E, L = (seq + [nxt])[-src.ctx:], (E + [E_new[i]])[-src.ctx:], (L + [lay_new[i]])[-src.ctx:]
            return out

        mdiff = lambda A, B: max(float((x - y).abs().max()) for a_, b_ in zip(A, B) for x, y in zip(a_, b_))
        dense = closed_loop(s_)
        kv = ce.StepPredictor(name, s_.path, s_.mean, s_.std, s_.dev, 4, kv=True)
        mine = closed_loop(kv)
        ar = a_eval.ARSource(name, s_.path, s_.mean, s_.std, s_.dev, 4, kv=True)

        def layout_of(fr, with_map, _c={}):
            k = (fr['token'], with_map)
            if k not in _c:
                _c[k] = nusc_data.render_layout(fr, map_vec=g.rend.map_vecs.for_frame(fr) if with_map else None)
            return _c[k]
        ref = ar.rollout_kv([feats[j] for j in ctx_j], [g.rend.nd_frame(scn.ids[j]) for j in ctx_j], [g.rend.nd_frame(p.tmpl_idx) for p in poses[1:]], layout_of)
        check(mdiff(mine, ref) == 0.0, f'{name} KV 逐步闭环（StepPredictor）vs a_eval.ARSource.rollout_kv：8 步 × 4 层 最大差 {mdiff(mine, ref):.3g}')
        check(mdiff(mine[:1], dense[:1]) == 0.0, f'{name} KV 第 1 步 vs 稠密：最大差 {mdiff(mine[:1], dense[:1]):.3g}')
        redo = closed_loop(kv, n=1)                    # kv 实例带着第 8 步的缓存；新序列 → 必须重建
        check(mdiff(redo, mine[:1]) == 0.0, f'{name} 缓存连续性检测：新序列自动重建，第 1 步与首次逐位相同（最大差 {mdiff(redo, mine[:1]):.3g}）')
        per = [mdiff(mine[i:i + 1], dense[i:i + 1]) for i in range(len(mine))]
        print(f'  [info] {name} KV 对稠密的逐步最大差（原始特征空间）：' + ' '.join(f'k{i + 1}={d:.3g}' for i, d in enumerate(per)))


def c_splat(g, si_name='scene-0103'):
    print(f'[② splat] {si_name}')
    name2si = {s['scene_name']: i for i, s in enumerate(g.data.scenes)}
    si = name2si[si_name]
    scn = ce.Scene(g, si)
    cfg = f'{BENCH}/splatad/runs/full/scene-0103/splatad/run/config.yml'
    ref_dir = f'{BENCH}/splatad/runs/full/render/shift_+0.0'
    tmp = tempfile.mkdtemp(prefix='c_splat_')
    try:
        cli = ce.SplatClient(cfg, os.path.join(tmp, 's.sock'), os.path.join(tmp, 'server.log'), f'{BENCH}/splatad/venv/bin/python',
                             _release_paths.NEURAD_ROOT, os.path.join(ROOT, 'splatad', 'scripts'),
                             f'{BENCH}/splatad/torch_hub')
        print(f'  服务加载 {cli.sec_load:.0f}s')
        try:
            n = bad = 0
            secs = []
            for j in (5, 20):
                info = g.data.infos[scn.ids[j]]
                files = {c: os.path.basename(info['cams'][c]['data_path']) for c in g.cams}
                r = cli.render(files, None, os.path.join(tmp, f'j{j}'))
                secs.append(r['sec_render'])
                for c in g.cams:
                    a = np.asarray(Image.open(os.path.join(tmp, f'j{j}', f'{c}.jpg')))
                    b = np.asarray(Image.open(os.path.join(ref_dir, c, os.path.splitext(files[c])[0] + '.jpg')))
                    n += 1
                    bad += int(not np.array_equal(a, b))
            check(n == 12 and bad == 0, f'D=None 渲染 vs render_eval shift_+0.0：{n} 张，不同 {bad} 张；渲染 {np.mean(secs):.3f} s/6 相机')
            req = json.load(open(f'{BENCH}/b_table/runs/b3d_full_20261106/requests/{si_name}.json'))
            it = [x for x in req['items'] if x['D'] is not None and 'lat+2.0' in x['out']][0]
            r = cli.render(it['files'], it['D'], os.path.join(tmp, 'b3d'))
            bad = sum(int(not np.array_equal(np.asarray(Image.open(os.path.join(tmp, 'b3d', f'{c}.jpg'))),
                                             np.asarray(Image.open(f'{BENCH}/b_table/runs/b3d_full_20261106/render/{si_name}/{it["out"]}/{c}.jpg'))))
                      for c in g.cams)
            check(bad == 0, f'D≠None（{it["out"]}）vs B 表 3DGS 作业渲染图：6 张不同 {bad} 张')
        finally:
            cli.close()
    finally:
        if os.path.exists(os.path.join(tmp, 'server.log')):
            shutil.copy(os.path.join(tmp, 'server.log'), os.environ.get('C_TEST_LOG_DIR', '/tmp') + '/test_splat_server.log')
        shutil.rmtree(tmp, ignore_errors=True)


def c_det(g, si, det_path):
    """⑤ 检测器框 layout（审稿 W4）：
       a) 检测器索引的帧除框以外与 GT 索引逐字段相同（相机 / 位姿 / token），且漏帧会报错；
       b) 用 GT 框伪造的"检测缓存"（boxes 按 det_layout 的 canonical 约定由 GT 反算）在 δ=0 / δ=+2 m 下渲染出的 layout 与 GT layout 逐字节相同
          （几何、类别映射、渲染路径与 GT 完全共用；只要缓存格式对，@det 与原臂的唯一差别就是框的来源）；
       c) 真实检测缓存：仿真位姿接口（sim_layout det=True）在恒等位姿上与日志帧接口（ctx_layout det=True）逐字节相同，且与 GT layout 不同。"""
    import det_layout as dl
    import nusc_data as nd
    from offlog.layout_reproject import SimLayoutRenderer
    print(f'[⑤ det] scene {si}  {det_path}')
    scn = ce.Scene(g, si)
    toks = [g.data.infos[i]['token'] for i in scn.ids]
    base = g.rend.idx
    # b) GT → canonical → 伪检测缓存
    fake = {}
    for i, t in zip(scn.ids, toks):
        info = g.data.infos[i]
        l2e_r, l2e_t, e2g_r, e2g_t = g.data._transforms(info)
        R, tt = l2e_r.T @ e2g_r.T, l2e_t @ e2g_r.T + e2g_t
        bx, sc, lab, tid = [], [], [], []
        inv = {c: k for k, c in reversed(list(enumerate(dl.LABEL2CLS.tolist())))}
        for b in base['frames'][t]['boxes']:
            c_l = (b['center'] - tt) @ R.T
            Rl = R @ b['R']
            yaw = np.arctan2(Rl[1, 0], Rl[0, 0])
            w, l, h = b['wlh']
            bx.append([c_l[0], c_l[1], c_l[2] - h / 2, l, w, h, yaw, 0, 0]); sc.append(1.0); lab.append(inv[b['cls']]); tid.append(len(tid))
        fake[t] = dict(boxes=np.asarray(bx, np.float64).reshape(-1, 9), scores=np.asarray(sc), labels=np.asarray(lab, np.int64),
                       track_ids=np.asarray(tid, np.int64), l2g_r=R, l2g_t=tt)
    tmp = tempfile.mkdtemp(prefix='c_det_')
    try:
        import pickle
        fp = os.path.join(tmp, 'fake.pkl')
        pickle.dump(dict(meta=dict(ckpt='fake_gt'), frames=fake), open(fp, 'wb'))
        ix = dl.load_det_index(fp, base, tokens=toks)
        same_geo = all(ix['frames'][t]['cams'] is base['frames'][t]['cams'] and np.array_equal(ix['frames'][t]['ego2global'], base['frames'][t]['ego2global'])
                       for t in toks)
        try:
            dl.load_det_index(fp, base, tokens=toks + [g.data.infos[0]['token'] if g.data.infos[0]['token'] not in fake else 'nope'])
            miss_raises = False
        except AssertionError:
            miss_raises = True
        check(same_geo and miss_raises, f'a) 检测器索引只换框（相机 / 位姿对象相同：{same_geo}）；漏帧报错：{miss_raises}')
        rd = SimLayoutRenderer(g.data.infos, nd_index=ix, map_vecs=g.rend.map_vecs)
        n = eq = 0
        an = scn.tr.anchors(stride=8, min_hist=4, horizon=8, max_scale=1.5, min_speed=5.0) or [4]
        for val in (0.0, 2.0):
            for p in scn.tr.rollout(an[0], 'lat', val, horizon=4)[1:]:
                for wm in (False, True):
                    n += 1
                    eq += int(np.array_equal(rd.render(p, with_map=wm), g.rend.render(p, with_map=wm)))
        check(eq == n, f'b) GT 反算的伪检测缓存 → layout 与 GT layout 逐字节相同：{eq}/{n}（δ=0 / +2 m，带 / 不带地图）')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # c) 真实检测缓存
    ix = dl.load_det_index(det_path, base, tokens=toks)
    g.rend_det = SimLayoutRenderer(g.data.infos, nd_index=ix, map_vecs=g.rend.map_vecs)
    n = eq = diff = 0
    for j in range(1, min(6, scn.n)):
        p0, p1 = cl.log_pose(scn.tr, j - 1), cl.log_pose(scn.tr, j)
        for wm in (False, True):
            a_ = scn.ctx_layout(j, wm, False, True)
            n += 1
            eq += int(np.array_equal(a_, scn.sim_layout(p0, p1, wm, False, True)))
            diff += int(not np.array_equal(a_, scn.ctx_layout(j, wm, False, False)))
    check(eq == n and diff > 0, f'c) 真实检测框：日志帧 vs 仿真位姿接口逐字节相同 {eq}/{n}；与 GT layout 不同的 {diff}/{n}')


def c_difix(g, si_name='scene-0103'):
    """⑥ Difix 逐步后处理（审稿 W2）：c_eval.difix_frame 对 B 表 3DGS 作业一组渲染图（δ=+2 m）的输出，与 O2 作业 difix_dir.py 的输出逐像素相同。"""
    print(f'[⑥ difix] {si_name}')
    o2 = f'{BENCH}/difix/runs/o2_20261107/difix_b/{si_name}'
    src = f'{BENCH}/b_table/runs/b3d_full_20261106/render/{si_name}'
    req = json.load(open(f'{BENCH}/b_table/runs/b3d_full_20261106/requests/{si_name}.json'))
    it = [x for x in req['items'] if x['D'] is not None and 'lat+2.0' in x['out']][0]
    meta = json.load(open(os.path.join(o2, '.difix_done_blat')))
    runner = ce.load_difix(meta['weights'], meta['size'])
    tmp = tempfile.mkdtemp(prefix='c_difix_')
    try:
        ce.difix_frame(runner, os.path.join(src, it['out']), tmp, g.cams)
        bad = sum(int(not np.array_equal(np.asarray(Image.open(os.path.join(tmp, f'{c}.jpg'))), np.asarray(Image.open(os.path.join(o2, it['out'], f'{c}.jpg')))))
                  for c in g.cams)
        diff_raw = sum(int(not np.array_equal(np.asarray(Image.open(os.path.join(tmp, f'{c}.jpg'))), np.asarray(Image.open(os.path.join(src, it['out'], f'{c}.jpg')))))
                       for c in g.cams)
        check(bad == 0 and diff_raw == 6, f'{it["out"]}：逐步 Difix vs O2 作业输出 6 张不同 {bad} 张（与渲染原图不同 {diff_raw}/6，确实处理过）')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checks', default='stream,pred,splat')
    ap.add_argument('--scene', type=int, default=1)
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--det-boxes', default='', help='⑤ det 检查用的 det_boxes.pkl')
    a = ap.parse_args()
    checks = a.checks.split(',')
    g, srcs = setup('pred' in checks or 'kv' in checks)
    if 'stream' in checks:
        c_stream(g, a.scene, a.steps)
    if 'pred' in checks:
        c_pred(g, srcs, a.scene)
    if 'kv' in checks:
        c_kv(g, srcs, a.scene)
    if 'splat' in checks:
        c_splat(g)
    if 'det' in checks:
        c_det(g, a.scene, a.det_boxes)
    if 'difix' in checks:
        c_difix(g)
    print('ALL PASS' if not FAILS else f'FAILED: {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
