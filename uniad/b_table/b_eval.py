"""B 表：偏离日志（设计稿 §5）。真图历史在锚点 a 分叉，之后 8 步（4 s）按**仿真位姿**滚动，对比仿真位姿下的 GT（地图栅格、GT 框）。

与 G4 探针（offlog/g4_probe.py）的关系：同一批锚点、同一套扰动、同一套指标（直接 import `frame_metrics`），区别是这里有**训练好的预测器臂**，
并且加了检测滞后（offlog/det_lag_gt.py，以 GT 世界状态为参照）。G4 的 copy 臂在这里也跑一遍（几乎不花时间），保证各臂在同一批锚点上配对。

臂（--arms，逗号分隔；real 恒有）
  real            真图，只有日志位姿（δ=0 / s=1）= 主流式在 a+k 帧的输出。基准线
  copy            锚点 a 的真实 FPN 冻结，img_metas 按仿真位姿（G4 的 copy-last 下限）
  <name>          --pred name=ckpt 的预测器，AR 8 步，条件取仿真位姿：action = 仿真相对位姿，layout = 仿真位姿下重投影（offlog/layout_reproject）
  <name>@zero     同上，但 action 置零（整个窗口，含真实上下文里的转移）。回应 W2：不画地图的变体在 action 置零后地图 IoU 随 δ 崩掉，才说明 action 真被用上了
  <name>@log      同上，但 action 取日志里同一时刻的相对运动（即"预测器被告知自车仍沿日志走"，layout / UniAD 的 metas 仍是仿真位姿）
  <name>@shuf     同上，但 action 取另一个锚点（跨场景、同扰动）的仿真 action，沿 batch 打乱
  identity        真特征 + 日志位姿（--identity 才跑；自检：必须与 real 逐位相同）
  3dgs            SplatAD 重建在**仿真位姿**下的渲染图（--render-root；先跑 gen_render_requests.py 与 splatad/scripts/render_requests.py）→ UniAD 主干 → 与其他臂同一条 UniAD 路径。
                  只做横向 δ 与航向 ψ（没有速度倍率的渲染）；锚点之前 UniAD 的状态仍是真图历史，a+1 起每帧都吃渲染图（= FlowSim 的 "生成历史"）
  3dgs_<tag>      同上，渲染图根目录取 --render-roots tag=path（如 Difix 后处理的图）；合并时是与 3dgs 不同的臂
  warp            FlowSim 的 BEV-Warp（R=0）：锚点 a 的真实 BEV 按"锚点日志位姿 → 第 k 步仿真位姿"的刚体运动直接 warp（不递归，视野外填 0，
                  = g1_warp.warp 的 'warp'），从 S_a 起每步 step_frame_bev 推进 UniAD 状态（= G1 的 V1，与本表其他臂同一状态口径）。δ=0 时逐位等于 G1 的 warp·V1

扰动族：lat（横向 δ，m）/ speed（速度倍率）/ yaw（航向偏置 ψ，deg，--yaw，默认不跑）。--only 'glob:族,族;glob:族' 按臂限定扰动
（族 = lat / speed / yaw / lat0（只 δ=0）；第一个匹配的规则生效，不匹配任何规则的臂跑全部扰动；不给 = 旧行为）。real 臂恒按 lat δ=0 出行。

条件拼接（与 a_eval.ARSource.rollout 同一约定，仅"目标帧"换成仿真位姿）
  序列 = 真实上下文 C 帧（日志位姿，最后一帧 = 锚点 a）+ 预测的 K 帧（仿真位姿 k=1..K）。
  act_seq[j] = 帧 j → 帧 j+1 的自车相对位姿 [dx,dy,dyaw]，用训练的 `nd.relative_action`（在渲染器的合成帧上算，恒等位姿逐位等于 A 表）。
  lay_seq[j] = 帧 j 的 filled layout（上下文帧 = 日志，目标帧 = `SimLayoutRenderer.render(pose_k)`），j ≥ 1。
  窗口第 i 帧：action = act_seq[w0+i]，layout = lay_seq[w0+i+1]。δ=0 时整个预测序列与 A 表的 `ARSource.rollout` 逐位相同（test_b_eval.py 校验）。

输出：<out>/scene<NNN>.jsonl（每个 (锚点, 扰动, 臂, k) 一行，字段 = G4 的 frame_metrics + lag_*）+ scene<NNN>.done.json。


用法（单卡）：
  CUDA_VISIBLE_DEVICES=2 ../venv/bin/python b_table/b_eval.py --scenes 1 --max-anchors 1 --pred ronomap=ro/runs/full_20261105/train_nomap/iter_6000.pt \
      --arms copy,ronomap,ronomap@zero --lat 0,2 --speed '' --out b_table/runs/smoke

"""
import argparse
import collections
import fnmatch
import json
import os
import socket
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.environ.get('G3_CODE_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, 'g1_warp'), os.path.join(ROOT, 'b_table'), os.path.join(ROOT, 'a_table'), os.path.join(ROOT, 'feat'), ROOT, os.path.join(ROOT, 'uniad_port')):
    if p not in sys.path:
        sys.path.insert(0, p)
from offlog import common  # noqa: E402
from offlog.det_lag_gt import build_polys, lag_match  # noqa: E402
from offlog.g4_probe import LAT_VALUES, SPEED_VALUES, frame_metrics, scene_anchors  # noqa: E402
from offlog.gt_world import GTWorld  # noqa: E402
from offlog.layout_reproject import SimLayoutRenderer  # noqa: E402
from offlog.map_gt import MapGT  # noqa: E402
from offlog.metas import OffLogEnv  # noqa: E402
from offlog.sim_pose import LogTrack  # noqa: E402
from render_paths import image_paths, is_3dgs_arm, parse_render_roots  # noqa: E402
from warp import warp_bev  # noqa: E402
import a_eval  # noqa: E402
import nusc_data as nd  # noqa: E402
from a_eval import ACT_MEAN, ACT_STD, ARSource, CKPTS, STATS  # noqa: E402

ACT_MODES = ('', 'zero', 'log', 'shuf')


def parse_arm(s):
    """'ronomap@zero' → ('ronomap', 'zero')；'ronomap' → ('ronomap', '')；'copy' → ('copy', '')。"""
    name, _, mode = s.partition('@')
    assert mode in ACT_MODES, f'未知 action 模式 {mode!r}（{ACT_MODES}）'
    return name, mode


def nd_actions(frames):
    """相邻合成帧的自车相对位姿 [n-1,3]（训练口径）。frames = nusc_data 风格的帧 dict（要有 'ego2global'）。"""
    if len(frames) < 2:
        return np.zeros((0, 3), np.float32)
    return np.stack([nd.relative_action(a, b) for a, b in zip(frames[:-1], frames[1:])]).astype(np.float32)


def make_act_seq(mode, ctx_act, sim_act, log_act, donor_act):
    """窗口用的 action 序列 [C-1+K,3]（原始单位，未标准化）。"""
    if mode == 'zero':
        return np.zeros((len(ctx_act) + len(sim_act), 3), np.float32)
    sim = {'': sim_act, 'log': log_act, 'shuf': donor_act}[mode]
    return np.concatenate([ctx_act, sim], 0).astype(np.float32)


class SimARSource(ARSource):
    """AR 预测器，条件由调用方给（见模块文档）。窗口 / 标准化 / bf16 autocast 与 ARSource.rollout 逐行一致。"""

    @torch.no_grad()
    def rollout_cond(self, ctx_raw, act_seq, lay_seq):
        """ctx_raw：锚点 a（含）之前最近 ≤ctx 帧的原始 FPN（旧→新）；act_seq：[C-1+K,3] 原始 action；
        lay_seq：长度 C+K 的 list，lay_seq[j]（j≥1）= 帧 j 的 uint8 layout [6,3,H,W]。返回 K 步预测，每步 list[4] 原始特征。"""
        seq = [self.norm(f) for f in ctx_raw]
        C = len(seq)
        K = len(lay_seq) - C
        assert len(act_seq) == C - 1 + K, (len(act_seq), C, K)
        out = []
        for _ in range(K):
            W = min(self.ctx, len(seq))
            win = seq[-W:]
            w0 = len(seq) - W
            feats = [torch.stack([f[l] for f in win]).unsqueeze(0) for l in range(4)]
            act = np.stack([act_seq[w0 + i] for i in range(W)])
            act = torch.from_numpy((act - ACT_MEAN) / ACT_STD).to(self.dev).unsqueeze(0)
            lay = torch.from_numpy(np.stack([lay_seq[w0 + i + 1] for i in range(W)])).to(self.dev).float().unsqueeze(0) / 255.0
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = self.model(feats, act, lay)
            nxt = [x[0, -1].float() for x in z]
            seq.append(nxt)
            out.append(self.denorm(nxt))
        return out


class LayoutCache:
    """上下文（日志）帧的 layout 缓存，与 a_eval.layout_of 同一写法（走训练的 nd.render_layout + 同一份地图向量）。"""

    def __init__(self, rend: SimLayoutRenderer, size=256):
        self.rend, self.size = rend, size
        self.c = collections.OrderedDict()

    def __call__(self, fr, with_map, prev=None):
        key = (fr['token'], with_map) if prev is None else (fr['token'], with_map, prev['token'])   # prev = 关联通道的前一帧（O5）
        if key not in self.c:
            mv = self.rend.map_vecs.for_frame(fr) if with_map else None
            self.c[key] = nd.render_layout(fr, map_vec=mv) if prev is None else nd.render_layout_cond(prev, fr, map_vec=mv, assoc=True)
            while len(self.c) > self.size:
                self.c.popitem(last=False)
        return self.c[key]


def make_pert(tr, env, gtw, data, rend, a, fam, val, horizon):
    """一个 (锚点, 扰动) 的全部与臂无关的量：位姿、GT、UniAD 输入帧、合成 nusc 帧、lag 折线。"""
    poses = tr.rollout(a, fam, val, horizon=horizon)
    by_k = {-2 + i: p for i, p in enumerate(tr.rollout(a - 2, 'lat', 0.0, horizon=2)[:2])}   # 锚点之前：日志位姿
    try:
        ext = tr.rollout(a, fam, val, horizon=horizon + 1)      # 多一步，给 k=horizon 的折线（m=+1）用
    except ValueError:                                          # 速度 ×1.5 时路径可能不够长
        ext = poses
    by_k.update({k: p for k, p in enumerate(ext)})
    gts = [None] + [env.gt(p) for p in poses[1:]]
    return SimpleNamespace(
        fam=fam, val=val, poses=poses, ident=all(p.identity for p in poses), gts=gts,
        frames=[None] + [env.sim_frame(p) for p in poses[1:]],
        sim_nd=[rend.sim_frame(p)[0] for p in poses],
        polys=[None] + [build_polys(gtw, data, by_k, k, gts[k]) for k in range(1, horizon + 1)],
        layouts={})     # (with_map, assoc) → [None, L1..LK]，按需渲染


def row_of(sc, si, a, pt, arm, k, out):
    p = pt.poses[k]
    row = dict(scene=sc['scene_name'], si=si, anchor=a, fam=pt.fam, val=pt.val, arm=arm, k=k, lat=round(p.lat_m, 4),
               dpsi=round(p.dpsi, 5), log_f=round(p.log_f, 3), **frame_metrics(out, pt.gts[k]))
    lm = lag_match(out['boxes'], out['scores'], out['labels'], pt.polys[k])
    row.update(lm if lm is not None else dict(lag_n_gt=-1, lag_n_match=0, lag_s=[], lag_perp=[]))
    return row


def parse_only(spec):
    """--only 'warp:lat,speed,yaw;*:yaw,lat0' → [(glob, {族})]。"""
    rules = []
    for part in (spec or '').split(';'):
        if not part.strip():
            continue
        g, _, fams = part.partition(':')
        fs = {f for f in fams.split(',') if f}
        assert g and fs <= {'lat', 'speed', 'yaw', 'lat0'}, f'--only 规则不合法：{part!r}'
        rules.append((g, fs))
    return rules


def arm_runs(rules, arm, fam, val):
    """该臂是否跑这个扰动（第一个匹配的规则生效；无规则 / 无匹配 = 跑）。"""
    for g, fs in rules:
        if fnmatch.fnmatchcase(arm, g):
            return fam in fs or (fam == 'lat' and val == 0.0 and 'lat0' in fs)
    return True


def parse_scenes(s, n):
    if s == 'all':
        return list(range(n))
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='')
    ap.add_argument('--scenes', default='all', help="'all' | '0,1,5' | '0-9'（UniADValData 场景下标）")
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--anchor-stride', type=int, default=8, help='与 G4 正式探针同（锚点 223 个）')
    ap.add_argument('--min-hist', type=int, default=4)
    ap.add_argument('--min-speed', type=float, default=5.0)
    ap.add_argument('--max-anchors', type=int, default=0)
    ap.add_argument('--horizon', type=int, default=8)
    ap.add_argument('--lat', default=','.join(str(v) for v in LAT_VALUES))
    ap.add_argument('--speed', default=','.join(str(v) for v in SPEED_VALUES))
    ap.add_argument('--yaw', default='', help='航向偏置 ψ（deg，逗号分隔）；空 = 不跑（与 11-06 作业相同）')
    ap.add_argument('--only', default='', help="按臂限定扰动族，如 'warp:lat,speed,yaw;*:yaw,lat0'（见模块文档）")
    ap.add_argument('--pred', action='append', default=[], help='name=path，可重复')
    ap.add_argument('--arms', default='copy', help='见模块文档；预测器臂用 --pred 里的 name，可加 @zero / @log / @shuf')
    ap.add_argument('--identity', action='store_true', help='加 identity 自检臂（只在 δ=0）')
    ap.add_argument('--render-root', default='', help='3dgs 臂的渲染图根目录（见 render_paths.py）')
    ap.add_argument('--render-roots', action='append', default=[], help='tag=path：3dgs_<tag> 臂的渲染图根目录（可重复）')
    ap.add_argument('--anchor-max-scale', type=float, default=1.5,
                    help='锚点筛选要求路径够 ceil(max_scale·horizon) 步；与 --speed 解耦，保证只跑横向扰动的运行（如 3dgs）与主作业锚点一致')
    ap.add_argument('--ctx', type=int, default=4, help='TF ckpt（没有 ctx 字段）的 AR 上下文长度')
    ap.add_argument('--ckpt', default='hf')
    ap.add_argument('--list-scenes', action='store_true', help='只打印有锚点的场景下标（空格分隔）然后退出')
    ap.add_argument('--list-anchors', action='store_true', help='每行打印 "场景下标 锚点数"（有锚点的场景，按锚点数降序）然后退出（distributed execution 入口估计任务耗时用）')
    a = ap.parse_args()

    from uniad_port.data import UniADValData
    from uniad_port.runner import UniADRunner
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    data = UniADValData(num_prefetch=4)
    perts = ([('lat', float(v)) for v in a.lat.split(',') if v] + [('speed', float(v)) for v in a.speed.split(',') if v]
             + [('yaw', float(v)) for v in a.yaw.split(',') if v])
    only = parse_only(a.only)
    roots3d = parse_render_roots(a.render_root, a.render_roots)
    assert ('lat', 0.0) in perts, 'real 臂复用 lat δ=0 这一档的位姿 / GT / 折线，--lat 里必须有 0'
    max_scale = max([v for f, v in perts if f == 'speed'] + [a.anchor_max_scale])
    anchor_cfg = SimpleNamespace(anchor_stride=a.anchor_stride, min_hist=a.min_hist, min_speed=a.min_speed,
                                 max_anchors=a.max_anchors, horizon=a.horizon)
    tracks = {}

    def track(si):
        if si not in tracks:
            tracks[si] = LogTrack(data, data.scenes[si])
        return tracks[si]

    def anchors_of(si):
        an = LogTrack(data, data.scenes[si]).anchors(stride=anchor_cfg.anchor_stride, min_hist=anchor_cfg.min_hist,
                                                     horizon=anchor_cfg.horizon, max_scale=max_scale, min_speed=anchor_cfg.min_speed)
        return an[:anchor_cfg.max_anchors] if anchor_cfg.max_anchors else an

    all_scenes = parse_scenes(a.scenes, len(data.scenes))
    # 全局锚点表 = 全部 val 场景（与 --scenes / --shard 无关）：@shuf 的 donor 取表里错开 N//2 起、第一个不同场景的锚点，
    
    full_with_anchor = [si for si in range(len(data.scenes)) if anchors_of(si)]
    g_anchors = [(si, an) for si in full_with_anchor for an in anchors_of(si)]
    with_anchor = [si for si in all_scenes if si in set(full_with_anchor)]
    if a.list_scenes or a.list_anchors:
        
        order = sorted(with_anchor, key=lambda s: (-len(anchors_of(s)), s))
        if a.list_anchors:
            print('\n'.join(f'{s} {len(anchors_of(s))}' for s in order))
        else:
            print(' '.join(str(s) for s in order))
        return
    assert a.out, '--out 必填'
    my_scenes = with_anchor[a.shard::a.nshard]

    arms = [x for x in a.arms.split(',') if x]
    srcs = {}
    need = {parse_arm(x)[0] for x in arms if parse_arm(x)[0] not in ('copy', 'warp') and not is_3dgs_arm(parse_arm(x)[0])}
    miss3d = [x for x in arms if is_3dgs_arm(x) and x not in roots3d]
    assert not miss3d, f'3DGS 类臂 {miss3d} 没有对应的 --render-root / --render-roots'
    use_warp = 'warp' in arms
    specs = dict(s.split('=', 1) for s in a.pred)
    assert need <= set(specs), f'--arms 里的预测器 {sorted(need - set(specs))} 没有对应的 --pred'
    dev = torch.device('cuda:0')
    st = torch.load(STATS.format(ckpt=a.ckpt))
    assert st.get('scale') == 1.0 and st.get('ckpt') == a.ckpt, f'统计量不是 1.0×/{a.ckpt}'
    mean, std = [m.to(dev) for m in st['mean']], [s.to(dev) for s in st['std']]
    for name in sorted(need):
        srcs[name] = SimARSource(name, specs[name], mean, std, dev, a.ctx)
        s_ = srcs[name]
        print(f'[b] 预测器 {name}: {s_.path} step={s_.step} params={s_.n_params / 1e6:.1f}M map_layout={s_.map_layout} ctx={s_.ctx} assoc={s_.assoc}', flush=True)

    runner = UniADRunner(CKPTS.get(a.ckpt, a.ckpt), device='cuda:0', verbose=False)
    gtw = GTWorld([i['token'] for i in data.infos], cache_path=os.path.join(common.CACHE_DIR, 'gt_world_val.pkl'))
    env = OffLogEnv(data, gtw, MapGT())
    rend = SimLayoutRenderer(data.infos)
    layout_ctx = LayoutCache(rend)
    os.makedirs(a.out, exist_ok=True)
    K = a.horizon
    t_roll, t_uniad = collections.defaultdict(list), []
    t0 = time.time()

    def donor_actions(si, anchor, fam, val):
        """@shuf：另一个**场景**的锚点在同一扰动下的仿真 action [K,3]（表里错开 N//2 起、第一个不同场景的锚点，确定性）。"""
        i = g_anchors.index((si, anchor))
        n = len(g_anchors)
        for off in range(n // 2, n):
            sj, aj = g_anchors[(i + off) % n]
            if sj != si:
                break
        else:
            raise ValueError('@shuf 需要全部 val 里至少两个有锚点的场景（不应发生）')
        pj = track(sj).rollout(aj, fam, val, horizon=K)
        return nd_actions([rend.sim_frame(p)[0] for p in pj])

    for si in my_scenes:
        tag = f'scene{si:03d}'
        path, done_path = os.path.join(a.out, tag + '.jsonl'), os.path.join(a.out, tag + '.done.json')
        if os.path.exists(done_path):
            print('已完成，跳过', done_path)
            continue
        sc = data.scenes[si]
        tr = track(si)
        anchors = anchors_of(si)
        last = max(anchors) + K
        # ---- pass 1：全场景抽 FPN（GPU fp32，≈7.6 GB/40 帧；与 a_eval 同）----
        runner.reset_scene()
        metas, feats_all = [], []
        for j, fr in enumerate(data.iter_scene(si)):
            if j > last:
                break
            feats_all.append(runner.extract_feats(fr['img']))
            metas.append({k_: v for k_, v in fr.items() if k_ != 'img'})
        nd_frames = [rend.nd_frame(i) for i in tr.ids[:len(metas)]]
        # ---- pass 2：真图主流式；走到锚点时分叉 ----
        runner.reset_scene()
        real_out, lat0 = {}, {}
        n_rows = 0
        fout = open(path + '.tmp', 'w')
        for j in range(last + 1):
            out = runner.step_frame(feats_all[j], metas[j], return_bev=(use_warp and j in anchors))
            bev_a = out.pop('bev_embed', None)                         # warp 臂：锚点的真实 BEV（return_bev 只多一份 clone，不改输出与状态）
            if j > min(anchors):
                real_out[j] = out
            if j not in anchors:
                continue
            st_a = runner.get_state()
            chk = [float(f.double().sum()) for f in feats_all[j]]
            lo = {n_: max(0, j - s_.ctx + 1) for n_, s_ in srcs.items()}
            for fam, val in perts:
                pt = make_pert(tr, env, gtw, data, rend, j, fam, val, K)
                if (fam, val) == ('lat', 0.0):
                    lat0[j] = pt
                for arm in arms + (['identity'] if (a.identity and pt.ident) else []):
                    name, mode = parse_arm(arm) if arm != 'identity' else ('identity', '')
                    if arm != 'identity' and not arm_runs(only, arm, fam, val):
                        continue
                    bevs = None
                    if name == 'identity':
                        seq = [feats_all[j + k] for k in range(1, K + 1)]
                    elif name == 'copy':
                        seq = [feats_all[j]] * K
                    elif name == 'warp':
                        torch.cuda.synchronize()
                        tp = time.perf_counter()
                        bevs = [warp_bev(bev_a, metas[j]['l2g_r_mat'], metas[j]['l2g_t'], pt.frames[k]['l2g_r_mat'], pt.frames[k]['l2g_t'])
                                for k in range(1, K + 1)]                # 每步都从真锚点直接 warp（不递归），视野外填 0
                        torch.cuda.synchronize()
                        t_roll[arm].append(time.perf_counter() - tp)
                    elif is_3dgs_arm(name):
                        if fam not in ('lat', 'yaw'):
                            continue                                    # 没有速度倍率的渲染
                        seq = []
                        for k in range(1, K + 1):
                            paths = image_paths(roots3d[name], sc['scene_name'], j, val, k, fam)
                            miss = [p for p in paths if not os.path.exists(p)]
                            assert not miss, f'缺渲染图（先跑 gen_render_requests + render_requests）：{miss[:2]} …共 {len(miss)} 张'
                            torch.cuda.synchronize()
                            tp = time.perf_counter()
                            img, _, _ = data._load_images(paths)            # 与真图完全同一条加载 / 归一化 / 补边路径
                            seq.append(runner.extract_feats(img))
                            torch.cuda.synchronize()
                            t_roll[arm].append(time.perf_counter() - tp)    # 3dgs 的"预测"耗时 = 读图 + 主干（渲染本身另计，见 splatad 日志）
                    else:
                        s_ = srcs[name]
                        wm, asc = s_.map_layout, s_.assoc
                        if (wm, asc) not in pt.layouts:
                            # O5 关联通道：第 k 步的前一帧 = 第 k-1 步的位姿（k=1 时 = 锚帧）；框取日志 GT（其他 agent 按日志回放）
                            pt.layouts[(wm, asc)] = [None] + [rend.render_cond(pt.poses[k - 1], pt.poses[k], wm, assoc=True) if asc
                                                              else rend.render(pt.poses[k], with_map=wm) for k in range(1, len(pt.poses))]
                        ctx_nd = nd_frames[lo[name]:j + 1]
                        C = len(ctx_nd)
                        sim_act = nd_actions(pt.sim_nd)
                        log_act = nd_actions([rend.nd_frame(p.tmpl_idx) for p in pt.poses]) if mode == 'log' else None
                        donor = donor_actions(si, j, fam, val) if mode == 'shuf' else None
                        act_seq = make_act_seq(mode, nd_actions(ctx_nd), sim_act, log_act, donor)
                        lay_seq = ([None] + [layout_ctx(ctx_nd[i], wm, prev=ctx_nd[i - 1] if asc else None) for i in range(1, C)]
                                   + pt.layouts[(wm, asc)][1:])
                        torch.cuda.synchronize()
                        tp = time.perf_counter()
                        seq = s_.rollout_cond(feats_all[lo[name]:j + 1], act_seq, lay_seq)
                        torch.cuda.synchronize()
                        t_roll[arm].append(time.perf_counter() - tp)
                    runner.set_state(st_a)
                    for k in range(1, K + 1):
                        ts = time.perf_counter()
                        if bevs is not None:
                            o = runner.step_frame_bev(bevs[k - 1], pt.frames[k])
                        else:
                            o = runner.step_frame(seq[k - 1], pt.frames[k] if name != 'identity' else metas[j + k])
                        t_uniad.append(time.perf_counter() - ts)
                        fout.write(json.dumps(row_of(sc, si, j, pt, arm, k, o)) + '\n')
                        n_rows += 1
            runner.set_state(st_a)
            assert [float(f.double().sum()) for f in feats_all[j]] == chk, '锚点特征被原地修改'
        # real 臂：锚点 a 的第 k 步 = 主流式在 a+k 帧的输出（位姿 = 日志，与 lat δ=0 同一个 pert）
        for an in anchors:
            for k in range(1, K + 1):
                fout.write(json.dumps(row_of(sc, si, an, lat0[an], 'real', k, real_out[an + k])) + '\n')
                n_rows += 1
        fout.close()
        os.replace(path + '.tmp', path)
        with open(done_path, 'w') as f:
            json.dump(dict(args=vars(a), scene=si, anchors=anchors, rows=n_rows, seconds=time.time() - t0, render_roots=roots3d,
                           host=socket.gethostname(), convention=runner.convention,
                           preds={n_: dict(path=s_.path, step=s_.step, ctx=s_.ctx, map_layout=s_.map_layout, assoc=s_.assoc) for n_, s_ in srcs.items()},
                           sec_rollout={k_: float(np.mean(v)) for k_, v in t_roll.items()},
                           sec_uniad_step=float(np.mean(t_uniad)) if t_uniad else None), f, indent=1)
        print(f'[b {a.shard}/{a.nshard}] scene {si} {sc["scene_name"]} anchors={anchors} rows={n_rows} '
              f'elapsed={time.time() - t0:.0f}s', flush=True)
        del feats_all, real_out


if __name__ == '__main__':
    main()
