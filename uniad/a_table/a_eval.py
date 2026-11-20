"""A 表：沿日志对齐的 AR 评测（设计稿 §4；复现 FlowSim Tab. I / Tab. III 的 "AR" 行同口径）。

协议（与 FlowSim 一致处标 [F]，我们的选择标 [本文]）：
  - 参照分支：UniAD 吃真图的整段流式输出，每个目标帧 t 一份 ``ref[t]``。
  - 被测分支：在锚点 a 之前与参照**共享真实历史**（UniAD 的 prev_bev / track query / memory bank 都在真图上建好，预测器的 L 帧上下文也是真实特征）。
    从 a+1 起每一帧都喂**预测的** FPN，自回归 k=1..8 步；每步的条件（自车 action、GT 框 layout、可选地图）取日志 [F]。
  - 只在 a+k 恰为 Δt∈{0.5, 1.5, 2.5, 4.0} s ↔ k∈{1, 3, 5, 8} 个关键帧（nuScenes 2 Hz）时与 ref[a+k] 比 ``frame_agreement`` [F]。
  - 每个锚点滚一次 8 步，同时得到四个 Δt 的结果；目标帧 t 在不同 Δt 下由不同的锚点（a=t−k）提供，**所以同一帧 t 有四条独立的滚动**。
  - 帧集合 = G1 的 geom 口径：8 ≤ t ≤ n−7（与 Tab. I 同；锚点 a=t−8 ≥ 0 保证 8 步链都在场景内）。**所有臂、所有 Δt 在同一批 t 上评**。
    refdet 口径（再要求 ref 在 ±50 m 内有 score≥0.3 的框）在合并时再取。
  - 被测分支的 UniAD 状态 [本文]：a+1 起 UniAD 的 prev_bev / track / memory **也由生成特征推进**（锚点之前是真实历史，之后全是生成帧），
    对应 FlowSim 的 "AR, 生成历史"，也是闭环部署的真实口径。结构上 = G1 的 **V1**（从真状态 S_a 出发，中间每帧都喂 source，流式走到 t），
    所以 Warp 行取 G1 的 ``('warp','V1',t,k)``；**不是** G1 的 V3（V3 真图历史走到 t−1、只换最后一帧，= G3 的单步口径，是 G1 对齐 FlowSim Tab. I 的那个协议）
    **也不是** V4（V4 整段场景从头全程喂 source、每帧都从 Δt 前的真锚点单独预测）。
    区别只有一点：Warp 每步都从真锚点直接 warp（不递归），我们的预测器是递归 AR，这正是 AR 对 VH 型 baseline 要回答的问题。

臂（``--arm``，可重复）：
  copy_last     把锚点 a 的真实 FPN 整帧复制 k 次（切在 FPN，≠ G1 的 BEV Warp）
  pred:<name>   ``name=ckpt`` 的预测器（G3/RO 训练出来的 PyramidPredictor），AR：每步把上一步预测拼进上下文，窗口滑动（与训练的 sliding 口径一致：
                 窗口恒长 ctx，丢最旧、拼最新预测；ctx 取 ckpt 里的 ``ctx``，没有（G3 的 TF ckpt）用 ``--ctx``）
                 name 以 ``@det`` 结尾时 layout 里的框换成 ``--det-boxes`` 的检测器框（feat/det_layout.py；地图、action 不变），审稿 W4 的非特权条件
  identity      真特征（自检：整个流程应与参照逐位相同）
  3dgs/V1 …     SplatAD 重建的**日志位姿**渲染图（--render-root；目录约定见 b_table/render_paths.py 的 log_image_paths）→ 与真图同一条加载路径 → 主干 → FPN。
                三种 UniAD 状态处理（11-07，SplatAD 校准用；渲染图只依赖目标帧、与锚点无关）：
                  3dgs/V1  与预测器同协议：真历史到锚点 a，a+1..a+k 每帧都喂渲染图（= "AR, 生成历史"，k 依赖）；
                  3dgs/V3  真图流走到 t−1、只把 t 帧换成渲染图（= FlowSim Tab. I 的 "SplatAD (gt hist.)" 行、G1 的 V3）；与 k 无关，各 Δt 填同一个 t 的值；
                  3dgs/V4  整段场景从第 0 帧起全程喂渲染图（= "SplatAD (gen. hist.)"：论文对 ≥1.5 s 重复同一行，说明与 Δt 无关）；各 Δt 填同一个 t 的值。
                --render-roots tag=path 加 3dgs_<tag>/V1|V3|V4（如 Difix 后处理）；path = '@nusc' 时读 nuScenes 原图
                （自检：/V1 必须与 identity、/V3 /V4 必须与参照逐位相同，结果打印 [3dgs real selfcheck]）。
Warp 行不在这里跑，由 g1_warp 的 V1 / V3 产物在 a_merge.py 里按 sample_token 取（V1 = 同状态协议的行；V3 = 对齐 FlowSim Tab. I 的校准行）。

用法：
  CUDA_VISIBLE_DEVICES=2 ../venv/bin/python a_table/a_eval.py --scenes 0 --max-anchors 3 --pred ro=ro/runs/full_20261105/train_full/iter_6000.pt --out a_table/runs/smoke

输出 <out>/shard{i}of{n}.pkl：{arm: {k: {metric: array[n_pairs]}}}（逐 (锚点, 目标帧) 一行）+ meta（场景、目标帧 token、锚点 token、自车位移）。
"""
import featsim_paths as _release_paths
import argparse
import collections
import os
import pickle
import sys
import time

import numpy as np
import torch

ROOT = os.environ.get('G3_CODE_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'uniad_port'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'feat'))
sys.path.insert(0, os.path.join(ROOT, 'b_table'))
from uniad_port.runner import UniADRunner  # noqa: E402
from uniad_port.data import UniADValData  # noqa: E402
from agreement.agreement import AgreementConfig, frame_agreement, HEADLINE_KEYS  # noqa: E402
import nusc_data as nd  # noqa: E402
from clip_dataset import ACT_MEAN, ACT_STD  # noqa: E402
from pyramid_predictor import load_predictor, trim_kv_cache, uses_assoc_layout  # noqa: E402
from render_paths import NUSC_REAL, log_image_paths, parse_render_roots  # noqa: E402

CKPTS = {
    'gh': _release_paths.CKPT_GH,
    'hf': _release_paths.CKPT_HF,
}
STATS = str(_release_paths.Path(_release_paths.BENCH) / 'feat/level_stats_1x_{ckpt}.pt')
KS = (1, 3, 5, 8)           # Δt = 0.5 / 1.5 / 2.5 / 4.0 s
KMAX = max(KS)
T_MIN, T_END_MARGIN = 8, 6  # geom：8 ≤ t ≤ n−7
KEEP_KEYS = HEADLINE_KEYS + ('det_box_iou', 'det_recall', 'det_precision', 'det_n_ref', 'det_n_src', 'det_n_match', 'motion_n_pairs')


PROTOS = ('V1', 'V3', 'V4')


def scene_list(data, scene_set, scenes):
    if scenes:                                  # 下标或场景名（名字换成下标：a_merge 按下标取 G1 产物）
        return [int(s) if s.isdigit() else data._name2scene[s] for s in scenes.split(',') if s]
    if scene_set == 'every5':
        return list(range(0, len(data.scenes), 5))
    if scene_set == 'all':
        return list(range(len(data.scenes)))
    raise ValueError(scene_set)


class ARSource:
    """AR 预测器：给定锚点处的真实上下文（最近 ≤ctx 帧原始 FPN），逐步预测 a+1..a+K。

    每步：上下文 = 当前序列（真实 + 已预测）的最后 ctx 帧（序列不足 ctx 帧时用全部，随预测追加逐步长到 ctx）；
    条件 = 窗口内各帧 → 下一帧的 action 与目标帧 layout（日志）。预测前把特征转成标准化空间，预测后转回原始空间。
    窗口里的已预测帧直接用预测器自己的（标准化空间）输出，与训练的 sliding_rollout 一致（见 test_a_eval.py 的逐位对拍）。

    推理选项（11-10，默认 = 原路径逐位不变）：
      dtype='bf16'  fp32 权重 + bf16 autocast（原口径）；'fp16' = 权重转 half、输入转 half、不用 autocast（warp 的 grid_sample 仍在 fp32）；'fp32' = 全精度参照
      kv=True       ``rollout_kv``：锚点处对真实窗口做一次稠密前向并建 KV cache，之后每步只算新的一帧（缓存只留最近 ctx-1 帧）。
                    **缓存淘汰后与稠密滑窗不等价**（见 PyramidPredictor.forward_step 的 docstring），所以它是近似，质量要看下游指标。"""
    dtype_name, kv, t_model = 'bf16', False, None    # 类属性默认值：用 __new__ 手工搭的测试对象走原路径、不计时

    def __init__(self, name, path, mean, std, dev, ctx_default, dtype='bf16', kv=False):
        self.name, self.path = name, path
        self.model, ck = load_predictor(path, dev)
        assert dtype in ('bf16', 'fp16', 'fp32'), dtype
        self.dtype_name, self.kv, self.t_model = dtype, bool(kv), []
        if dtype == 'fp16':
            self.model = self.model.half()
        self.step = ck.get('step')
        self.map_layout = bool(ck['args'].get('map_layout', False))
        self.assoc = uses_assoc_layout(ck)          # O5 关联通道：layout 6 通道，后 3 通道 = 窗口里前一帧（日志 GT 框）画进目标帧相机
        self.ctx = int(ck['args']['ctx']) if ck['args'].get('rollout') == 'sliding' else ctx_default
        self.det = name.endswith('@det')            # layout 的框取检测器（调用方传入检测器索引的帧）
        self.mean, self.std, self.dev = mean, std, dev
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def norm(self, fs):
        return [(f - self.mean[l].view(1, -1, 1, 1)) / self.std[l].view(1, -1, 1, 1) for l, f in enumerate(fs)]

    def denorm(self, fs):
        return [f * self.std[l].view(1, -1, 1, 1) + self.mean[l].view(1, -1, 1, 1) for l, f in enumerate(fs)]

    def _model_call(self, fn, *args):
        """在 bf16 autocast / fp16 权重 / fp32 下调 ``fn(*args)``（args 里的张量或张量 list 按 fp16 转 half），并把这次调用的 GPU 时间（前后 sync）记进 t_model。"""
        if self.dtype_name == 'fp16':
            args = tuple([t.half() for t in x] if isinstance(x, list) else x.half() for x in args)
        if self.t_model is not None:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if self.dtype_name == 'bf16':
            with torch.autocast('cuda', dtype=torch.bfloat16):
                r = fn(*args)
        else:
            r = fn(*args)
        if self.t_model is not None:
            torch.cuda.synchronize()
            self.t_model.append(time.perf_counter() - t0)
        return r

    def _cond(self, nds, i0, n, layout_of):
        """nds 里帧 i0..i0+n-1 的条件：action = 帧 i→i+1（标准化）、layout = 目标帧 i+1（关联通道 ckpt 另带前一帧 i）。返回 ([1,n,3], [1,n,N,C,H,W])。"""
        act = np.stack([nd.relative_action(nds[i], nds[i + 1]) for i in range(i0, i0 + n)])
        act = torch.from_numpy((act - ACT_MEAN) / ACT_STD).to(self.dev).unsqueeze(0)
        if self.assoc:                                                    # 关联通道的"前一帧" = 窗口里帧 i（即使它的特征是预测的，框也取日志 GT）
            lays = [layout_of(nds[i + 1], self.map_layout, prev=nds[i]) for i in range(i0, i0 + n)]
        else:
            lays = [layout_of(nds[i + 1], self.map_layout) for i in range(i0, i0 + n)]
        lay = torch.from_numpy(np.stack(lays)).to(self.dev).float().unsqueeze(0) / 255.0
        return act, lay

    def prewarm(self, layout_of, nds):
        """把 nds[1:] 的 layout 渲染进调用方的缓存，让各臂的计时不含"谁先渲染谁吃 CPU"的偏差。"""
        for i in range(len(nds) - 1):
            layout_of(nds[i + 1], self.map_layout, prev=nds[i]) if self.assoc else layout_of(nds[i + 1], self.map_layout)

    @torch.no_grad()
    def rollout(self, ctx_raw, ctx_nd, tgt_nd, layout_of):
        """ctx_raw: 锚点 a（含）之前最近 ≤ctx 帧的原始 FPN list[4] of [6,256,h,w]（旧→新）；ctx_nd: 对应的 nusc 帧 dict；
        tgt_nd: 锚点之后要预测的各帧（长度 = 要滚的步数 K）。返回 K 步预测，每步 list[4] 原始特征。"""
        if self.kv:
            return self.rollout_kv(ctx_raw, ctx_nd, tgt_nd, layout_of)
        seq = [self.norm(f) for f in ctx_raw]                             # list over frames of list[4]（标准化）
        nds = list(ctx_nd) + list(tgt_nd)                                 # nds[i] ↔ seq[i]；i ≥ len(ctx) 的是待预测帧
        out = []
        for _ in range(len(tgt_nd)):
            W = min(self.ctx, len(seq))
            win = seq[-W:]
            w0 = len(seq) - W                                             # 窗口第一帧在 nds 里的下标
            feats = [torch.stack([f[l] for f in win]).unsqueeze(0) for l in range(4)]           # [1,W,6,256,h,w]
            act, lay = self._cond(nds, w0, W, layout_of)
            z = self._model_call(self.model, feats, act, lay)
            nxt = [x[0, -1].float() for x in z]                           # 帧 len(seq) 的预测（标准化空间）
            seq.append(nxt)
            out.append(self.denorm(nxt))
        return out

    @torch.no_grad()
    def rollout_kv(self, ctx_raw, ctx_nd, tgt_nd, layout_of):
        """``rollout`` 的 KV cache 版：真实窗口（≤ctx 帧）一次稠密前向建缓存，得到第 1 步预测；之后每步只前向"上一步预测出的那一帧"，
        对缓存里的历史帧做注意力，缓存只留最近 ctx-1 帧（新帧 + 缓存 = 与稠密滑窗相同的 ctx 帧上下文长度）。
        第 1 步（缓存未淘汰）与稠密路径只差浮点误差；第 ctx 步起缓存里的旧帧"多看了"已淘汰的更早历史，不再等价。"""
        seq = [self.norm(f) for f in ctx_raw]
        nds = list(ctx_nd) + list(tgt_nd)
        W0 = len(seq)
        assert W0 <= self.ctx, (W0, self.ctx)
        feats = [torch.stack([f[l] for f in seq]).unsqueeze(0) for l in range(4)]
        act, lay = self._cond(nds, 0, W0, layout_of)
        z, cache = self._model_call(lambda f, a, l: self.model(f, a, l, return_kv_cache=True), feats, act, lay)
        nxt = [x[0, -1].float() for x in z]
        out = [self.denorm(nxt)]
        for i in range(W0, W0 + len(tgt_nd) - 1):                         # i = 本步的新帧（上一步的预测）在 nds 里的下标，也是它的 RoPE 时间位置
            act, lay = self._cond(nds, i, 1, layout_of)
            cache = trim_kv_cache(cache, self.ctx - 1)
            z, cache = self._model_call(lambda f, a, l: self.model.forward_step(f, a, l, pos=i, kv_cache=cache), [x[None, None] for x in nxt], act, lay)
            nxt = [x[0, 0].float() for x in z]
            out.append(self.denorm(nxt))
        return out


def ref_gt_plan_error(out, fr):
    """参照分支（UniAD 吃真图）对 GT 规划轨迹的 ADE/FDE（6 个路点 = 3 s，逐步 L2 取均值/末步；GT 不足 6 步的帧记 NaN）。给 A 表一个"满分一致性的尺度"。"""
    m = np.asarray(fr['sdc_planning_mask']).astype(bool)[:6]
    if not m.all():
        return np.nan, np.nan
    e = np.linalg.norm(np.asarray(out['plan'], np.float64)[:6, :2] - np.asarray(fr['sdc_planning'], np.float64)[:6, :2], axis=-1)
    return float(e.mean()), float(e[-1])


def valid_ks(a_idx, n):
    """锚点 a 能打分的 k：目标帧 a+k 必须在 geom 帧集合 [T_MIN, n-1-T_END_MARGIN]（与 G1 一致）。"""
    return [k for k in KS if T_MIN <= a_idx + k <= n - 1 - T_END_MARGIN]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='hf')
    ap.add_argument('--scene-set', default='every5', choices=['every5', 'all'])
    ap.add_argument('--scenes', default='')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--pred', action='append', default=[], help='name=path，可重复')
    ap.add_argument('--pred-opt', action='append', default=[],
                    help="name=dtype:bf16|fp16|fp32,kv:0|1：给 --pred 的预测器加推理选项（缺省 = bf16 autocast + 稠密滑窗 = 原路径）；"
                         "同一个 ckpt 想跑多种推理方式就用不同的 name 各给一份 --pred")
    ap.add_argument('--ctx', type=int, default=4, help='TF ckpt（没有 ctx 字段）的 AR 上下文长度')
    ap.add_argument('--no-baselines', action='store_true', help='不跑 identity / copy_last')
    ap.add_argument('--max-anchors', type=int, default=0, help='每场景最多评多少个锚点（0=全部；冒烟用，取等间隔的）')
    ap.add_argument('--render-root', default='', help='3dgs 臂：日志位姿渲染图根目录（<root>/<scene>/shift_+0.0/<CAM>/<stem>.jpg）')
    ap.add_argument('--render-roots', action='append', default=[], help="tag=path：3dgs_<tag> 臂（可重复；path='@nusc' = 原图，自检用）")
    ap.add_argument('--det-boxes', default='', help='feat/det_layout.py 的 det_boxes.pkl：名字以 @det 结尾的预测器用它的框画 layout')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    torch.backends.cudnn.benchmark = False
    dev = torch.device('cuda:0')
    runner = UniADRunner(CKPTS.get(a.ckpt, a.ckpt), device='cuda:0', verbose=(a.shard == 0))
    data = UniADValData(num_prefetch=4)
    cfg = AgreementConfig(motion_traj_kind='relative')
    st = torch.load(STATS.format(ckpt=a.ckpt))
    assert st.get('scale') == 1.0 and st.get('ckpt') == a.ckpt, f'统计量不是 1.0×/{a.ckpt}'
    mean, std = [m.to(dev) for m in st['mean']], [s.to(dev) for s in st['std']]
    idx = nd.build_index()
    lay_cache = collections.OrderedDict()
    map_vecs = []

    def layout_of(fr, with_map=False, prev=None):
        key = (fr['token'], with_map) if prev is None else (fr['token'], with_map, prev['token'])   # prev = 关联通道的前一帧（O5）
        if fr.get('box_src', 'gt') != 'gt':                                # 检测器框的帧与 GT 帧同 token，缓存键要分开
            key = key + (fr['box_src'],)
        if key not in lay_cache:
            mv = None
            if with_map:
                if not map_vecs:
                    from nusc_map import MapVectors
                    map_vecs.append(MapVectors())
                mv = map_vecs[0].for_frame(fr)
            lay_cache[key] = nd.render_layout(fr, map_vec=mv) if prev is None else nd.render_layout_cond(prev, fr, map_vec=mv, assoc=True)
            while len(lay_cache) > 256:
                lay_cache.popitem(last=False)
        return lay_cache[key]

    popts = {}
    for spec in a.pred_opt:
        name, o = spec.split('=', 1)
        d = dict(kv.split(':', 1) for kv in o.split(',') if kv)
        assert set(d) <= {'dtype', 'kv'}, f'--pred-opt {spec}: 只认 dtype / kv'
        popts[name] = dict(dtype=d.get('dtype', 'bf16'), kv=bool(int(d.get('kv', 0))))
    assert set(popts) <= {s.split('=', 1)[0] for s in a.pred}, f'--pred-opt 里有没给 --pred 的名字: {set(popts) - {s.split("=", 1)[0] for s in a.pred}}'
    srcs = {}
    for spec in a.pred:
        name, path = spec.split('=', 1)
        srcs[f'pred:{name}'] = ARSource(name, path, mean, std, dev, a.ctx, **popts.get(name, {}))
        s_ = srcs[f'pred:{name}']
        print(f'[a] 预测器 {name}: {path} step={s_.step} params={s_.n_params / 1e6:.1f}M map_layout={s_.map_layout} ctx={s_.ctx} assoc={s_.assoc} det={s_.det} dtype={s_.dtype_name} kv={s_.kv}', flush=True)
    idx_det = None
    if any(s_.det for s_ in srcs.values()):
        import det_layout
        assert a.det_boxes, '有 @det 预测器但没给 --det-boxes'
        assert not any(s_.det and s_.assoc for s_ in srcs.values()), '@det 不支持关联通道 ckpt'
        idx_det = det_layout.load_det_index(a.det_boxes, idx)
        print(f'[a] 检测器框：{idx_det["det_meta"]}', flush=True)
    roots3d = parse_render_roots(a.render_root, a.render_roots)
    for g, r in roots3d.items():
        print(f'[a] 3DGS 来源 {g}: {r}', flush=True)
    names = ([] if a.no_baselines else ['identity', 'copy_last']) + list(srcs) + [f'{g}/{p}' for g in roots3d for p in PROTOS]
    assert names, '没有任何来源'
    my_scenes = scene_list(data, a.scene_set, a.scenes)[a.shard::a.nshards]

    # rec[arm][k][metric] / meta[k][field]：每个 (锚点, 目标帧) 一行；同一个 k 下各臂的行顺序相同
    rec = {n: {k: {m: [] for m in KEEP_KEYS} for k in KS} for n in names}
    META_FIELDS = ('scene', 'target_token', 'anchor_token', 'frame_idx', 'ego_disp', 'ego_disp_ak', 'ctx_len')
    meta = {k: {f: [] for f in META_FIELDS} for k in KS}   # 显式初始化：某个 k 在本分片没有样本时也要有空数组（合并时按字段拼接）
    ref_info = dict(sample_token=[], scene=[], frame_idx=[], n=[], gt_plan_ade=[], gt_plan_fde=[], det_n_ref=[], ref_n_boxes=[])
    t_roll, t_step, t0 = collections.defaultdict(list), [], time.time()
    tlog = collections.defaultdict(lambda: dict(K=[], sec=[], sec_model=[]))   # 每次 rollout：滚的步数 / 总耗时（含条件构造）/ 其中模型 GPU 耗时
    for si, sc in enumerate(my_scenes):
        # ---- pass 1：真图整段流式一遍。存参照输出 ref[t]、每帧原始 FPN（GPU fp32，约 7.6 GB/40 帧；fp16 会破坏 identity 逐位自检）----
        runner.reset_scene()
        frames, feats_all, refs = [], [], []
        for fr in data.iter_scene(sc):
            f = runner.extract_feats(fr['img'])
            refs.append(runner.step_frame(f, fr))
            feats_all.append(f)
            frames.append({k: v for k, v in fr.items() if k != 'img'})
        n = len(frames)
        scene_name = frames[0]['scene_name']
        # ---- 3DGS：每帧的渲染图 → FPN（与真图同一条 _load_images 路径）；V4 = 整段只喂渲染图的流式 ----
        geom_t = range(T_MIN, n - T_END_MARGIN)
        rfeats, v3, v4 = {}, {}, {}
        for g, root in roots3d.items():
            rf = []
            for t in range(n):
                paths = log_image_paths(root, scene_name, frames[t]['img_metas']['filename'])
                miss = [p for p in paths if not os.path.exists(p)]
                assert not miss, f'{g} 缺渲染图：{miss[:2]} …共 {len(miss)} 张'
                torch.cuda.synchronize()
                tp = time.perf_counter()
                img, _, _ = data._load_images(paths)
                rf.append(runner.extract_feats(img))
                torch.cuda.synchronize()
                t_roll[g].append(time.perf_counter() - tp)
            rfeats[g] = rf
            runner.reset_scene()
            v4[g] = {}
            for t in range(n):
                o = runner.step_frame(rf[t], frames[t])
                if t in geom_t:
                    v4[g][t] = frame_agreement(refs[t], o, cfg)
            v3[g] = {}
        nd_frames = [idx['frames'][fr['sample_token']] for fr in frames]
        assert all(nf['scene'] == scene_name and nf['idx'] == i for i, nf in enumerate(nd_frames)), f'{scene_name}: nusc 帧序与 UniAD 帧序不一致'
        nd_frames_det = None
        if idx_det is not None:                                            # 漏帧直接报错（不静默回退到 GT 框）
            nd_frames_det = [idx_det['frames'][fr['sample_token']] for fr in frames]
            assert all(nf.get('box_src') == 'det' for nf in nd_frames_det), f'{scene_name}: 检测器缓存缺帧'
        for t in range(n):
            ade, fde = ref_gt_plan_error(refs[t], frames[t])
            rr = frame_agreement(refs[t], refs[t], cfg)
            ref_info['sample_token'].append(frames[t]['sample_token']); ref_info['scene'].append(scene_name)
            ref_info['frame_idx'].append(int(frames[t]['frame_idx'])); ref_info['n'].append(n)
            ref_info['gt_plan_ade'].append(ade); ref_info['gt_plan_fde'].append(fde)
            ref_info['det_n_ref'].append(float(rr['det_n_ref'])); ref_info['ref_n_boxes'].append(int((refs[t]['scores'] >= cfg.score_thr).sum()))
        # ---- pass 2：重新推进真图流，走到每个锚点 a 时分叉 ----
        anchors = [a_ for a_ in range(n) if valid_ks(a_, n)]
        if a.max_anchors and len(anchors) > a.max_anchors:
            anchors = [anchors[i] for i in np.unique(np.linspace(0, len(anchors) - 1, a.max_anchors).round().astype(int))]
        anchor_set = set(anchors)
        runner.reset_scene()
        lt = lambda i: np.asarray(frames[i]['l2g_t'], np.float64).reshape(3)
        for t in range(n):
            if roots3d and t in geom_t:                                    # 3dgs/V3：真图流已走到 t−1，只把 t 帧换成渲染图走一步，再恢复
                st_p = runner.get_state()
                for g in roots3d:
                    v3[g][t] = frame_agreement(refs[t], runner.step_frame(rfeats[g][t], frames[t]), cfg)
                    runner.set_state(st_p)
            runner.step_frame(feats_all[t], frames[t])                     # 真图流先推进到 t（含 t 帧），再决定要不要在 t 处分叉
            if t in anchor_set:
                vk = valid_ks(t, n)
                K = max(vk)                                                # 只滚到最后一个要打分的 k
                st_a = runner.get_state()                                  # 锚点 a=t：已推进到 a（含 a 帧）的真图流状态
                for nme in names:
                    if nme == 'identity':
                        seq = [feats_all[t + k] for k in range(1, K + 1)]
                    elif nme == 'copy_last':
                        seq = [feats_all[t]] * K
                    elif nme.endswith('/V3') or nme.endswith('/V4'):
                        continue                                           # 与锚点无关，场景末尾按行序补
                    elif nme.endswith('/V1'):
                        seq = [rfeats[nme[:-3]][t + k] for k in range(1, K + 1)]
                    else:
                        s = srcs[nme]
                        lo = max(0, t - s.ctx + 1)
                        nfs = nd_frames_det if s.det else nd_frames                # ego2global 两者相同，只有框不同
                        s.prewarm(layout_of, nfs[lo:t + 1 + K])                    # layout 渲染不计入各臂耗时（否则第一个臂吃掉所有缓存未命中）
                        torch.cuda.synchronize()
                        n_m0 = len(s.t_model)
                        tp = time.perf_counter()
                        seq = s.rollout(feats_all[lo:t + 1], nfs[lo:t + 1], nfs[t + 1:t + 1 + K], layout_of)
                        torch.cuda.synchronize()
                        t_roll[nme].append(time.perf_counter() - tp)
                        tlog[nme]['K'].append(K); tlog[nme]['sec'].append(t_roll[nme][-1]); tlog[nme]['sec_model'].append(float(sum(s.t_model[n_m0:])))
                    runner.set_state(st_a)
                    for k in range(1, K + 1):
                        ts = time.perf_counter()
                        out = runner.step_frame(seq[k - 1], frames[t + k])   # UniAD 状态也由生成特征推进（G1 的 V1 同协议 / "AR, 生成历史"）
                        t_step.append(time.perf_counter() - ts)
                        if k in vk:
                            r = frame_agreement(refs[t + k], out, cfg)
                            for m in KEEP_KEYS:
                                rec[nme][k][m].append(float(r.get(m, np.nan)))
                runner.set_state(st_a)                                     # 回到真图流
                for k in vk:
                    M = meta[k]
                    M['scene'].append(scene_name); M['target_token'].append(frames[t + k]['sample_token'])
                    M['anchor_token'].append(frames[t]['sample_token']); M['frame_idx'].append(int(frames[t + k]['frame_idx']))
                    M['ego_disp'].append(float(np.linalg.norm(lt(t + k)[:2] - lt(t + k - 1)[:2])))   # 目标帧自己那一步（与 G3 同口径）
                    M['ego_disp_ak'].append(float(np.linalg.norm(lt(t + k)[:2] - lt(t)[:2])))       # 锚点 → 目标帧
                    M['ctx_len'].append(min(max(s_.ctx for s_ in srcs.values()) if srcs else 1, t + 1))
        for t_a in anchors:                                                # 3dgs/V3、/V4：与锚点无关，按与 meta 相同的行序（锚点升序、k 升序）填目标帧 t 的值
            for k in valid_ks(t_a, n):
                for g in roots3d:
                    for p, vv in (('V3', v3[g]), ('V4', v4[g])):
                        for m in KEEP_KEYS:
                            rec[f'{g}/{p}'][k][m].append(float(vv[t_a + k].get(m, np.nan)))
        print(f'[shard {a.shard}/{a.nshards}] scene {si + 1}/{len(my_scenes)} {scene_name}: {len(anchors)} anchors, {time.time() - t0:.0f}s elapsed', flush=True)
        del feats_all, refs, rfeats

    res = dict(ckpt=a.ckpt, convention=runner.convention, arms=names, render_roots=roots3d, scenes=[str(s) for s in my_scenes], ks=KS, ctx=a.ctx,
               preds={n: dict(path=s.path, step=s.step, n_params=s.n_params, ctx=s.ctx, map_layout=s.map_layout, assoc=s.assoc, det=s.det,
                              dtype=s.dtype_name, kv=s.kv) for n, s in srcs.items()},
               rollout_timing={n: {k: np.asarray(v, np.float64) for k, v in d.items()} for n, d in tlog.items()},
               det_meta=idx_det["det_meta"] if idx_det is not None else None,
               rec={n: {k: {m: np.asarray(v, np.float32) for m, v in d.items()} for k, d in dk.items()} for n, dk in rec.items()},
               meta={k: {f: np.asarray(v) for f, v in M.items()} for k, M in meta.items()},
               ref={f: np.asarray(v) for f, v in ref_info.items()},
               sec_per_uniad_step=float(np.mean(t_step)) if t_step else float('nan'),
               sec_rollout={n: float(np.mean(v)) for n, v in t_roll.items()},
               wall_s=time.time() - t0, cfg=vars(cfg))
    path = os.path.join(a.out, f'shard{a.shard}of{a.nshards}.pkl')
    with open(path + '.tmp', 'wb') as f:
        pickle.dump(res, f)
    os.replace(path + '.tmp', path)
    ident = res['rec'].get('identity')
    if ident is not None and any(len(ident[k]['plan_ade']) for k in KS):
        bad = (max(np.nanmax(ident[k]['plan_ade']) for k in KS if len(ident[k]['plan_ade'])),
               min(np.nanmin(ident[k]['det_iou']) for k in KS if len(ident[k]['det_iou'])))
        print(f'[identity selfcheck] max plan_ade={bad[0]:.3g} min det_iou={bad[1]:.3g} ' + ('PASS' if bad[0] == 0 and bad[1] == 1 else 'FAIL'), flush=True)
    for g, root in roots3d.items():                                        # '@nusc' 来源：与真图逐位相同才算过
        if root != NUSC_REAL or not any(len(meta[k]['scene']) for k in KS):
            continue
        ok = True
        for p in PROTOS:
            for k in KS:
                r_ = res['rec'][f'{g}/{p}'][k]
                if p == 'V1' and ident is not None:
                    ok &= all(np.array_equal(r_[m], ident[k][m], equal_nan=True) for m in KEEP_KEYS)
                ok &= bool(np.all(r_['plan_ade'] == 0)) and bool(np.all(r_['det_iou'][np.isfinite(r_['det_iou'])] == 1))
        print(f'[3dgs real selfcheck] {g}: ' + ('PASS' if ok else 'FAIL'), flush=True)
    npairs = {k: len(meta[k]['scene']) for k in KS}
    print(f'saved {path}: pairs per k {npairs} x {len(names)} arms, {res["wall_s"]:.0f}s', flush=True)


if __name__ == '__main__':
    main()
