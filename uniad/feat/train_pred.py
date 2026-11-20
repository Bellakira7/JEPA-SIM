"""PyramidPredictor 在 nuScenes train 上的训练（冻结 UniAD 前端在线抽特征，标准化空间 L1）。


  CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train_pred.py --out runs/smoke_1x --steps 40 --ckpt hf
1.0× 输入（928×1600，每帧 4 层 × 6 相机约 18.5 万位置）；``--ckpt`` 选前端权重，统计量文件按 ckpt 区分（``level_stats_1x_{ckpt}.pt``）。

阶段：
  - TF：teacher forcing，对 T-1 个转移一次前向，目标 = 标准化后的真实下一帧特征。

  - **滑窗 rollout**（``--rollout sliding``，设计稿 §3.4，主线论文 A.2 做法）：上下文窗口恒长 ``--ctx`` = L（与评测/部署的上下文一致），
    clip = L+F 帧；窗口 0 稠密 TF，之后每步丢最旧帧、拼最新预测，喂给 predictor 前对上下文 detach（TBPTT）；
    ``loss = (L_TF + Σ_{k=2..F} L_k) / F``，F 走 ``--F_schedule``（如 ``0:2,1500:4,3000:8`` = 按步数升 horizon），最终 F=``--F``。
    ``--ctx 4 --F 8`` = 覆盖 4 s。热启动用 ``--init`` 读 TF 阶段的 ckpt。
数据：``--map_layout``（layout 先画矢量地图再叠框 = 完整条件变体；不加 = 只画框的"不画地图"变体）、``--sweeps``（相机 sweeps 相位增广，约 ×6 样本）、
``--assoc_layout``（关联通道：layout 多 3 通道 = 帧 t 的框画进目标帧 t+1 的相机，见 pyramid_predictor 文档第 7 条；新参数零初始化，
``--init`` 允许只缺 ``layout_assoc.*``，所以可从不带它的 ckpt 热启动且初始输出逐位相同）。
诊断：copy-last 的 L1（同一标准化空间），以及 ratio = L1_pred / L1_copy（逐层）；滑窗时另报逐步（horizon k）的 ratio。
"""
import featsim_paths as _release_paths
import argparse
import datetime
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_dataset import NuscClips, collate, normalize_on_gpu
from extractor import UniADFrontend
from pyramid_predictor import PyramidPredictor, sliding_rollout

STATS = str(_release_paths.Path(_release_paths.BENCH) / 'feat/level_stats_1x_{ckpt}.pt')


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--T', type=int, default=5, help='clip 帧数；--rollout sliding 时忽略，自动 = ctx + F')
    ap.add_argument('--bs', type=int, default=1, help='每卡 clip 数')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=0.04)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--depth', type=int, default=8)
    ap.add_argument('--pools', type=int, nargs=4, default=[8, 4, 2, 1])
    ap.add_argument('--no_layout', action='store_true')
    ap.add_argument('--map_layout', action='store_true', help='layout 里画矢量地图（车道线/人行横道/道路边界）再叠框：完整条件变体')
    ap.add_argument('--sweeps', action='store_true', help='训练集用相机 sweeps 相位增广（每次取样随机相位 0..5；val 恒为相位 0）')
    ap.add_argument('--assoc_layout', action='store_true', help='layout 加关联通道（帧 t 的框画进目标帧 t+1 的相机，只画两帧都在场的实例）')
    ap.add_argument('--residual', action='store_true', help='所有层输出 = 上一帧同层特征 + Δ（proj 零初始化）')
    ap.add_argument('--warp_levels', type=int, nargs='*', default=[], help='这些层（0=P3,1=P4,…）输出 = 流场 warp 后的上一帧同层 + Δ（见 pyramid_predictor 文档第 6 条）')
    ap.add_argument('--flow_px', type=float, default=32.0, help='流场单位：1.0 的流 = 多少像素')
    ap.add_argument('--ar_k', type=int, default=0)
    ap.add_argument('--ar_start', type=int, default=0)
    ap.add_argument('--rollout', default='tf', choices=['tf', 'sliding'], help="tf = 旧口径（TF，可加 --ar_k 的 grow AR）；sliding = 全 horizon 滑窗 rollout")
    ap.add_argument('--ctx', type=int, default=4, help='滑窗上下文长度 L（= 评测/部署的上下文）')
    ap.add_argument('--F', type=int, default=8, help='滑窗 rollout 的最终预测帧数 F（含第 1 步）；clip = ctx + F 帧')
    ap.add_argument('--F_schedule', default=None, help="F 课程，'iter:F,iter:F,…'（升序，按步数查表），如 '0:2,1500:4,3000:8'；不给 = 恒为 --F")
    ap.add_argument('--max_clips', type=int, default=None, help='过拟合/冒烟用：只取前 N 个 clip')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--val_every', type=int, default=1000)
    ap.add_argument('--val_clips', type=int, default=64)
    ap.add_argument('--val_at_start', action='store_true', help='step 0（非续训）先验证一次：热启动模型在本阶段口径下的基线')
    ap.add_argument('--ckpt_every', type=int, default=1000)
    ap.add_argument('--init', default=None)
    ap.add_argument('--resume', action='store_true', help='输出目录里有 iter_*.pt 就从最新一个续训（含优化器状态与步数）；中断后恢复训练')
    ap.add_argument('--ckpt', default='hf', choices=['hf', 'gh'], help='UniAD 前端权重（与下游评测用同一份）')
    ap.add_argument('--stats', default=None, help='默认 STATS.format(ckpt=...)')
    a = ap.parse_args(argv)
    a.sched = parse_F_schedule(a.F_schedule, a.F) if a.rollout == 'sliding' else None
    if a.rollout == 'sliding':
        assert a.ctx >= 2 and a.F >= 1
        a.T = a.ctx + a.F  # 窗口 0 的 L 帧 + F 个目标帧（目标帧 L..L+F-1；上下文只用前 L 帧）
        assert not a.ar_k, '--rollout sliding 与 --ar_k（grow AR）互斥'
    return a


def parse_F_schedule(s, F_final):
    """'0:2,1500:4,3000:8' → [(0,2),(1500,4),(3000,8)]；None → [(0,F_final)]。要求升序，最后一项必须 = F_final。"""
    if not s:
        return [(0, F_final)]
    sch = [tuple(int(x) for x in kv.split(':')) for kv in s.split(',')]
    assert [i for i, _ in sch] == sorted(i for i, _ in sch) and sch[0][0] == 0, f'F_schedule 必须升序且从 0 开始: {sch}'
    assert sch[-1][1] == F_final, f'F_schedule 的最后一项 {sch[-1]} 必须等于 --F {F_final}'
    assert all(1 <= f <= F_final for _, f in sch)
    return sch


def F_at(sched, step):
    f = sched[0][1]
    for it, k in sched:
        if step >= it:
            f = k
    return f


def standardize(fs, mean, std):
    """fs: list[4] of [B*T*6,256,h,w] → 标准化 → [B,T,6,256,h,w]"""
    return [(f - m.view(1, -1, 1, 1)) / s.view(1, -1, 1, 1) for f, m, s in zip(fs, mean, std)]


@torch.no_grad()
def extract(fe, imgs_u8, mean, std):
    """imgs_u8 [B,T,6,3,928,1600] uint8 (GPU) → list[4] of [B,T,6,256,h,w]（标准化，fp32）"""
    B, T = imgs_u8.shape[:2]
    outs = None
    # 逐帧过前端，避免一次 6*B*T 张图占显存
    per = []
    for b in range(B):
        for t in range(T):
            x = normalize_on_gpu(imgs_u8[b, t])
            per.append(fe(x))
    outs = []
    for l in range(4):
        f = torch.stack([p[l] for p in per])  # [B*T,6,256,h,w]
        f = (f - mean[l].view(1, 1, -1, 1, 1)) / std[l].view(1, 1, -1, 1, 1)
        outs.append(f.view(B, T, *f.shape[1:]))
    return outs


def l1_per_level(pred, tgt):
    return [(p.float() - t.float()).abs().mean() for p, t in zip(pred, tgt)]


class SlidingRollout(nn.Module):
    """把整个滑窗 rollout 包成一次 forward（DDP 只看到一次 forward；predictor 在里面被多次调用，梯度在一次 backward 里累加）。
    state_dict 的键仍由外面的 ``predictor`` 单独保存，ckpt 格式不变。"""

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, feats, actions, layout, L, Fn):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return sliding_rollout(self.predictor, feats, actions, layout, L, Fn)


def sliding_losses(z_tf, z_ar, feats, L, Fn):
    """L1（4 层平均）：L_TF = 窗口 0 全部 L 个位置；L_k = 第 k 步预测帧（帧 L+k-1）。loss = (L_TF + Σ_{k=2..F} L_k) / F（主线 Eq.12 去掉 SIGReg）。
    返回 (loss, 诊断 dict)；诊断里 ar[k-2] / copy[k-1] 都是 4 层平均的标量 tensor（detach）。copy_k = 把最后一个真实上下文帧（L-1）复制成帧 L+k-1 的 L1。"""
    lv = lambda p, t: sum(l1_per_level(p, t)) / 4
    l_tf = lv(z_tf, [f[:, 1:L + 1] for f in feats])
    l_k = [lv([z[:, i:i + 1] for z in z_ar], [f[:, L + 1 + i:L + 2 + i] for f in feats]) for i in range(Fn - 1)]
    loss = (l_tf + sum(l_k)) / Fn
    with torch.no_grad():
        copy = [lv([f[:, L - 1:L] for f in feats], [f[:, L + k - 1:L + k] for f in feats]).detach() for k in range(1, Fn + 1)]
        step_l1 = [lv([z[:, L - 1:L] for z in z_tf], [f[:, L:L + 1] for f in feats]).detach()] + [x.detach() for x in l_k]
    return loss, dict(tf=l_tf.detach(), step=step_l1, copy=copy)


def main():
    a = get_args()
    ddp = 'RANK' in os.environ
    if ddp:
        dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=60))  # 默认 10 min：个别 rank 取 sweeps 图慢一点就会 NCCL 超时（见 sliding_xl 11-05 那次）
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    else:
        rank, world = 0, 1
    dev = torch.device('cuda')
    os.makedirs(a.out, exist_ok=True)
    log = (lambda *s: print(*s, flush=True)) if rank == 0 else (lambda *s: None)

    stats_path = a.stats or STATS.format(ckpt=a.ckpt)
    st = torch.load(stats_path)
    assert st.get('scale') == 1.0 and st.get('ckpt') == a.ckpt, f'{stats_path} 不是 1.0×/{a.ckpt} 的统计量: {st.get("scale")}, {st.get("ckpt")}'
    mean = [m.to(dev) for m in st['mean']]
    std = [s.to(dev) for s in st['std']]
    fe = UniADFrontend(a.ckpt).to(dev)
    mk = dict(level_pools=a.pools, depth=a.depth, enable_layout=not a.no_layout, residual=a.residual,
              warp_levels=tuple(a.warp_levels), flow_px=a.flow_px)
    if a.assoc_layout:  # 不开时 mk（会存进 ckpt 的 model_kwargs）与加这个开关之前逐键相同
        mk['assoc_layout'] = True
    model = PyramidPredictor(**mk, use_activation_checkpointing=True).to(dev)
    if a.init:
        sd = torch.load(a.init, map_location='cpu')['model']
        if a.assoc_layout:
            missing, unexpected = model.load_state_dict(sd, strict=False)
            assert not unexpected and all(k.startswith('layout_assoc.') for k in missing), f'热启动键不匹配: missing={missing} unexpected={unexpected}'
            log(f'[init] 热启动自 {a.init}（新建、零初始化：{missing}）')
        else:
            model.load_state_dict(sd)
            log(f'[init] 热启动自 {a.init}')
    log('params(M)', sum(p.numel() for p in model.parameters()) / 1e6, 'tokens/frame', model.tokens_per_frame)
    sliding = a.rollout == 'sliding'
    mod = SlidingRollout(model) if sliding else model  # 被 DDP 包的模块；ckpt 只存 model.state_dict()
    net = torch.nn.parallel.DistributedDataParallel(mod, device_ids=[dev.index]) if ddp else mod

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and 'embed' not in n]
    nodecay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and 'embed' not in n)]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=a.wd), dict(params=nodecay, weight_decay=0.0)], lr=a.lr, betas=(0.9, 0.999))
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))) * 0.99 + 0.01)

    ds = NuscClips('train', T=a.T, step=1, max_clips=a.max_clips, with_layout=not a.no_layout, map_layout=a.map_layout,
                   sweeps=a.sweeps, phase_mode='random', assoc_layout=a.assoc_layout)
    vds = NuscClips('val', T=a.T, step=7, max_clips=a.val_clips, with_layout=not a.no_layout, map_layout=a.map_layout,
                    assoc_layout=a.assoc_layout)
    sampler = DistributedSampler(ds, world, rank, shuffle=True, drop_last=True) if ddp else None
    dl = DataLoader(ds, batch_size=a.bs, sampler=sampler, shuffle=(sampler is None), num_workers=a.workers,
                    collate_fn=collate, drop_last=True, persistent_workers=True, prefetch_factor=2)
    vdl = DataLoader(vds, batch_size=1, num_workers=4, collate_fn=collate)
    log(f'train clips {len(ds)}  val clips {len(vds)}  world {world}  bs/gpu {a.bs}  T {a.T}  map_layout {a.map_layout}  sweeps {a.sweeps}'
        + (f'  assoc_layout True' if a.assoc_layout else '')
        + (f'  sliding L={a.ctx} F={a.F} schedule={a.sched}' if sliding else ''))

    def forward_loss(imgs, actions, layout, step, train=True, F_now=None):
        lay = layout.float() / 255.0 if layout is not None else None
        if sliding:
            Fn = F_now if F_now is not None else F_at(a.sched, step)
            L = a.ctx
            Te = L + Fn  # 只抽用得到的帧：上下文 L 帧 + 目标帧 L..L+Fn-1
            feats = extract(fe, imgs[:, :Te], mean, std)
            z_tf, z_ar = net(feats, actions[:, :Te - 1], lay[:, :Te - 1] if lay is not None else None, L, Fn)
            loss, d = sliding_losses(z_tf, z_ar, feats, L, Fn)
            return loss, dict(F=Fn, tf=d['tf'].item(), step=[x.item() for x in d['step']], copy=[x.item() for x in d['copy']])
        feats = extract(fe, imgs, mean, std)  # list[4] [B,T,6,256,h,w]
        T = a.T
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = net([f[:, :T - 1] for f in feats], actions[:, :T - 1], lay[:, :T - 1] if lay is not None else None)
        tgt = [f[:, 1:] for f in feats]
        l_tf = l1_per_level(z, tgt)
        loss = sum(l_tf) / 4
        l_copy = l1_per_level([f[:, :T - 1] for f in feats], tgt)
        out = dict(tf=[x.item() for x in l_tf], copy=[x.item() for x in l_copy])
        if a.ar_k > 1 and step >= a.ar_start:
            # grow 式 AR：种子 [h_0, z_tf_0]，每轮把最新预测拼到输入末尾；z_ar 对齐 h[:, 1:k+1]
            ctx = [torch.cat([f[:, :1], zz[:, :1].to(f.dtype)], 1) for f, zz in zip(feats, z)]
            l_ar = []
            for k in range(1, a.ar_k):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    zk = net(ctx, actions[:, :k + 1], lay[:, :k + 1] if lay is not None else None)
                tk = [f[:, 1:k + 2] for f in feats]
                l_ar.append(l1_per_level([x[:, -1:] for x in zk], [x[:, -1:] for x in tk]))
                ctx = [torch.cat([c, x[:, -1:].to(c.dtype)], 1) for c, x in zip(ctx, zk)]
            ar = sum(sum(x) / 4 for x in l_ar) / len(l_ar)
            loss = (loss + ar) / 2
            out['ar'] = ar.item()
        return loss, out

    @torch.no_grad()
    def validate(step):
        net.eval()
        agg = None
        n = 0
        for b in vdl:
            _, o = forward_loss(b['imgs'].to(dev), b['actions'].to(dev), b['layout'].to(dev) if b['layout'] is not None else None, step, False,
                                F_now=a.F if sliding else None)
            v = torch.tensor(o['step'] + o['copy'] if sliding else o['tf'] + o['copy'], device=dev)
            agg = v if agg is None else agg + v
            n += 1
        agg = agg / n
        if ddp:
            dist.all_reduce(agg)
            agg = agg / world
        net.train()
        if ddp:  # 滑窗里同一个 predictor 在一次前向里被调用多次：核对 DDP 梯度同步没出问题（各 rank 参数应逐位一致）
            chk = torch.stack([p.detach().double().sum() for p in model.parameters()]).sum().reshape(1)
            g = [torch.zeros_like(chk) for _ in range(world)]
            dist.all_gather(g, chk)
            g = torch.cat(g)
            log(f'[sync] step {step} 参数校验和跨 rank 最大差 {(g.max() - g.min()).item():.3e}' + ('' if (g.max() - g.min()).item() == 0 else '  ⚠️ 各 rank 参数不一致'))
        if sliding:  # 逐步 horizon k=1..F 的绝对 L1 与 copy-last（最后一个真实上下文帧复制）的比
            st_, cp = agg[:a.F].tolist(), agg[a.F:].tolist()
            log(f'[val {step}] step_l1', [f'{x:.3f}' for x in st_], 'copy', [f'{x:.3f}' for x in cp], 'ratio', [f'{x / y:.3f}' for x, y in zip(st_, cp)])
        else:
            tf, cp = agg[:4].tolist(), agg[4:].tolist()
            log(f'[val {step}] tf', [f'{x:.3f}' for x in tf], 'copy', [f'{x:.3f}' for x in cp], 'ratio', [f'{x / y:.3f}' for x, y in zip(tf, cp)])

    step, t0 = 0, time.time()
    if a.resume:
        done = sorted((int(f[5:-3]) for f in os.listdir(a.out) if f.startswith('iter_') and f.endswith('.pt')))
        if done:
            ck = torch.load(f'{a.out}/iter_{done[-1]}.pt', map_location='cpu', weights_only=False)
            assert 'opt' in ck, f'iter_{done[-1]}.pt 没有优化器状态，不能续训'
            model.load_state_dict(ck['model'])
            opt.load_state_dict(ck['opt'])
            step = int(ck['step'])
            log(f'[resume] 从 {a.out}/iter_{step}.pt 续训，步数 {step}/{a.steps}')
    step0 = step
    net.train()
    if a.val_at_start and step == 0:
        validate(0)
    ema = None
    while step < a.steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for b in dl:
            for g in opt.param_groups:
                g['lr'] = a.lr * sched(step)
            loss, o = forward_loss(b['imgs'].to(dev, non_blocking=True), b['actions'].to(dev),
                                   b['layout'].to(dev) if b['layout'] is not None else None, step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            step += 1
            if step % 20 == 0:
                tail = (f' gn {gn:.2f} lr {opt.param_groups[0]["lr"]:.2e} {(time.time() - t0) / max(step - step0, 1):.2f}s/it'
                        f' mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G')
                if sliding:
                    r = [x / y for x, y in zip(o['step'], o['copy'])]
                    log(f'step {step} loss {loss.item():.4f} F {o["F"]} tf {o["tf"]:.3f} step_ratio {[f"{x:.2f}" for x in r]}' + tail)
                else:
                    r = [x / y for x, y in zip(o['tf'], o['copy'])]
                    log(f'step {step} loss {loss.item():.4f} tf {[f"{x:.3f}" for x in o["tf"]]} ratio {[f"{x:.2f}" for x in r]}'
                        + (f' ar {o["ar"]:.3f}' if 'ar' in o else '') + tail)
            if step % a.val_every == 0 or step == a.steps:
                validate(step)
            if rank == 0 and (step % a.ckpt_every == 0 or step == a.steps):
                tmp = f'{a.out}/.iter_{step}.pt.tmp'  # 先写临时文件再改名，平台随时可能杀进程
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), args=vars(a), model_kwargs=mk, step=step), tmp)
                os.replace(tmp, f'{a.out}/iter_{step}.pt')
            if step >= a.steps:
                break
    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
