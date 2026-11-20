"""G4 区分度探针（B 表协议的无训练版）：真图历史在锚帧分叉，之后 8 步喂"冻结"的特征 + 仿真位姿的 img_metas，
对比仿真位姿下的 GT（地图栅格、GT 框）。见设计稿 §5、§5.5、§7 G4。

臂
  real          真图，只有日志位姿（δ=0 / s=1）。= 主流式在 a+k 帧的输出（与在锚帧分叉后继续喂真图逐位相同）。
  copy          特征冻结在锚帧的真实 FPN，img_metas（can_bus / l2g）按仿真位姿。copy-last 下限。
  copy_logpose  同 copy，但 img_metas 用日志位姿（BEVFormer 不知道自车偏了）。与 copy 对比，看 can_bus/l2g 平移能补回多少（§5.5）。
                恒等扰动下与 copy 完全相同，跳过。
  zero          特征全零，img_metas 按仿真位姿。下限。
  （action 打乱臂需要训练好的预测器，本探针不含。）

扰动（sim_pose.LogTrack.rollout）
  lat   δ ∈ {0, ±0.5, ±1, ±2} m（4 个关键帧内 smoothstep 偏到 δ 后保持）
  speed s ∈ {0.5, 1.5}（s=1 即 lat δ=0）

每个 (锚帧, 扰动, 臂, k) 写一行 jsonl；`offlog/g4_merge.py` 汇总成"指标 vs δ"表和图。

用法（单卡；分片按场景轮转）::
  venv/bin/python -m offlog.g4_probe --out offlog/runs/g4_smoke --scenes 0,1 --max-anchors 2 --device cuda:0
  venv/bin/python -m offlog.g4_probe --out <dir> --scenes all --shard 3 --nshard 16 --anchor-stride 12
"""
import argparse
import json
import os
import socket
import time

import numpy as np
import torch

from . import common
from .gt_world import GTWorld
from .map_gt import MapGT
from .metas import OffLogEnv
from .selfcheck import det_match, iou, plan_in_drivable
from .sim_pose import LogTrack

LAT_VALUES = (0.0, -0.5, 0.5, -1.0, 1.0, -2.0, 2.0)
SPEED_VALUES = (0.5, 1.5)
ARMS = ('copy', 'copy_logpose', 'zero')


def frame_metrics(out, gt) -> dict:
    li, lu = iou(out['seg']['lane'] > 0.5, gt['map']['lane'])[1:]
    di, du = iou(out['seg']['drivable'] > 0.5, gt['map']['drivable'])[1:]
    m = det_match(out['boxes'], out['scores'], out['labels'], gt)
    dxy = np.array(m['dxy']).reshape(-1, 2)
    return dict(lane_i=li, lane_u=lu, drv_i=di, drv_u=du, n_gt=m['n_gt'], n_tp=m['n_tp'], n_pred=m['n_pred'],
                err_sum=float(np.sum(m['err'])), err=[round(e, 3) for e in m['err']],
                dx_sum=float(dxy[:, 0].sum()) if len(dxy) else 0.0,
                plan_in_drv=plan_in_drivable(out['plan'], gt['map']['drivable']),
                ego_drv=ego_on_drivable(gt['map']['drivable']),
                plan_end=[round(float(v), 3) for v in out['plan'][-1]])


def ego_on_drivable(drivable, half=51.2, res=0.512, lidar_fwd=0.94):
    """自车足迹（ego 原点附近 5×3 格 ≈ 2.6 m × 1.5 m）是否全在 GT 可行驶区域。栅格以 LiDAR 原点为中心，LiDAR 在 ego 前方约 0.94 m。
    横向漂移会把自车推出路面（val 实测 δ=+2 m 时约 1/3），此时 plan∈drivable 不再反映仿真器保真度，汇总时按它分组。"""
    row = int(np.floor((-lidar_fwd + half) / res))
    col = int(np.floor(half / res))
    return bool(drivable[row - 2:row + 3, col - 1:col + 2].all())


def parse_scenes(s, n):
    if s == 'all':
        return list(range(n))
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def scene_anchors(data, si, args, max_scale):
    an = LogTrack(data, data.scenes[si]).anchors(stride=args.anchor_stride, min_hist=args.min_hist, horizon=args.horizon,
                                                 max_scale=max_scale, min_speed=args.min_speed)
    return an[:args.max_anchors] if args.max_anchors else an


def balanced_shard(data, scene_ids, args, perts, arms, max_scale):
    """按估计步数贪心分片（大的先分给当前最轻的分片），确定性，各分片墙钟接近。"""
    n_branch = sum(len(arms) - (1 if ('copy_logpose' in arms and (f, v) in (('lat', 0.0), ('speed', 1.0))) else 0)
                   for f, v in perts)
    cost = []
    for si in scene_ids:
        an = scene_anchors(data, si, args, max_scale)
        cost.append((len(an) * n_branch * args.horizon + (max(an) + args.horizon + 1 if an else 0), si))
    load = [0] * args.nshard
    assign = [[] for _ in range(args.nshard)]
    for c, si in sorted(cost, key=lambda x: (-x[0], x[1])):
        if c == 0:
            continue
        j = min(range(args.nshard), key=lambda i: (load[i], i))
        load[j] += c
        assign[j].append(si)
    print(f'[shard] 估计步数 {load}（本分片 {load[args.shard]}）', flush=True)
    return sorted(assign[args.shard])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--scenes', default='all', help="'all' | '0,1,5' | '0-9'（UniADValData 场景下标）")
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--anchor-stride', type=int, default=4, help='锚帧间隔（关键帧）')
    ap.add_argument('--min-hist', type=int, default=4, help='锚帧前至少多少帧真实历史')
    ap.add_argument('--min-speed', type=float, default=5.0, help='窗口内日志最低速度（m/s），见 LogTrack.anchors')
    ap.add_argument('--max-anchors', type=int, default=0, help='每个场景最多多少个锚帧（0=不限；取前若干个）')
    ap.add_argument('--horizon', type=int, default=8)
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--lat', default=','.join(str(v) for v in LAT_VALUES))
    ap.add_argument('--speed', default=','.join(str(v) for v in SPEED_VALUES))
    ap.add_argument('--ckpt', default='hf', help='hf | gh | 路径')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    from uniad_port.data import UniADValData
    from uniad_port.runner import UniADRunner
    torch.set_grad_enabled(False)
    data = UniADValData()
    perts = [('lat', float(v)) for v in args.lat.split(',') if v] + [('speed', float(v)) for v in args.speed.split(',') if v]
    max_scale = max([v for f, v in perts if f == 'speed'] + [1.0])
    arms = [a for a in args.arms.split(',') if a]
    scene_ids = balanced_shard(data, parse_scenes(args.scenes, len(data.scenes)), args, perts, arms, max_scale)
    os.makedirs(args.out, exist_ok=True)
    tag = f'shard{args.shard:03d}of{args.nshard:03d}'
    path = os.path.join(args.out, f'{tag}.jsonl')
    done_path = os.path.join(args.out, f'{tag}.done.json')
    if os.path.exists(done_path):
        print('已完成，跳过', done_path)
        return
    ckpt = {'hf': common.CKPT_HF, 'gh': common.CKPT_GH}.get(args.ckpt, args.ckpt)
    runner = UniADRunner(ckpt, device=args.device, verbose=False)
    gtw = GTWorld([i['token'] for i in data.infos], cache_path=os.path.join(common.CACHE_DIR, 'gt_world_val.pkl'))
    env = OffLogEnv(data, gtw, MapGT())

    t_start = time.time()
    n_rows = n_steps = 0
    fout = open(path + '.tmp', 'w')
    for si in scene_ids:
        sc = data.scenes[si]
        tr = LogTrack(data, sc)
        anchors = scene_anchors(data, si, args, max_scale)
        if not anchors:
            continue
        last = max(anchors) + args.horizon
        runner.reset_scene()
        real_out = {}
        for j, fr in enumerate(data.iter_scene(si)):
            if j > last:
                break
            feats = runner.extract_feats(fr['img'])
            out = runner.step_frame(feats, fr)
            n_steps += 1
            if j > min(anchors):
                p = tr.rollout(j, 'lat', 0.0, horizon=0)[0]
                real_out[j] = frame_metrics(out, env.gt(p))
            if j not in anchors:
                continue
            a = j
            st_a = runner.get_state()
            feats_chk = [float(f.double().sum()) for f in feats]
            zero = [torch.zeros_like(f) for f in feats]
            for fam, val in perts:
                poses = tr.rollout(a, fam, val, horizon=args.horizon)
                gts = [None] + [env.gt(p) for p in poses[1:]]
                frames = [None] + [env.sim_frame(p) for p in poses[1:]]
                ident = all(p.identity for p in poses)
                for arm in arms:
                    if arm == 'copy_logpose' and ident:
                        continue
                    runner.set_state(st_a)
                    for k in range(1, args.horizon + 1):
                        p = poses[k]
                        f = frames[k]
                        if arm == 'copy_logpose':
                            f = data.get_frame(p.tmpl_idx, load_images=False)
                            f['command'] = frames[k]['command']
                        o = runner.step_frame(zero if arm == 'zero' else feats, f)
                        n_steps += 1
                        row = dict(scene=sc['scene_name'], si=si, anchor=a, fam=fam, val=val, arm=arm, k=k,
                                   lat=round(p.lat_m, 4), dpsi=round(p.dpsi, 5), log_f=round(p.log_f, 3),
                                   **frame_metrics(o, gts[k]))
                        fout.write(json.dumps(row) + '\n')
                        n_rows += 1
            runner.set_state(st_a)
            # 冻结特征在各分支里被反复使用，确认模型没有原地改写它
            assert [float(f.double().sum()) for f in feats] == feats_chk, '锚帧特征被原地修改'
        # real 臂：锚帧 a 的第 k 步 = 主流式在 a+k 帧的输出
        for a in anchors:
            for k in range(1, args.horizon + 1):
                row = dict(scene=sc['scene_name'], si=si, anchor=a, fam='lat', val=0.0, arm='real', k=k, lat=0.0,
                           dpsi=0.0, log_f=float(tr.ids[a + k]), **real_out[a + k])
                fout.write(json.dumps(row) + '\n')
                n_rows += 1
        fout.flush()
        el = time.time() - t_start
        print(f'[{tag}] scene {si} {sc["scene_name"]} anchors={anchors} rows={n_rows} steps={n_steps} '
              f'elapsed={el:.0f}s ({el / max(n_steps, 1):.3f} s/step)', flush=True)
    fout.close()
    os.replace(path + '.tmp', path)
    with open(done_path, 'w') as f:
        json.dump(dict(args=vars(args), scenes=scene_ids, rows=n_rows, steps=n_steps, seconds=time.time() - t_start,
                       host=socket.gethostname(), convention=runner.convention), f, indent=1)
    print('完成', path, n_rows, 'rows', f'{time.time() - t_start:.0f}s')


if __name__ == '__main__':
    main()
