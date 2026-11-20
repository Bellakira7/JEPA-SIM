"""C 表：整段闭环（设计稿 §6）。每个 val 场景先真图 warmup W 帧（UniAD 流式状态与预测器上下文都在真图上建好），之后 UniAD 自己开到场景结束（0.5 s 一步）。

每一步（仿真第 j 步 = 日志第 j 个关键帧时刻，j ≥ W）
  1. 自车位姿：上一步 UniAD 规划的第一个路点 → 运动学自行车模型走 0.5 s（c_table/closed_loop.py）；W−1 → W 这一步用 warmup 末帧的规划、日志速度起步。
  2. 输入帧：`offlog.metas.OffLogEnv.sim_frame`（img_metas / can_bus / l2g 按仿真位姿；lidar2img、timestamp 取时间对齐的日志帧），
     导航命令 = 日志路线 + 仿真位姿（`closed_loop.Route.command`，覆盖 sim_frame 给的日志命令）。
  3. 特征（臂）：
       log      沿日志真图：位姿强制 = 日志、特征 = 真图主干。感知保真的上限，也给"日志自车"的驾驶指标作参照（碰撞指标的健全性：日志自车应几乎无有责碰撞）
       copy     warmup 末帧的真实 FPN 冻结
       <name>   --pred name=ckpt 的预测器，一步一步 AR：上下文 = 最近 ctx 帧（warmup 真特征 + 之后自己的预测），条件 = 仿真位姿的相对运动
                （训练公式 nd.relative_action）+ 仿真位姿下重投影的 layout（offlog/layout_reproject，画不画地图按 ckpt）；与 B 表 SimARSource.rollout_cond
                同一窗口 / 标准化 / bf16 约定（test_c_eval.py 对拍逐位相同）
       3dgs     SplatAD 在仿真位姿下在线渲染（c_table/splat_server.py，SplatAD venv 里的常驻进程）→ JPEG q95 → 与真图同一条加载路径 → 主干
       3dgs_difix  同上，渲染图落盘后**逐步**过 Difix（difix/difix_core.py，nvidia/difix 无参考图、576×1024，与 B 表 / A 表 O2 臂同一后处理）再读回 → 主干。
                审稿 W2：闭环里补最强的渲染基线。Difix 与 UniAD 同进程同卡（~5 GB），--difix-weights 给权重目录；task group = 3dgs_difix
       <name>@det  预测器 <name> 的同一份 ckpt，但 layout 里的框换成检测器框（--det-boxes，feat/det_layout.py；日志时间对齐的检测，按仿真位姿重投影；
                地图 / action 不变）。审稿 W4：去掉"目标帧 GT 框"这一特权信息
  4. UniAD 从 FPN 起跑一步 → 规划 / 检测 / 地图。
  5. 指标（仿真位姿下的 GT，与 B 表同一套）：offlog.g4_probe.frame_metrics（lane / drivable IoU、recall@2m、…）+ offlog.det_lag_gt 的 lag（折线要 j+1 的位姿，
     所以最后一步不算 lag）；自车框是否全在 GT drivable；本步区间内的碰撞（closed_loop.collisions_interval，每 0.1 s）。

输出（--out）：每个 (场景, 臂) 一个 scene<NNN>__<arm>.jsonl（每步一行，warmup 不写）+ scene<NNN>__<arm>.done.json（场景级汇总 + warmup 摘要哈希 + 预测器元信息）。
**已有 done.json 的（场景, 臂）跳过**：追加新预测器臂（J2 的 tffull、O4 的 F16 / F8 续训、O5 的关联通道）时只给新臂（--arms / C_MAIN_ARMS），合并可以把多次运行并起来（c_merge.py 吃多个目录）。


用法（单卡冒烟）：
  CUDA_VISIBLE_DEVICES=2 venv/bin/python c_table/c_eval.py --tasks "1:main" --out c_table/runs/smoke \
      --pred rofull6000=ro/runs/full_20261105/train_full/iter_6000.pt --max-steps 6
"""
import featsim_paths as _release_paths
import argparse
import collections
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
import traceback
from multiprocessing.connection import Client

import numpy as np
import torch

ROOT = os.environ.get('G3_CODE_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, 'difix'), os.path.join(ROOT, 'c_table'), os.path.join(ROOT, 'a_table'), os.path.join(ROOT, 'feat'), ROOT, os.path.join(ROOT, 'uniad_port')):
    if p not in sys.path:
        sys.path.insert(0, p)
from offlog import common  # noqa: E402
from offlog.det_lag_gt import build_polys, lag_match  # noqa: E402
from offlog.g4_probe import frame_metrics  # noqa: E402
from offlog.gt_world import GTWorld  # noqa: E402
from offlog.layout_reproject import IDENTITY_ATOL, SimLayoutRenderer, pose_to_matrix  # noqa: E402
from offlog.map_gt import MapGT  # noqa: E402
from offlog.metas import OffLogEnv  # noqa: E402
from offlog.sim_pose import LogTrack  # noqa: E402
import closed_loop as cl  # noqa: E402
import nusc_data as nd  # noqa: E402
from a_eval import ARSource, CKPTS, STATS  # noqa: E402
from pyramid_predictor import trim_kv_cache  # noqa: E402
from clip_dataset import ACT_MEAN, ACT_STD  # noqa: E402

BENCH = common.UNIAD_BENCH
OFF_ROUTE_M = 10.0          # 偏离日志路线超过这个距离（m）记为离开路线（之后的进度 / 命令只作参考）
RENDER_KINDS = ('3dgs', '3dgs_difix')     # 渲染臂（task group 名 = 臂名）
DEFAULT_PREDS = ()


def jdefault(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def out_digest(out) -> str:
    """UniAD 一步输出的摘要哈希（规划 / 框 / 分数 / 类别 / track id / 地图栅格），warmup 逐位一致性核对用。"""
    h = hashlib.sha1()
    for a in (out['plan'], out['boxes'], out['scores'], out['labels'], out['track_ids'], out['seg']['lane'], out['seg']['drivable']):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


class StepPredictor(ARSource):
    """一步一步的 AR 预测器（闭环里条件逐步才知道）。窗口 / 标准化 / bf16 与 b_eval.SimARSource.rollout_cond 逐行一致。
    关联通道 ckpt（O5，``pyramid_predictor.uses_assoc_layout``）：layout 是 [6,6,H,W]（主通道 + 上一步框画进目标帧相机），由调用方按 self.assoc 渲染。

    推理选项 dtype / kv 同 ``a_eval.ARSource``（默认 bf16 autocast + 稠密滑窗 = 原路径逐位不变）。kv=True 时 ``predict`` 有状态：
    warmup 之后的第一次调用对真实窗口做稠密前向并建缓存，之后每次只前向"上一步预测出的那一帧"（调用方把上一次返回的 nxt 原样追加进 seq，
    靠 ``seq[-1] is 上次返回的 nxt`` 判断缓存是否连续；不连续——新场景 / 新臂的第一步——就重建，不需要显式 reset）。"""

    def __init__(self, name, path, mean, std, dev, ctx_default, dtype='bf16', kv=False):
        super().__init__(name, path, mean, std, dev, ctx_default, dtype=dtype, kv=kv)
        import pyramid_predictor as pp
        ck = torch.load(path, map_location='cpu', weights_only=False)
        self.assoc = bool(pp.uses_assoc_layout(ck)) if hasattr(pp, 'uses_assoc_layout') else False
        self._cache, self._pos, self._last_nxt = None, 0, None

    def _act(self, Ew):
        """Ew：连续 n+1 帧的 ego2global → 标准化 action [1,n,3]。"""
        act = np.stack([nd.relative_action({'ego2global': Ew[i]}, {'ego2global': Ew[i + 1]}) for i in range(len(Ew) - 1)]).astype(np.float32)
        return torch.from_numpy((act - ACT_MEAN) / ACT_STD).to(self.dev).unsqueeze(0)

    @torch.no_grad()
    def predict(self, seq, E, lays):
        """seq：最近 ≤ctx 帧的标准化特征（旧→新，list of list[4]）；E：与 seq 对齐再多一帧（目标帧）的 4×4 ego2global；
        lays：与 seq 对齐再多一帧的 uint8 layout（lays[0] 不用）。返回 (目标帧的标准化特征, 原始特征)。"""
        if self.kv:
            return self.predict_kv(seq, E, lays)
        W = min(self.ctx, len(seq))
        win = seq[-W:]
        feats = [torch.stack([f[l] for f in win]).unsqueeze(0) for l in range(4)]
        act = self._act(E[-(W + 1):])
        lay = torch.from_numpy(np.stack(lays[-W:])).to(self.dev).float().unsqueeze(0) / 255.0
        z = self._model_call(self.model, feats, act, lay)
        nxt = [x[0, -1].float() for x in z]
        return nxt, self.denorm(nxt)

    @torch.no_grad()
    def predict_kv(self, seq, E, lays):
        if self._cache is not None and seq[-1] is self._last_nxt:         # 接着上一步：只前向新帧 seq[-1]（上一步的预测）
            lay = torch.from_numpy(np.asarray(lays[-1])[None]).to(self.dev).float().unsqueeze(0) / 255.0
            self._cache = trim_kv_cache(self._cache, self.ctx - 1)
            z, self._cache = self._model_call(lambda f, a, l: self.model.forward_step(f, a, l, pos=self._pos, kv_cache=self._cache),
                                              [x[None, None] for x in seq[-1]], self._act(E[-2:]), lay)
            self._pos += 1
        else:                                                              # 重建：真实窗口稠密前向 + 建缓存（位置 0..W-1）
            W = min(self.ctx, len(seq))
            feats = [torch.stack([f[l] for f in seq[-W:]]).unsqueeze(0) for l in range(4)]
            lay = torch.from_numpy(np.stack(lays[-W:])).to(self.dev).float().unsqueeze(0) / 255.0
            z, self._cache = self._model_call(lambda f, a, l: self.model(f, a, l, return_kv_cache=True), feats, self._act(E[-(W + 1):]), lay)
            self._pos = W
        nxt = [x[0, -1].float() for x in z]
        self._last_nxt = nxt
        return nxt, self.denorm(nxt)


class SplatClient:
    """SplatAD 渲染服务（splat_server.py，在 SplatAD venv 里）的客户端：每个场景起一个服务进程（同一张卡），AF_UNIX socket 通信。"""

    def __init__(self, cfg, sock, log_path, spy, neurad, scripts_dir, torch_home, timeout=900):
        if os.path.exists(sock):
            os.unlink(sock)
        self.key = os.urandom(16)
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'G3_CODE_ROOT')}
        # OMP 线程必须限住：SplatAD 加载时的 LiDAR 预处理全是 CPU 小算子，torch 默认开满核会严重超订（E0：1173 s → 23 s，splatad/README）
        omp = os.environ.get('SPLAT_OMP') or '8'
        env.update(PYTHONPATH=neurad, TORCH_HOME=torch_home, PYTHONUNBUFFERED='1', SPLAT_AUTHKEY=self.key.hex(),
                   OMP_NUM_THREADS=omp, MKL_NUM_THREADS=omp)
        self.logf = open(log_path, 'a')
        server = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'splat_server.py')
        self.proc = subprocess.Popen([spy, server, '--config', cfg, '--address', sock, '--scripts-dir', scripts_dir],
                                     cwd=neurad, env=env, stdout=self.logf, stderr=subprocess.STDOUT)
        t0 = time.time()
        while not os.path.exists(sock):
            if self.proc.poll() is not None:
                raise RuntimeError(f'SplatAD 服务启动失败（退出码 {self.proc.returncode}），见 {log_path}')
            if time.time() - t0 > timeout:
                self.proc.kill()
                raise RuntimeError(f'SplatAD 服务 {timeout}s 内没就绪，见 {log_path}')
            time.sleep(1.0)
        self.conn = Client(sock, family='AF_UNIX', authkey=self.key)
        self.sec_load = time.time() - t0

    def render(self, files, D, out_dir):
        self.conn.send(dict(op='render', files=files, D=D, out_dir=out_dir))
        r = self.conn.recv()
        if not r.get('ok'):
            raise RuntimeError(f'SplatAD 渲染失败：{r.get("error")}')
        return r

    def close(self):
        try:
            self.conn.send(dict(op='quit'))
            self.conn.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.logf.close()


class Scene:
    """一个场景的与臂无关的量（日志轨迹 / 路线 / 上下文帧的 nusc 帧与 layout 缓存）。"""

    def __init__(self, g, si):
        self.g, self.si = g, si
        self.sc = g.data.scenes[si]
        self.name = self.sc['scene_name']
        self.tr = LogTrack(g.data, self.sc)
        self.route = cl.Route(g.data, self.tr)
        self.n = self.tr.n
        self.ids = self.tr.ids
        self.l2e_t = tuple(self.tr.infos[0]['lidar2ego_translation'][:2])
        self.lay_cache = {}


    def ctx_E(self, j):
        return self.g.rend.nd_frame(self.ids[j])['ego2global']

    def check_det(self):
        """@det 臂开跑前核对：本场景每个关键帧都有检测器缓存（不静默回退到 GT 框）。"""
        miss = [j for j in range(self.n) if self.g.rend_det.nd_frame(self.ids[j]).get('box_src') != 'det']
        assert not miss, f'{self.name}: 检测器缓存缺 {len(miss)} 帧'

    def ctx_layout(self, j, wm, assoc=False, det=False):
        """日志帧 j 的 layout（与 b_eval.LayoutCache / a_eval.layout_of 同一写法）；assoc=True 时 = nd.render_layout_cond(日志帧 j−1, 日志帧 j, assoc=True)。
        det=True：框取检测器（g.rend_det 的索引，feat/det_layout.py），地图与几何不变。"""
        key = (j, wm, assoc, det)
        rend = self.g.rend_det if det else self.g.rend
        if key not in self.lay_cache:
            fr = rend.nd_frame(self.ids[j])
            mv = rend.map_vecs.for_frame(fr) if wm else None
            if assoc:
                self.lay_cache[key] = nd.render_layout_cond(rend.nd_frame(self.ids[j - 1]), fr, map_vec=mv, assoc=True)
            else:
                self.lay_cache[key] = nd.render_layout(fr, map_vec=mv)
        return self.lay_cache[key]

    def sim_layout(self, pose_prev, pose, wm, assoc=False, det=False):
        """仿真位姿 pose 下的 layout；assoc=True 时关联通道取 pose_prev 时间对齐的日志框（offlog.layout_reproject.render_cond）。
        det=True：其他 agent 取时间对齐日志帧的检测器框（同样按仿真位姿重投影）。"""
        rend = self.g.rend_det if det else self.g.rend
        if assoc:
            return rend.render_cond(pose_prev, pose, with_map=wm, assoc=True)
        return rend.render(pose, with_map=wm)

    def frame_of(self, pose, cmd):
        fr = self.g.env.sim_frame(pose)
        fr['command'] = int(cmd)
        return fr


def scene_metrics_row(scn, arm, j, pose, st, out, gt, cmd, colls, tm):
    xy, yaw = cl.pose_xy_yaw(pose)
    dac, dac_frac = cl.footprint_on_drivable(gt['map']['drivable'])
    row = dict(scene=scn.name, si=scn.si, arm=arm, j=j, tau=round((j - scn.g.W + 1) * cl.DT, 2),
               x=round(float(xy[0]), 3), y=round(float(xy[1]), 3), yaw=round(float(yaw), 5),
               v=round(st['v'], 3), a=round(st['a'], 3), f=round(st['f'], 3), lat=round(st['lat'], 3),
               steer=round(st.get('steer', 0.0), 4), clipped_steer=st.get('clipped_steer', False), clipped_acc=st.get('clipped_acc', False),
               cmd=int(cmd), cmd_log=int(scn.g.data.planning_label(scn.ids[j])[2]),
               ego_dac=dac, ego_dac_frac=round(dac_frac, 3), coll=colls, **tm)
    row.update(frame_metrics(out, gt))
    return row


def run_arm(g, scn, arm, kind, warm_feats, src=None, img_iter=None, splat=None, render_dir=None, max_steps=0, digests=None, feats_out=None):
    """一个臂的整段闭环。kind ∈ {log, copy, pred, 3dgs, 3dgs_difix}。返回 (rows, summary)。
    digests（list，自检用）：追加每一步（含 warmup）UniAD 输出的摘要哈希；feats_out（dict，自检用）：记录闭环各步喂给 UniAD 的原始特征。"""
    runner, data, tr, route, W = g.runner, g.data, scn.tr, scn.route, g.W
    n_end = scn.n if not max_steps else min(scn.n, W + max_steps)
    runner.reset_scene()
    poses, gts, outs_box, rows = {}, {}, {}, []
    warm_dig = []
    seq, E_hist, lay_hist = [], [], []
    wm = bool(src.map_layout) if src is not None else False
    asc = bool(getattr(src, 'assoc', False))
    dt_ = bool(getattr(src, 'det', False))
    if dt_:
        scn.check_det()
    # ---- warmup：真图，日志位姿（命令走同一个 Route.command，恒等位姿下 = 官方命令）----
    out = None
    for j in range(W):
        pose = cl.log_pose(tr, j)
        poses[j] = pose
        f = warm_feats[j]
        out = runner.step_frame(f, scn.frame_of(pose, route.command(pose, j)))
        warm_dig.append(out_digest(out))
        if digests is not None:
            digests.append(warm_dig[-1])
        if kind == 'pred':
            seq.append(src.norm(f))
            E_hist.append(scn.ctx_E(j))
            lay_hist.append(scn.ctx_layout(j, wm, asc, dt_) if j >= 1 else None)
            seq, E_hist, lay_hist = seq[-src.ctx:], E_hist[-src.ctx:], lay_hist[-src.ctx:]
    v, f_route, plan = float(tr.speed[W - 1]), float(W - 1), out['plan']
    seen, colls_all = set(), []
    t_arm = time.time()
    for j in range(W, n_end):
        tm = {}
        if kind == 'log':
            pose = cl.log_pose(tr, j)
            sub = cl.log_substeps(tr, j - 1, j)
            f_route, lat = float(j), 0.0
            v = sub[-1][4]
            st = dict(v=v, a=0.0, f=f_route, lat=0.0, sub_w=sub)
        else:
            pose, st = cl.advance(tr, route, poses[j - 1], f_route, v, plan, j, log_kin=g.log_kin)
            v, f_route = st['v'], st['f']
            sub = st['sub_w']
        poses[j] = pose
        cmd = route.command(pose, int(round(min(max(f_route, 0.0), scn.n - 1))))
        fr = scn.frame_of(pose, cmd)
        # ---- 特征 ----
        if kind == 'log':
            t0 = time.perf_counter()
            img = next(img_iter)
            t1 = sync()
            feats = runner.extract_feats(img)
            tm.update(t_io=round(t1 - t0, 4), t_backbone=round(sync() - t1, 4))
        elif kind == 'copy':
            feats = warm_feats[W - 1]
        elif kind == 'pred':
            t0 = time.perf_counter()
            E_new = g.rend.sim_frame(pose)[0]['ego2global']
            lay_new = scn.sim_layout(poses[j - 1], pose, wm, asc, dt_)
            t1 = sync()
            nxt, feats = src.predict(seq, E_hist + [E_new], lay_hist + [lay_new])
            tm.update(t_cond=round(t1 - t0, 4), t_pred=round(sync() - t1, 4))
            seq, E_hist, lay_hist = (seq + [nxt])[-src.ctx:], (E_hist + [E_new])[-src.ctx:], (lay_hist + [lay_new])[-src.ctx:]
        elif kind in ('3dgs', '3dgs_difix'):
            info = data.infos[pose.tmpl_idx]
            E_sim, E_l = pose_to_matrix(pose), np.asarray(g.rend.nd_frame(pose.tmpl_idx)['ego2global'], np.float64)
            same = bool(np.allclose(E_sim, E_l, rtol=0.0, atol=IDENTITY_ATOL))
            D = None if same else (E_sim @ np.linalg.inv(E_l)).tolist()
            files = {c: os.path.basename(info['cams'][c]['data_path']) for c in g.cams}
            odir = os.path.join(render_dir, scn.name, f'j{j:02d}')
            if kind == '3dgs_difix':
                odir = os.path.join(odir, 'raw')
            t0 = time.perf_counter()
            r = splat.render(files, D, odir)
            t1 = time.perf_counter()
            if kind == '3dgs_difix':
                ddir = os.path.join(render_dir, scn.name, f'j{j:02d}', 'difix')
                difix_frame(g.difix, odir, ddir, g.cams)
                t1d = sync()
                tm['t_difix'] = round(t1d - t1, 4)
                odir, t1 = ddir, t1d
            img, _, _ = data._load_images([os.path.join(odir, f'{c}.jpg') for c in g.cams])
            t2 = sync()
            feats = runner.extract_feats(img)
            tm.update(t_render=round(r['sec_render'], 4), t_save=round(r['sec_save'], 4), t_req=round(t1 - t0, 4),
                      t_io=round(t2 - t1, 4), t_backbone=round(sync() - t2, 4))
        else:
            raise ValueError(kind)
        t0 = sync()
        out = runner.step_frame(feats, fr)
        tm['t_uniad'] = round(sync() - t0, 4)
        if digests is not None:
            digests.append(out_digest(out))
        if feats_out is not None:
            feats_out[j] = feats
        plan = out['plan']
        gt = g.env.gt(pose)
        gts[j] = {k_: gt[k_] for k_ in ('boxes', 'det_name', 'num_lidar_pts', 'instance')}
        outs_box[j] = (out['boxes'], out['scores'], out['labels'])
        dac_end, _ = cl.footprint_on_drivable(gt['map']['drivable'])
        cs = cl.collisions_interval(g.gtw, data.infos[scn.ids[j - 1]]['token'], data.infos[scn.ids[j]]['token'], sub, dac_end, seen, scn.l2e_t)
        for c in cs:
            c['j'] = j
        colls_all += cs
        st_row = dict(st, lat=st['lat'] if kind != 'log' else 0.0)
        rows.append(scene_metrics_row(scn, arm, j, pose, st_row, out, gt, cmd, cs, tm))
    # ---- lag：折线要 j−2…j+1 的位姿（最后一步没有 j+1，不算）----
    for row in rows:
        j = row['j']
        polys = build_polys(g.gtw, data, poses, j, gts[j]) if (j + 1) in poses else None
        lm = lag_match(*outs_box[j], polys)
        row.update(lm if lm is not None else dict(lag_n_gt=-1, lag_n_match=0, lag_s=[], lag_perp=[]))
    # ---- 场景级 ----
    f_end = rows[-1]['f'] if rows else float(W - 1)
    log_arc = route.arc(min(n_end - 1, scn.n - 1)) - route.arc(W - 1)
    at_fault = [c for c in colls_all if c['at_fault']]
    summ = dict(arm=arm, kind=kind, scene=scn.name, si=scn.si, n_steps=len(rows), n_frames=scn.n, W=W,
                warm_digest=hashlib.sha1(''.join(warm_dig).encode()).hexdigest(),
                at_fault=bool(at_fault), at_fault_agent=any(c['group'] == 'agent' for c in at_fault),
                any_coll=bool(colls_all), first_at_fault_tau=(min(c['j'] for c in at_fault) - W + 1) * cl.DT if at_fault else None,
                coll_kinds=dict(collections.Counter(c['kind'] for c in colls_all)), colls=colls_all,
                dac=all(r['ego_dac'] for r in rows), dac_frac=float(np.mean([r['ego_dac'] for r in rows])) if rows else float('nan'),
                progress=(route.arc(max(f_end, 0.0)) - route.arc(W - 1)) / log_arc if log_arc > 1.0 else float('nan'),
                log_arc_m=round(float(log_arc), 2),
                max_abs_lat=float(max([abs(r['lat']) for r in rows] + [0.0])),
                off_route=any(abs(r['lat']) > OFF_ROUTE_M for r in rows),
                first_off_route_tau=next((r['tau'] for r in rows if abs(r['lat']) > OFF_ROUTE_M), None),
                stalled=any(r['v'] < 0.5 for r in rows if r['tau'] > 1.0) and (rows[-1]['v'] < 0.5 if rows else False),
                mean_v=float(np.mean([r['v'] for r in rows])) if rows else float('nan'),
                sec=time.time() - t_arm)
    for k_ in ('t_io', 't_backbone', 't_cond', 't_pred', 't_render', 't_save', 't_req', 't_difix', 't_uniad'):
        vs = [r[k_] for r in rows if k_ in r]
        if vs:
            summ['mean_' + k_] = float(np.mean(vs))
    return rows, summ


def difix_frame(runner, src_dir, dst_dir, cams):
    """一步的 6 路渲染图 → Difix → <dst_dir>/<CAM>.jpg。读写与 batch 组成都与 difix/difix_dir.py 相同（PIL RGB / 255、JPEG q95、
    按文件名排序的 6 张一批，同一个固定种子），所以同一组渲染图的输出与 O2 作业（A / B 表的 3dgs_difix 臂）相同（test_c_eval 的 difix 检查）。"""
    import difix_dir as dd
    names = sorted(f'{c}.jpg' for c in cams)
    ys = runner(torch.stack([dd.load(os.path.join(src_dir, n)) for n in names]))
    for n, y in zip(names, ys):
        dd.save(y, os.path.join(dst_dir, n))


def load_difix(weights, size):
    """DifixRunner（与 difix_dir.py 同一构造：UNet fp16、后验采样、种子 0）。size = 'native' 或 'HxW'。"""
    import difix_core as dc
    return dc.DifixRunner(weights, size='native' if size == 'native' else tuple(int(v) for v in size.lower().split('x')))


def real_images(data, scn, j0, j1):
    """日志帧 j0..j1-1 的真图（CPU 张量），与 data.iter_scene 同一条加载路径。"""
    for j in range(j0, j1):
        yield data.get_frame(scn.ids[j], load_images=True)['img']


class G:
    """全局句柄（runner / data / env / rend / gtw / W）。"""
    log_kin = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--tasks', required=True, help="任务：'si:group,si:group' 或一个文件（每行 'si group'）")
    ap.add_argument('--claims', default='', help='领片目录（distributed execution 多 worker）；空 = 不领片，按顺序全跑')
    ap.add_argument('--worker', default='0')
    ap.add_argument('--pred', action='append', default=None, help='name=path，可重复；缺省只运行不依赖预测器的对照臂')
    ap.add_argument('--pred-opt', action='append', default=[],
                    help="name=dtype:bf16|fp16|fp32,kv:0|1：给 --pred 的预测器加推理选项（缺省 = bf16 autocast + 稠密滑窗 = 原路径；同 a_eval.py）")
    ap.add_argument('--arms', default='', help="main 任务跑哪些臂（逗号分隔：log / copy / --pred 的 name）；缺省 = log,copy + 全部 --pred。"
                                               "已有 scene<NNN>__<arm>.done.json 的（场景, 臂）跳过，所以追加新预测器臂时只给新臂即可")
    ap.add_argument('--warmup', type=int, default=4, help='真图 warmup 帧数（≥ 预测器上下文 L=4）')
    ap.add_argument('--egostat', default='sim', choices=['sim', 'log'],
                    help="can_bus 的运动量（速度 / 加速度 / 角速度）：sim = 仿真运动学（正式口径）；log = 取时间对齐的日志帧（诊断：切断 UniAD 对自车状态的回灌）。"
                         "log 时非 log 臂的名字加后缀 @logstat")
    ap.add_argument('--ctx', type=int, default=4, help='TF ckpt（没有 ctx 字段）的 AR 上下文长度')
    ap.add_argument('--ckpt', default='hf')
    ap.add_argument('--max-steps', type=int, default=0, help='每臂最多闭环多少步（冒烟用；0 = 到场景结束）')
    ap.add_argument('--splat-configs', default='', help='JSON：场景名 → SplatAD config.yml（3dgs 任务用）')
    ap.add_argument('--splat-python', default=sys.executable)
    ap.add_argument('--neurad', default=_release_paths.NEURAD_ROOT)
    ap.add_argument('--splat-scripts', default=os.path.join(ROOT, 'splatad', 'scripts'))
    ap.add_argument('--torch-home', default=f'{BENCH}/splatad/torch_hub')
    ap.add_argument('--render-dir', default='', help='3dgs 渲染图落盘目录（缺省 <out>/../render）')
    ap.add_argument('--difix-weights', default=f'{BENCH}/downloads/difix_hf', help='3dgs_difix 任务：nvidia/difix 权重目录')
    ap.add_argument('--difix-size', default='576x1024', help='Difix 推理分辨率（与 O2 作业相同）')
    ap.add_argument('--det-boxes', default='', help='feat/det_layout.py 的 det_boxes.pkl：名字以 @det 结尾的预测器用它的框画 layout')
    a = ap.parse_args()

    from uniad_port.data import UniADValData, CAM_ORDER
    from uniad_port.runner import UniADRunner
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    os.makedirs(a.out, exist_ok=True)
    if os.path.isfile(a.tasks):
        tasks = [tuple(l.split()) for l in open(a.tasks) if l.strip() and not l.startswith('#')]
    else:
        tasks = [tuple(t.split(':')) for t in a.tasks.split(',') if t]
    tasks = [(int(si), grp) for si, grp in tasks]
    assert all(grp in ('main',) + RENDER_KINDS for _, grp in tasks), tasks
    g = G()
    g.W = a.warmup
    g.log_kin = a.egostat == 'log'
    g.data = data = UniADValData(num_prefetch=0)
    g.cams = tuple(CAM_ORDER)
    dev = torch.device('cuda:0')
    st = torch.load(STATS.format(ckpt=a.ckpt))
    assert st.get('scale') == 1.0 and st.get('ckpt') == a.ckpt, f'统计量不是 1.0×/{a.ckpt}'
    mean, std = [m.to(dev) for m in st['mean']], [s.to(dev) for s in st['std']]
    popts = {}
    for spec in a.pred_opt:
        pn, o = spec.split('=', 1)
        d = dict(kv_.split(':', 1) for kv_ in o.split(',') if kv_)
        assert set(d) <= {'dtype', 'kv'}, f'--pred-opt {spec}: 只认 dtype / kv'
        popts[pn] = dict(dtype=d.get('dtype', 'bf16'), kv=bool(int(d.get('kv', 0))))
    srcs = {}
    if any(grp == 'main' for _, grp in tasks):
        pred_specs = a.pred if a.pred is not None else DEFAULT_PREDS
        assert set(popts) <= {sp.split('=', 1)[0] for sp in pred_specs}, f'--pred-opt 里有没给 --pred 的名字: {set(popts) - {sp.split("=", 1)[0] for sp in pred_specs}}'
        for spec in pred_specs:
            name, path = spec.split('=', 1)
            assert name not in ('log', 'copy') + RENDER_KINDS, name
            srcs[name] = StepPredictor(name, path, mean, std, dev, a.ctx, **popts.get(name, {}))
            s_ = srcs[name]
            assert s_.ctx <= g.W, f'warmup {g.W} 帧不够预测器上下文 {s_.ctx}'
            print(f'[c {a.worker}] 预测器 {name}: {s_.path} step={s_.step} map_layout={s_.map_layout} ctx={s_.ctx} dtype={s_.dtype_name} kv={s_.kv}', flush=True)
    g.runner = UniADRunner(CKPTS.get(a.ckpt, a.ckpt), device='cuda:0', verbose=False)
    g.gtw = GTWorld([i['token'] for i in data.infos], cache_path=os.path.join(common.CACHE_DIR, 'gt_world_val.pkl'))
    g.env = OffLogEnv(data, g.gtw, MapGT())
    g.rend = SimLayoutRenderer(data.infos)
    g.rend_det = None
    if any(s_.det for s_ in srcs.values()):
        import det_layout
        assert a.det_boxes, '有 @det 预测器但没给 --det-boxes'
        assert not any(s_.det and s_.assoc for s_ in srcs.values()), '@det 不支持关联通道 ckpt'
        idx_det = det_layout.load_det_index(a.det_boxes, g.rend.idx)        # 缺帧在 Scene() 里按场景核对（不静默回退到 GT 框）
        g.rend_det = SimLayoutRenderer(data.infos, nd_index=idx_det, map_vecs=g.rend.map_vecs)
        print(f'[c {a.worker}] 检测器框：{idx_det["det_meta"]}', flush=True)
    g.difix = None
    splat_cfgs = json.load(open(a.splat_configs)) if a.splat_configs else {}
    render_dir = a.render_dir or os.path.join(os.path.dirname(os.path.abspath(a.out)), 'render')
    host = socket.gethostname()
    n_fail = 0
    main_arms = [x for x in (a.arms.split(',') if a.arms else ['log', 'copy'] + list(srcs)) if x]
    assert all(x in ('log', 'copy') or x in srcs for x in main_arms), f'--arms {main_arms} 里有没给 --pred 的预测器'

    def arm_done(si, arm):
        return os.path.join(a.out, f'scene{si:03d}__{arm}.done.json')

    def save_arm(si, scn, arm, rows, summ, t0, grp, s_):
        """一个 (场景, 臂) 的产物：jsonl（每步一行）+ done.json（场景级汇总）；先写 .tmp 再 replace，done.json 最后写。"""
        path = os.path.join(a.out, f'scene{si:03d}__{arm}.jsonl')
        with open(path + '.tmp', 'w') as f:
            for r in rows:
                f.write(json.dumps(r, default=jdefault) + '\n')
        os.replace(path + '.tmp', path)
        meta = dict(path=s_.path, step=s_.step, ctx=s_.ctx, map_layout=s_.map_layout, assoc=s_.assoc, det=s_.det) if s_ is not None else None
        summ['egostat'] = a.egostat
        dp = arm_done(si, arm)
        with open(dp + '.tmp', 'w') as f:
            json.dump(dict(args=vars(a), scene=scn.name, si=si, arm=arm, group=grp, host=host, seconds=time.time() - t0, pred=meta,
                           rows=len(rows), summary=summ), f, indent=1, default=jdefault)
        os.replace(dp + '.tmp', dp)

    sfx = '@logstat' if g.log_kin else ''

    def out_name(arm):
        return arm if arm == 'log' else arm + sfx

    for si, grp in tasks:
        todo = [x for x in (main_arms if grp == 'main' else [grp]) if not os.path.exists(arm_done(si, out_name(x)))]
        if not todo:
            continue
        tag = f'scene{si:03d}__{grp}'
        if a.claims:
            try:
                os.mkdir(os.path.join(a.claims, tag))
            except FileExistsError:
                continue
        t0 = time.time()
        msgs = []
        try:
            scn = Scene(g, si)
            assert scn.n > g.W + 1, f'{scn.name} 只有 {scn.n} 帧'
            if grp == 'main':
                # warmup 帧的真特征（log 臂顺带抽，其他臂复用；不跑 log 时只抽 W 帧）
                imgs = real_images(data, scn, 0, scn.n if 'log' in todo else g.W)
                warm_feats = [g.runner.extract_feats(next(imgs)) for _ in range(g.W)]
                for arm in todo:
                    kind = arm if arm in ('log', 'copy') else 'pred'
                    ta = time.time()
                    rows, summ = run_arm(g, scn, out_name(arm), kind, warm_feats, src=srcs.get(arm), img_iter=imgs if kind == 'log' else None,
                                         max_steps=a.max_steps)
                    save_arm(si, scn, out_name(arm), rows, summ, ta, grp, srcs.get(arm))
                    msgs.append(f'{out_name(arm)}:af={int(summ["at_fault"])},dac={summ["dac_frac"]:.2f},prog={summ["progress"]:.2f},v={summ["mean_v"]:.1f}')
            else:
                cfg = splat_cfgs.get(scn.name)
                assert cfg and os.path.isfile(cfg), f'{scn.name} 没有 SplatAD 重建 config（--splat-configs）'
                warm_feats = [g.runner.extract_feats(im) for im in real_images(data, scn, 0, g.W)]
                if grp == '3dgs_difix' and g.difix is None:
                    g.difix = load_difix(a.difix_weights, a.difix_size)
                sock = f'/tmp/splat_{os.getpid()}_{si}.sock'
                splat = SplatClient(cfg, sock, os.path.join(a.out, f'{tag}.server.log'), a.splat_python, a.neurad, a.splat_scripts, a.torch_home)
                try:
                    rows, summ = run_arm(g, scn, out_name(grp), grp, warm_feats, splat=splat, render_dir=render_dir, max_steps=a.max_steps)
                    summ['sec_server_load'] = splat.sec_load
                finally:
                    splat.close()
                save_arm(si, scn, out_name(grp), rows, summ, t0, grp, None)
                msgs.append(f'{grp}:af={int(summ["at_fault"])},dac={summ["dac_frac"]:.2f},prog={summ["progress"]:.2f},v={summ["mean_v"]:.1f}')
            print(f'[c {a.worker}] {tag} {scn.name} n={scn.n} {time.time() - t0:.0f}s  {" ".join(msgs)}', flush=True)
        except Exception:
            n_fail += 1
            traceback.print_exc()
            with open(os.path.join(a.out, f'failed_{tag}'), 'w') as f:
                f.write(traceback.format_exc())
            print(f'[c {a.worker}] {tag} FAILED（已完成的臂：{msgs}）', flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
