# -*- coding: utf-8 -*-
"""agreement.py 的合成数据单测。

  pytest:  /usr/local/bin/python -m pytest -q test_agreement.py
  直接跑:  /usr/local/bin/python test_agreement.py
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agreement as A  # noqa: E402
import flowsim_tab1  # noqa: E402

CFG = A.AgreementConfig()


# ------------------------------------------------------------------ 工具
def mk_boxes(centers, yaws=None, dims=(2.0, 4.5)):
    n = len(centers)
    b = np.zeros((n, 9))
    b[:, :2] = np.asarray(centers, float)
    b[:, 3], b[:, 4], b[:, 5] = dims[0], dims[1], 1.6
    if yaws is not None:
        b[:, 6] = yaws
    return b


def mk_out(boxes, scores=None, motion=None, seg=None, plan=None, meta=None):
    n = len(boxes)
    o = dict(boxes=np.asarray(boxes, float), scores=np.ones(n) if scores is None else np.asarray(scores, float),
             labels=np.zeros(n, np.int64), track_ids=np.arange(n))
    if motion is not None:
        o['motion'] = motion
    if seg is not None:
        o['seg'] = seg
    if plan is not None:
        o['plan'] = plan
    if meta is not None:
        o['meta'] = meta
    return o


def const_motion(boxes, vel, T=12, dt=0.5, K=1):
    """常速轨迹，绝对坐标（含中心）。vel [N,2]。"""
    boxes, vel = np.asarray(boxes), np.asarray(vel, float)
    t = (np.arange(1, T + 1) * dt)[None, None, :, None]
    traj = boxes[:, None, None, :2] + vel[:, None, None, :] * t
    traj = np.repeat(traj, K, axis=1)
    return dict(traj=traj, score=np.zeros((len(boxes), K)))


SEP = [(0, 0), (20, 5), (-15, 25), (30, -30), (-35, -10)]  # 互相 ≥ 17 m，平移 ≤ 8 m 内最近邻不会换人


def scene(rng, n=5):
    c = np.array(SEP[:n], float)
    yaws = rng.uniform(-math.pi, math.pi, n)
    return mk_boxes(c, yaws)


def rnd_seg(rng, H=200, W=200):
    return dict(lane=rng.random((H, W)).astype(np.float32), drivable=rng.random((H, W)).astype(np.float32))


# ------------------------------------------------------------------ 1. reference 对自己 = 满分
def test_self_agreement_is_perfect():
    rng = np.random.default_rng(0)
    b = scene(rng)
    motion = const_motion(b, rng.normal(size=(5, 2)) * 3, K=6)
    motion['score'] = rng.normal(size=(5, 6))
    plan = np.cumsum(rng.normal(size=(6, 2)), 0)
    o = mk_out(b, motion=motion, seg=rnd_seg(rng), plan=plan)
    f = A.frame_agreement(o, o, CFG)
    assert f['det_iou'] == 1.0
    assert f['det_cade'] == 0.0
    assert f['det_box_iou'] == pytest.approx(1.0, abs=1e-9)
    assert f['det_recall'] == f['det_precision'] == 1.0
    assert f['lane_iou'] == 1.0 and f['drive_iou'] == 1.0
    for k in ('motion_ade', 'motion_fde', 'motion_top1_ade', 'motion_wmean_ade', 'motion_mink_ade', 'motion_stay_ade'):
        assert f[k] == pytest.approx(0.0, abs=1e-9), k
    assert f['motion_cov'] == 1.0 and f['motion_n_pairs'] == 5
    assert f['plan_ade'] == 0.0 and f['plan_fde'] == 0.0


# ------------------------------------------------------------------ 2. 平移固定距离 -> cADE 精确等于该距离
@pytest.mark.parametrize('d,theta', [(1.3, 0.0), (0.7, 1.1), (4.0, -2.0), (8.0, 0.4)])
def test_shift_gives_exact_cade(d, theta):
    rng = np.random.default_rng(1)
    b = scene(rng)
    s = b.copy()
    s[:, 0] += d * math.cos(theta)
    s[:, 1] += d * math.sin(theta)
    for match in ('nn', 'hungarian'):
        cfg = A.AgreementConfig(det_match=match)
        f = A.frame_agreement(mk_out(b), mk_out(s), cfg)
        assert f['det_cade'] == pytest.approx(d, abs=1e-9), match
    assert f['det_cade_ref2src'] == pytest.approx(d, abs=1e-9)
    assert f['det_cade_src2ref'] == pytest.approx(d, abs=1e-9)


# ------------------------------------------------------------------ 3. IoU 解析值 + 栅格 vs 精确多边形
@pytest.mark.parametrize('yaw', [0.0, 0.7, -2.2])
def test_iou_half_shift_is_one_third(yaw):
    # 2 x 4 的矩形沿局部 y(长边) 平移 2 m: 交 = 2*2, 并 = 8+8-4 => 1/3
    b = mk_boxes([(5.0, -3.0)], [yaw], dims=(2.0, 4.0))
    s = b.copy()
    s[0, 0] += -2.0 * math.sin(yaw)
    s[0, 1] += 2.0 * math.cos(yaw)
    assert A.box_pair_iou(b[0], s[0]) == pytest.approx(1 / 3, abs=1e-9)
    f = A.frame_agreement(mk_out(b), mk_out(s), CFG)
    assert f['det_iou'] == pytest.approx(1 / 3, abs=0.01)


def test_footprint_dim_order():
    # dim0 沿局部 x: yaw=0 时 dim0=2 在 x 方向、dim1=6 在 y 方向。平移 3 m 沿 y 应是 1/3；沿 x 平移 1 m 应是 1/3 (2 宽)
    b = mk_boxes([(0.0, 0.0)], [0.0], dims=(2.0, 6.0))
    sy = b.copy(); sy[0, 1] += 3.0
    sx = b.copy(); sx[0, 0] += 1.0
    assert A.box_pair_iou(b[0], sy[0]) == pytest.approx(1 / 3, abs=1e-9)
    assert A.box_pair_iou(b[0], sx[0]) == pytest.approx(1 / 3, abs=1e-9)


def test_raster_vs_exact_and_shapely():
    rng = np.random.default_rng(3)
    for _ in range(25):
        b1 = mk_boxes([rng.uniform(-10, 10, 2)], rng.uniform(-3, 3, 1), dims=rng.uniform(1.5, 5, 2))
        b2 = mk_boxes([b1[0, :2] + rng.uniform(-2.5, 2.5, 2)], rng.uniform(-3, 3, 1), dims=rng.uniform(1.5, 5, 2))
        exact = A.box_pair_iou(b1[0], b2[0])
        f = A.frame_agreement(mk_out(b1), mk_out(b2), CFG)
        assert f['det_iou'] == pytest.approx(exact, abs=0.012)
        try:
            from shapely.geometry import Polygon
        except ImportError:
            continue
        p1 = Polygon(A.rect_corners(b1[0, 0], b1[0, 1], b1[0, 3], b1[0, 4], b1[0, 6]))
        p2 = Polygon(A.rect_corners(b2[0, 0], b2[0, 1], b2[0, 3], b2[0, 4], b2[0, 6]))
        assert exact == pytest.approx(p1.intersection(p2).area / p1.union(p2).area, abs=1e-9)


# ------------------------------------------------------------------ 4. seg
def test_seg_opposite_is_zero_and_identical_is_one():
    rng = np.random.default_rng(4)
    m = (rng.random((200, 200)) > 0.5).astype(np.float32)
    ref = mk_out(mk_boxes([(0, 0)]), seg=dict(lane=m, drivable=m))
    src = mk_out(mk_boxes([(0, 0)]), seg=dict(lane=1 - m, drivable=1 - m))
    f = A.frame_agreement(ref, src, CFG)
    assert f['lane_iou'] == 0.0 and f['drive_iou'] == 0.0
    f = A.frame_agreement(ref, ref, CFG)
    assert f['lane_iou'] == 1.0 and f['drive_iou'] == 1.0


def test_seg_partial_overlap_analytic_and_logit_and_crop():
    a = np.zeros((200, 200), np.float32); a[:, 50:100] = 1      # 50 列
    b = np.zeros((200, 200), np.float32); b[:, 75:125] = 1      # 与 a 重叠 25 列
    cfg = A.AgreementConfig(seg_crop=False)
    f = A.frame_agreement(mk_out(mk_boxes([(0, 0)]), seg=dict(lane=a, drivable=a)),
                          mk_out(mk_boxes([(0, 0)]), seg=dict(lane=b, drivable=b)), cfg)
    assert f['lane_iou'] == pytest.approx(25 / 75)
    # logit 输入与概率输入等价
    la, lb = np.where(a > 0, 5.0, -5.0), np.where(b > 0, 5.0, -5.0)
    meta = dict(seg_kind='logit')
    f2 = A.frame_agreement(mk_out(mk_boxes([(0, 0)]), seg=dict(lane=la, drivable=la), meta=meta),
                           mk_out(mk_boxes([(0, 0)]), seg=dict(lane=lb, drivable=lb), meta=meta), cfg)
    assert f2['lane_iou'] == pytest.approx(25 / 75)
    # 只在 ±50 m 之外的格子上有内容：裁剪后两侧都空 -> NaN（无信息），不裁剪时 IoU=1
    edge = np.zeros((200, 200), np.float32); edge[:2, :] = 1     # 最外两行，|坐标| > 50 m
    o = mk_out(mk_boxes([(0, 0)]), seg=dict(lane=edge, drivable=edge))
    assert math.isnan(A.frame_agreement(o, o, A.AgreementConfig(seg_crop=True))['lane_iou'])
    assert A.frame_agreement(o, o, cfg)['lane_iou'] == 1.0


# ------------------------------------------------------------------ 5. 空检测 / N=0 / 缺字段
def test_src_no_detection_and_symmetry_of_empty_convention():
    rng = np.random.default_rng(5)
    b = scene(rng)
    full, empty = mk_out(b), mk_out(np.zeros((0, 9)))
    f1 = A.frame_agreement(full, empty, CFG)      # src 空
    f2 = A.frame_agreement(empty, full, CFG)      # ref 空
    for f in (f1, f2):
        assert f['det_iou'] == 0.0
        assert f['det_cade'] == CFG.cade_empty_m
        assert f['det_one_empty'] == 1 and f['det_both_empty'] == 0
    assert f1['det_recall'] == 0.0 and math.isnan(f1['det_precision'])
    assert math.isnan(f1['motion_ade'])           # 没有配对 -> 无信息
    # src 没有任何 agent（连 motion 字段都没有）：'stay' 口径把 ref 的 agent 全记为未配上；'skip' 口径仍是 NaN
    rng = np.random.default_rng(55)
    b5 = scene(rng)
    ref_m = mk_out(b5, motion=const_motion(b5, np.full((5, 2), 2.0)))
    f4 = A.frame_agreement(ref_m, empty, CFG)
    t6 = np.arange(1, 13) * 0.5
    assert math.isnan(f4['motion_ade']) and f4['motion_stay_ade'] == pytest.approx((2.0 * math.sqrt(2) * t6).mean())
    f5 = A.frame_agreement(ref_m, empty, A.AgreementConfig(motion_unmatched='stay'))
    assert f5['motion_ade'] == pytest.approx(f4['motion_stay_ade'])
    # 有框却没有 motion 字段 = 数据缺失，不是 "没有 agent"
    assert math.isnan(A.frame_agreement(ref_m, mk_out(b5), A.AgreementConfig(motion_unmatched='stay'))['motion_ade'])
    # 两侧都空 -> 无信息 (NaN)，不是满分也不是零分
    f3 = A.frame_agreement(empty, empty, CFG)
    assert math.isnan(f3['det_iou']) and math.isnan(f3['det_cade']) and f3['det_both_empty'] == 1


def test_n0_and_missing_fields_do_not_crash():
    for ref, src in [({}, {}), (dict(boxes=None), dict(boxes=None)),
                     (dict(boxes=np.zeros((0, 9)), scores=np.zeros(0), labels=np.zeros(0, np.int64),
                           motion=dict(traj=np.zeros((0, 6, 12, 2)), score=np.zeros((0, 6))),
                           seg=None, plan=None), dict(boxes=np.zeros(0)))]:
        f = A.frame_agreement(ref, src, CFG)
        assert math.isnan(f['det_iou']) and math.isnan(f['lane_iou']) and math.isnan(f['plan_ade'])
        agg = A.aggregate([f, f])
        assert agg['det_iou']['n'] == 0 and agg['n_frames'] == 2
    # 只有 box，没有 motion / seg / plan
    rng = np.random.default_rng(6)
    o = mk_out(scene(rng))
    f = A.frame_agreement(o, o, CFG)
    assert f['det_iou'] == 1.0 and math.isnan(f['motion_ade']) and math.isnan(f['drive_iou'])
    # 对不上的 N 要报错而不是静默算错
    bad = mk_out(scene(rng), motion=dict(traj=np.zeros((3, 1, 12, 2)), score=np.zeros((3, 1))))
    with pytest.raises(ValueError):
        A.frame_agreement(bad, bad, CFG)


# ------------------------------------------------------------------ 6. motion
def _two_mode_agent(p_ref, p_src):
    """1 个 agent、2 个模式：A 沿 +x 每步 1 m，B 沿 +y 每步 1 m（相对，中心在原点）。"""
    T = 12
    t = np.arange(1, T + 1, dtype=float)
    A_ = np.stack([t, 0 * t], -1)
    B_ = np.stack([0 * t, t], -1)
    traj = np.stack([A_, B_])[None]                     # [1,2,T,2]
    ref = mk_out(mk_boxes([(0, 0)]), motion=dict(traj=traj, score=np.array([p_ref])))
    src = mk_out(mk_boxes([(0, 0)]), motion=dict(traj=traj.copy(), score=np.array([p_src])))
    return ref, src, t


def test_motion_mode_selection():
    ref, src, t = _two_mode_agent([0.7, 0.3], [0.3, 0.7])       # ref top1=A, src top1=B
    f = A.frame_agreement(ref, src, CFG)
    assert f['motion_top1_ade'] == pytest.approx(math.sqrt(2) * t.mean())
    assert f['motion_top1_fde'] == pytest.approx(math.sqrt(2) * 12)
    assert f['motion_mink_ade'] == pytest.approx(0.0) and f['motion_mink_fde'] == pytest.approx(0.0)
    assert f['motion_wmean_ade'] == pytest.approx(0.4 * math.sqrt(2) * t.mean())
    assert f['motion_wmean_fde'] == pytest.approx(0.4 * math.sqrt(2) * 12)
    assert f['motion_ade'] == f['motion_top1_ade']             # 默认主定义 = top1
    f = A.frame_agreement(ref, src, A.AgreementConfig(motion_mode='mink'))
    assert f['motion_ade'] == pytest.approx(0.0)
    # log-softmax 打分 与 概率打分 在 top1 / wmean 上等价
    ref2, src2, _ = _two_mode_agent(np.log([0.7, 0.3]), np.log([0.3, 0.7]))
    f2 = A.frame_agreement(ref2, src2, CFG)
    assert f2['motion_wmean_ade'] == pytest.approx(f['motion_wmean_ade'])
    assert f2['motion_top1_ade'] == pytest.approx(f['motion_top1_ade'])


def test_motion_top1_picks_argmax_three_modes():
    # 3 个模式 A(+x) B(+y) C(-x)；ref 的 argmax=A，src 的 argmax=B -> top1 误差 = |A-B| = sqrt(2)*t
    # （argmin 会选 C vs A，误差 2t，能区分）
    t = np.arange(1, 13, dtype=float)
    A_, B_, C_ = (np.stack([t, 0 * t], -1), np.stack([0 * t, t], -1), np.stack([-t, 0 * t], -1))
    traj = np.stack([A_, B_, C_])[None]
    ref = mk_out(mk_boxes([(0, 0)]), motion=dict(traj=traj, score=np.array([[0.5, 0.3, 0.2]])))
    src = mk_out(mk_boxes([(0, 0)]), motion=dict(traj=traj.copy(), score=np.array([[0.2, 0.5, 0.3]])))
    f = A.frame_agreement(ref, src, CFG)
    assert f['motion_top1_ade'] == pytest.approx(math.sqrt(2) * t.mean())
    # mink: src 里最接近 ref top1(A) 的就是 A 本身 -> 0
    assert f['motion_mink_ade'] == pytest.approx(0.0) and f['motion_mink_fde'] == pytest.approx(0.0)


def test_motion_frame_relative_vs_absolute():
    b = mk_boxes([(10.0, 0.0)])
    vel = np.array([[3.0, 1.0]])
    s = b.copy(); s[0, 0] += 1.0                         # src 的框右移 1 m，轨迹整体右移 1 m
    ref = mk_out(b, motion=const_motion(b, vel))
    src = mk_out(s, motion=const_motion(s, vel))
    assert A.frame_agreement(ref, src, A.AgreementConfig(motion_frame='relative'))['motion_ade'] == pytest.approx(0.0, abs=1e-12)
    assert A.frame_agreement(ref, src, A.AgreementConfig(motion_frame='absolute'))['motion_ade'] == pytest.approx(1.0)
    # 速度不同: 相对误差 = |dv| * t
    src2 = mk_out(b, motion=const_motion(b, vel + np.array([[0.0, 2.0]])))
    f = A.frame_agreement(ref, src2, CFG)
    t = np.arange(1, 13) * 0.5
    assert f['motion_ade'] == pytest.approx((2.0 * t).mean())
    assert f['motion_fde'] == pytest.approx(2.0 * 6.0)


def test_motion_traj_kind_relative_input_equivalent_to_absolute():
    rng = np.random.default_rng(12)
    b = scene(rng)
    vel = rng.normal(size=(5, 2)) * 3
    ab = const_motion(b, vel)
    rel = dict(traj=ab['traj'] - b[:, None, None, :2], score=ab['score'])
    s_b = b.copy(); s_b[:, :2] += 0.4
    s_ab = const_motion(s_b, vel + 0.5)
    s_rel = dict(traj=s_ab['traj'] - s_b[:, None, None, :2], score=s_ab['score'])
    f1 = A.frame_agreement(mk_out(b, motion=ab), mk_out(s_b, motion=s_ab), A.AgreementConfig(motion_frame='absolute'))
    f2 = A.frame_agreement(mk_out(b, motion=rel), mk_out(s_b, motion=s_rel),
                           A.AgreementConfig(motion_frame='absolute', motion_traj_kind='relative'))
    assert f1['motion_ade'] == pytest.approx(f2['motion_ade']) and f1['motion_fde'] == pytest.approx(f2['motion_fde'])


def test_motion_unmatched_policy_and_baseline():
    b_ref = mk_boxes([(0, 0), (20, 0)])
    b_src = mk_boxes([(0.5, 0)])                         # 只配上第一个
    vel = np.array([[0.0, 0.0], [2.0, 0.0]])             # 第二个 ref agent 以 2 m/s 前进
    ref = mk_out(b_ref, motion=const_motion(b_ref, vel))
    src = mk_out(b_src, motion=const_motion(b_src, vel[:1]))
    t = np.arange(1, 13) * 0.5
    stay_ade, stay_fde = (2.0 * t).mean(), 2.0 * 6.0
    f = A.frame_agreement(ref, src, A.AgreementConfig(motion_unmatched='skip'))
    assert f['motion_n_pairs'] == 1 and f['motion_ade'] == pytest.approx(0.0, abs=1e-12)
    assert f['motion_cov'] == 0.5
    f = A.frame_agreement(ref, src, A.AgreementConfig(motion_unmatched='stay'))
    assert f['motion_n_pairs'] == 2
    assert f['motion_ade'] == pytest.approx(stay_ade / 2) and f['motion_fde'] == pytest.approx(stay_fde / 2)
    # 无信息基线只依赖 ref：两个 ref agent 的 "相对 top1 轨迹 vs 原地不动"
    assert f['motion_base_ade'] == pytest.approx(stay_ade / 2) and f['motion_base_fde'] == pytest.approx(stay_fde / 2)
    # 门限外不配对
    far = mk_out(mk_boxes([(5.0, 0)]), motion=const_motion(mk_boxes([(5.0, 0)]), vel[:1]))
    assert A.frame_agreement(ref, far, CFG)['motion_n_pairs'] == 0


# ------------------------------------------------------------------ 7. planning 手算对拍
def test_planning_hand_computed():
    ref = np.stack([np.zeros(6), np.arange(1, 7.0)], 1)              # (0,1) ... (0,6)
    k = np.arange(1, 7.0)[:, None]
    src = ref + k * np.array([[0.3, 0.4]])                          # 误差 0.5k
    f = A.frame_agreement(mk_out(mk_boxes([(0, 0)]), plan=ref), mk_out(mk_boxes([(0, 0)]), plan=src), CFG)
    assert f['plan_ade'] == pytest.approx(0.5 * 3.5)               # mean(0.5*[1..6]) = 1.75
    assert f['plan_fde'] == pytest.approx(3.0)
    f = A.frame_agreement(mk_out(mk_boxes([(0, 0)]), plan=ref), mk_out(mk_boxes([(0, 0)]), plan=src),
                          A.AgreementConfig(plan_steps=3))
    assert f['plan_ade'] == pytest.approx(1.0) and f['plan_fde'] == pytest.approx(1.5)
    # [1,6,2] 形状与 5 列 (UniAD gaussian 参数) 也接受
    f = A.frame_agreement(mk_out(mk_boxes([(0, 0)]), plan=ref[None]),
                          mk_out(mk_boxes([(0, 0)]), plan=np.concatenate([src, np.ones((6, 3))], 1)), CFG)
    assert f['plan_ade'] == pytest.approx(1.75)


# ------------------------------------------------------------------ 8. 性质
def test_symmetry_permutation_invariance_range():
    rng = np.random.default_rng(7)
    for _ in range(10):
        n1, n2 = rng.integers(1, 8, 2)
        a = mk_boxes(rng.uniform(-40, 40, (n1, 2)), rng.uniform(-3, 3, n1), dims=(1.9, 4.4))
        b = mk_boxes(rng.uniform(-40, 40, (n2, 2)), rng.uniform(-3, 3, n2), dims=(1.9, 4.4))
        fab = A.frame_agreement(mk_out(a), mk_out(b), CFG)
        fba = A.frame_agreement(mk_out(b), mk_out(a), CFG)
        for k in ('det_iou', 'det_cade', 'det_box_iou'):
            assert fab[k] == pytest.approx(fba[k], abs=1e-12), k           # 对称
        assert 0.0 <= fab['det_iou'] <= 1.0
        perm = rng.permutation(n1)
        fp = A.frame_agreement(mk_out(a[perm]), mk_out(b), CFG)
        for k in ('det_iou', 'det_cade', 'det_box_iou'):
            assert fp[k] == pytest.approx(fab[k], abs=1e-12), k            # 框顺序无关


def test_iou_monotone_and_cade_linear_in_shift():
    rng = np.random.default_rng(8)
    b = scene(rng)
    ds = np.linspace(0, 6, 13)
    ious, cades = [], []
    for d in ds:
        s = b.copy(); s[:, 0] += d
        f = A.frame_agreement(mk_out(b), mk_out(s), CFG)
        ious.append(f['det_iou']); cades.append(f['det_cade'])
    ious, cades = np.array(ious), np.array(cades)
    assert ious[0] == 1.0 and (np.diff(ious) <= 1e-12).all() and ious[-1] < 0.2
    assert np.allclose(cades, ds, atol=1e-9)
    assert ((0 <= ious) & (ious <= 1)).all()


def test_score_threshold_and_range_filter():
    b = mk_boxes([(0, 0), (20, 0), (60, 0), (0, -70)])
    s = np.array([0.9, 0.2, 0.9, 0.9])
    o = mk_out(b, scores=s)
    f = A.frame_agreement(o, o, CFG)
    assert f['det_n_ref'] == 1                           # 0.2 < 0.3 丢弃；(60,0)、(0,-70) 超出 ±50 m 丢弃
    f = A.frame_agreement(o, o, A.AgreementConfig(score_thr=0.1))
    assert f['det_n_ref'] == 2
    f = A.frame_agreement(o, o, A.AgreementConfig(score_thr=0.1, range_m=80.0))
    assert f['det_n_ref'] == 4
    # 分数低于阈值的 src 框等于没有该框
    src_low = mk_out(b[:1], scores=[0.1])
    f = A.frame_agreement(mk_out(b[:1]), src_low, CFG)
    assert f['det_iou'] == 0.0 and f['det_n_src'] == 0


def test_seg_lane_fallback_union_of_parts():
    d = np.zeros((200, 200), np.float32); d[:, 10:20] = 1
    c = np.zeros((200, 200), np.float32); c[50:60, :] = 1
    ref = mk_out(mk_boxes([(0, 0)]), seg=dict(divider=d, crossing=c))
    src = mk_out(mk_boxes([(0, 0)]), seg=dict(lane=np.maximum(d, c)))
    assert A.frame_agreement(ref, src, A.AgreementConfig(seg_crop=False))['lane_iou'] == 1.0


def test_fuzz_no_crash_and_ranges():
    rng = np.random.default_rng(11)
    for _ in range(60):
        def rnd():
            n = int(rng.integers(0, 12))
            b = rng.uniform(-70, 70, (n, 9)); b[:, 3:6] = rng.uniform(0.3, 12, (n, 3)); b[:, 6] = rng.uniform(-4, 4, n)
            if n and rng.random() < 0.3:
                b[rng.integers(0, n), :3] = np.nan                       # 坏框
            K, T = int(rng.integers(1, 7)), 12
            traj = rng.normal(size=(n, K, T, 5)).cumsum(2)
            if n and rng.random() < 0.3:
                traj[rng.integers(0, n)] = np.nan                        # 坏轨迹
            return mk_out(b, scores=rng.random(n), motion=dict(traj=traj, score=rng.normal(size=(n, K))),
                          plan=rng.normal(size=(6, 2)).cumsum(0))
        r, s = rnd(), rnd()
        for cfg in (CFG, A.AgreementConfig(motion_unmatched='stay', motion_mode='wmean', det_match='hungarian')):
            if r['motion']['traj'].shape[1] != s['motion']['traj'].shape[1]:
                continue
            f = A.frame_agreement(r, s, cfg)
            for k in ('det_iou', 'det_box_iou', 'det_recall', 'det_precision'):
                assert math.isnan(f[k]) or 0.0 <= f[k] <= 1.0 + 1e-12, (k, f[k])
            for k in ('det_cade', 'motion_ade', 'motion_fde', 'plan_ade'):
                assert math.isnan(f[k]) or f[k] >= 0.0, (k, f[k])
            A.aggregate([f, f])


# ------------------------------------------------------------------ 9. 聚合 / 表
def test_aggregate_mean_median_n_pooled():
    # 3 帧: IoU 逐帧 (交,并) = (2,4),(1,1),(0,5)  -> 帧均值 (0.5+1+0)/3，池化 3/10；第 4 帧无信息 NaN
    fr = []
    for inter, union in [(2, 4), (1, 1), (0, 5)]:
        fr.append({'det_iou': inter / union, 'det_n_ref': 2, '_pool': {'det_iou': (inter, union)}})
    fr.append({'det_iou': float('nan'), 'det_n_ref': 0, '_pool': {'det_iou': (0.0, 0.0)}})
    agg = A.aggregate(fr)
    s = agg['det_iou']
    assert agg['n_frames'] == 4 and s['n'] == 3
    assert s['mean'] == pytest.approx(0.5) and s['median'] == pytest.approx(0.5)
    assert s['pooled'] == pytest.approx(3 / 10)
    assert s['std'] == pytest.approx(np.std([0.5, 1, 0], ddof=1))
    assert agg['counts']['det_n_ref'] == 6
    assert A.aggregate([])['n_frames'] == 0


def test_aggregate_on_real_frames_matches_manual():
    rng = np.random.default_rng(9)
    frames, cades = [], []
    for i in range(6):
        b = scene(rng); s = b.copy(); s[:, 1] += 0.5 * (i + 1)
        frames.append(A.frame_agreement(mk_out(b), mk_out(s), CFG)); cades.append(0.5 * (i + 1))
    agg = A.aggregate(frames)
    assert agg['det_cade']['mean'] == pytest.approx(np.mean(cades))
    assert agg['det_cade']['median'] == pytest.approx(np.median(cades))
    assert agg['det_cade']['pooled'] == pytest.approx(np.mean(cades))     # 每帧框数相同 -> 池化 = 帧均值


def test_make_table_columns_and_bold():
    t = A.make_table(flowsim_tab1.TAB1)
    lines = t.split('\n')
    hdr = [c.strip() for c in lines[0].strip('|').split('|')]
    assert hdr[:2] == ['Source', 'Δt (s)']
    assert [h.split('↑')[0].split('↓')[0].strip() for h in hdr[2:10]] == \
        ['Det IoU', 'Det cADE', 'Lane', 'Drive.', 'Mot ADE', 'Mot FDE', 'Plan ADE', 'Plan FDE']
    assert '**0.75**' in t and '**0.30**' in t and '**5.87**' in t        # 0.5 s 块里的最优值
    assert t.count('\n') == 2 + sum(len(v) for v in flowsim_tab1.TAB1.values()) - 1
    # aggregate() 的输出 + (source, dt) 元组键 + 缺项 -> '–'
    rng = np.random.default_rng(10)
    b = scene(rng)
    agg = A.aggregate([A.frame_agreement(mk_out(b), mk_out(b), CFG)])
    t2 = A.make_table({('ref-vs-ref', 0.5): agg}, show_n=True)
    row = t2.split('\n')[-1]
    assert '| ref-vs-ref | 0.5 | **1.00** | **0.00** | – | – | – | – | – | – | 1 |' == row


if __name__ == '__main__':
    fails = 0
    names = sorted(n for n in dir() if n.startswith('test_'))
    # 参数化用例交给 pytest；这里直接跑时展开
    import inspect
    for n in names:
        fn = globals()[n]
        marks = getattr(fn, 'pytestmark', [])
        cases = [m.args[1] for m in marks if getattr(m, 'name', '') == 'parametrize']
        argn = [m.args[0].split(',') for m in marks if getattr(m, 'name', '') == 'parametrize']
        try:
            if cases:
                for c in cases[0]:
                    c = c if isinstance(c, tuple) else (c,)
                    fn(**dict(zip([a.strip() for a in argn[0]], c)))
            else:
                fn()
            print(f'PASS {n}')
        except Exception as e:  # noqa: BLE001
            fails += 1
            import traceback
            traceback.print_exc()
            print(f'FAIL {n}: {e!r}')
    print(f'\n{len(names) - fails}/{len(names)} passed')
    sys.exit(1 if fails else 0)
