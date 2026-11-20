"""SplatAD 重建的评测与渲染：关键帧训练视角 PSNR/SSIM/LPIPS、横向平移视角渲染、渲染耗时。

对每个 nuScenes 关键帧（6 相机）:
  - 平移 0：渲染并对 GT 算 PSNR/SSIM/LPIPS（train_split=1.0 时就是训练视角）；
  - 平移 δ（自车右为正，水平面内垂直于 CAM_FRONT 朝向，与 neurad-studio 的 lane-shift FID 同一定义）：只渲染，
    记覆盖率（accumulation 均值，<1 的部分是背景色填的"空洞"）。
渲染图按原文件名存成 JPG（out/shift_{δ}/CAM_xxx/<原文件名>.jpg），可直接替换 nuScenes 的 samples/ 根目录喂 UniAD。
CAM_BACK 在 dataparser 里被裁掉底部 80 行（自车车尾），存图时把这 80 行用原图补回，保持 1600×900。

用法（在 neurad-studio 下）:
  PYTHONPATH=$PWD ../venv/bin/python ../scripts/render_eval.py --config <run>/config.yml --out <dir> \
      [--shifts -2 -1 -0.5 0.5 1 2] [--save-shifted all|first|none] [--max-keyframes N]
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nerfstudio.utils.eval_utils import eval_setup


def to_uint8(img):
    return (img.clamp(0, 1) * 255).round().byte().cpu().numpy()


def save_full(rgb_u8, orig_path, out_path):
    """渲染图高度不足 900（CAM_BACK 裁了底部）时，用原图同位置的像素补齐。"""
    h = rgb_u8.shape[0]
    orig = np.asarray(Image.open(orig_path).convert('RGB'))
    if h < orig.shape[0]:
        rgb_u8 = np.concatenate([rgb_u8, orig[h:]], axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_u8).save(out_path, quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shifts', type=float, nargs='*', default=[-2, -1, -0.5, 0.5, 1, 2])
    ap.add_argument('--save-shifted', choices=['all', 'first', 'none'], default='all')
    ap.add_argument('--save-unshifted', action='store_true', default=True)
    ap.add_argument('--max-keyframes', type=int, default=None)
    ap.add_argument('--sheet-keyframes', type=int, nargs='*', default=None,
                    help='拼对比图的关键帧序号（默认取首/中/尾）')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    config, pipeline, ckpt_path, step = eval_setup(Path(args.config), test_mode='test')
    t_load = time.time() - t0
    model = pipeline.model
    model.eval()
    dm = pipeline.datamanager
    ds = dm.eval_dataset
    cams = ds.cameras
    fnames = [Path(p) for p in ds._dataparser_outputs.image_filenames]
    is_kf = cams.metadata['is_key_frame'].squeeze(-1).bool()
    sensor_name = [p.parent.name for p in fnames]

    # 第 k 个关键帧 = 每个相机按时间排序后的第 k 个关键帧（nuScenes 每个 sample 6 相机各一帧关键帧）
    per_cam = defaultdict(list)
    for i in torch.nonzero(is_kf).squeeze(-1).tolist():
        per_cam[sensor_name[i]].append(i)
    cam_order = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    for c in cam_order:
        per_cam[c].sort(key=lambda i: float(cams.times[i]))
    n_kf = min(len(per_cam[c]) for c in cam_order)
    assert all(len(per_cam[c]) == n_kf for c in cam_order), {c: len(per_cam[c]) for c in cam_order}
    if args.max_keyframes:
        n_kf = min(n_kf, args.max_keyframes)
    sheet_kf = args.sheet_keyframes if args.sheet_keyframes is not None else sorted({0, n_kf // 2, n_kf - 1})

    shifts = [0.0] + [s for s in args.shifts if s != 0]
    metrics = []
    render_s = []  # 每个关键帧 6 相机（平移 0）的渲染耗时
    cover = defaultdict(list)
    sheets = defaultdict(dict)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for k in range(n_kf):
            idxs = [per_cam[c][k] for c in cam_order]
            # 平移方向用 CAM_FRONT 的水平朝向；nerfstudio 相机看向 -z
            c2w_front = cams.camera_to_worlds[idxs[0]]
            fwd = -c2w_front[:3, 2].clone()
            fwd[2] = 0
            fwd = fwd / fwd.norm()
            right = torch.stack([fwd[1], -fwd[0], torch.zeros(())])  # = cross(fwd, z)，与 lane-shift FID 同定义
            for s in shifts:
                torch.cuda.synchronize()
                t1 = time.time()
                outs = []
                for i in idxs:
                    cam = cams[i:i + 1].to(model.device)
                    if s != 0:
                        cam.camera_to_worlds[0, :2, 3] += s * right[:2].to(model.device)
                    outs.append(model.get_outputs_for_camera(cam))
                torch.cuda.synchronize()
                if s == 0:
                    render_s.append(time.time() - t1)
                for c, i, o in zip(cam_order, idxs, outs):
                    cover[s].append(float(o['accumulation'].mean()))
                    if s == 0:
                        batch = ds[i]
                        m, _ = model.get_image_metrics_and_images(o, batch)
                        metrics.append({'kf': k, 'cam': c, **{kk: float(v) for kk, v in m.items()}})
                    save = (s == 0 and args.save_unshifted) or args.save_shifted == 'all' or \
                           (args.save_shifted == 'first' and k == 0)
                    rgb = to_uint8(o['rgb'])
                    if save:
                        save_full(rgb, fnames[i], out / f'shift_{s:+.1f}' / c / (fnames[i].stem + '.jpg'))
                    if k in sheet_kf and c == 'CAM_FRONT':
                        sheets[k][s] = rgb
                        if s == 0:
                            sheets[k]['gt'] = np.asarray(Image.open(fnames[i]).convert('RGB'))[:rgb.shape[0]]
            print(f'kf {k + 1}/{n_kf}  render(6 cam, shift 0) {render_s[-1]:.3f}s  '
                  f'psnr {np.mean([m["psnr"] for m in metrics[-6:]]):.2f}', flush=True)

    # 对比图：每个选中的关键帧一行，CAM_FRONT 的 GT / 平移 0 / 各平移，缩到 1/4
    for k, row in sheets.items():
        keys = ['gt'] + shifts
        tiles = [Image.fromarray(row[key]).resize((400, 225 if row[key].shape[0] == 900 else 205)) for key in keys]
        sheet = Image.new('RGB', (400 * len(tiles), 225), 'white')
        for j, t in enumerate(tiles):
            sheet.paste(t, (400 * j, 0))
        sheet.save(out / f'sheet_front_kf{k:02d}.jpg', quality=90)
    # 6 相机拼图（平移 0 与 +1 m），只做第一个选中的关键帧
    k0 = sheet_kf[0]
    for s in [0.0, 1.0] if 1.0 in shifts else [0.0]:
        d = out / f'shift_{s:+.1f}'
        tiles = []
        for c in cam_order:
            p = d / c / (fnames[per_cam[c][k0]].stem + '.jpg')
            tiles.append(Image.open(p).resize((533, 300)) if p.exists() else Image.new('RGB', (533, 300)))
        grid = Image.new('RGB', (533 * 3, 600))
        for j, t in enumerate(tiles):
            grid.paste(t, (533 * (j % 3), 300 * (j // 3)))
        grid.save(out / f'grid6_kf{k0:02d}_shift{s:+.1f}.jpg', quality=90)

    def agg(key):
        v = np.array([m[key] for m in metrics if key in m])
        return {'mean': float(v.mean()), 'median': float(np.median(v)), 'n': int(v.size)} if v.size else None

    per_cam_psnr = {c: float(np.mean([m['psnr'] for m in metrics if m['cam'] == c])) for c in cam_order}
    rs = np.array(render_s[1:] if len(render_s) > 1 else render_s)  # 第一帧含 CUDA 预热，丢掉
    summary = {
        'config': args.config, 'ckpt': str(ckpt_path), 'step': int(step), 'n_keyframes': n_kf,
        'n_gaussians': int(model.num_points) if hasattr(model, 'num_points') else None,
        'psnr': agg('psnr'), 'ssim': agg('ssim'), 'lpips': agg('lpips'), 'psnr_per_cam': per_cam_psnr,
        'render_s_per_frame_6cam': {'mean': float(rs.mean()), 'median': float(np.median(rs)), 'n': int(rs.size)},
        'peak_mem_gb_render': torch.cuda.max_memory_allocated() / 2 ** 30,
        'coverage_mean_by_shift': {f'{s:+.1f}': float(np.mean(v)) for s, v in cover.items()},
        'load_s': t_load,
        'gpu': torch.cuda.get_device_name(0),
    }
    (out / 'metrics_per_image.json').write_text(json.dumps(metrics, indent=1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
