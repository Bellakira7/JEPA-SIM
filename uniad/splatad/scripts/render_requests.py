"""按请求渲染 SplatAD 重建在**任意自车位姿**下的 6 相机图（B 表的 3DGS 臂，设计稿 §5.4）。

请求文件（`uniad_bench/b_table/gen_render_requests.py` 生成，JSON）::

    {"scene": "scene-0103",
     "items": [{"out": "a04_lat+2.0/k3",                     # 输出子目录 <out>/<out>/<CAM>.jpg
                "files": {"CAM_FRONT": "n008-...__CAM_FRONT__1531....jpg", ...},   # 6 路相机的 nuScenes 原图文件名（UniAD 的相机名）
                "D": null | [[4x4]]}]}                       # null = 日志位姿；否则 = **global 系**里 "日志自车 → 仿真自车" 的刚体变换 E_sim · inv(E_log)

相机位姿 = D · c2w_global。渲染用的世界系是 dataparser 把 global 平移 / 旋转后的系（`dataparser_transform` = W，3×4，scale=1.0，刚体），
所以 c2w_render' = W · D · W⁻¹ · c2w_render（与 `offlog/layout_reproject.py` 的 "E_cam_sim = E_sim · inv(E_lidar_log) · E_cam_log" 同一个定义，
日志里激光→相机时刻的相对运动被保留）。D 为 null 时**不做任何变换**，与 `render_eval.py` 的平移 0 逐位相同。

CAM_BACK 在 dataparser 里被裁掉底部 80 行（自车车尾），存图时用原图补回（自车车身与位姿无关），和 render_eval 一致。
已存在的输出（6 张都在）跳过，所以可重入。

用法（在 neurad-studio 下，SplatAD 的 venv）:
  PYTHONPATH=$PWD ../venv/bin/python ../scripts/render_requests.py --config <run>/config.yml --requests req.json --out <dir>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_eval import to_uint8, save_full  # noqa: E402  (模块级只有函数定义)
from nerfstudio.utils.eval_utils import eval_setup  # noqa: E402

CAMS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')   # UniAD 的相机顺序


def to4(m):
    out = torch.eye(4, dtype=torch.float64)
    out[:3, :4] = torch.as_tensor(m, dtype=torch.float64)[:3, :4]
    return out


def moved_c2w(c2w34, D_global, W34):
    """渲染世界系里的相机位姿 c2w34 [3,4] → 自车做了 global 系刚体变换 D 之后的位姿 [3,4]（float64 计算，float32 输出）。"""
    W = to4(W34)
    c = to4(c2w34)
    return (W @ D_global @ torch.linalg.inv(W) @ c)[:3, :4].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--requests', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    req = json.load(open(args.requests))
    out = Path(args.out)
    items = [it for it in req['items'] if not all((out / it['out'] / f'{c}.jpg').exists() for c in CAMS)]
    print(f'[render] {req["scene"]}: {len(req["items"])} 个请求，待渲染 {len(items)}', flush=True)
    if not items:
        return

    t0 = time.time()
    config, pipeline, ckpt_path, step = eval_setup(Path(args.config), test_mode='test')
    model = pipeline.model
    model.eval()
    ds = pipeline.datamanager.eval_dataset
    cams = ds.cameras
    fnames = [Path(p) for p in ds._dataparser_outputs.image_filenames]
    W = ds._dataparser_outputs.dataparser_transform
    assert float(ds._dataparser_outputs.dataparser_scale) == 1.0
    is_kf = cams.metadata['is_key_frame'].squeeze(-1).bool()
    by_name = {}
    for i in torch.nonzero(is_kf).squeeze(-1).tolist():
        assert fnames[i].name not in by_name, f'重名图像 {fnames[i].name}'
        by_name[fnames[i].name] = i
    print(f'[render] 加载 {time.time() - t0:.0f}s step={step} 关键帧图像 {len(by_name)} 张', flush=True)

    n = 0
    t1 = time.time()
    with torch.no_grad():
        for it in items:
            D = None if it['D'] is None else torch.as_tensor(it['D'], dtype=torch.float64)
            if D is not None and torch.allclose(D, torch.eye(4, dtype=torch.float64), atol=0, rtol=0):
                D = None
            for c in CAMS:
                i = by_name[it['files'][c]]
                cam = cams[i:i + 1].to(model.device)
                if D is not None:
                    cam.camera_to_worlds = moved_c2w(cams.camera_to_worlds[i], D, W).unsqueeze(0).to(model.device)
                o = model.get_outputs_for_camera(cam)
                save_full(to_uint8(o['rgb']), fnames[i], out / it['out'] / f'{c}.jpg')
            n += 1
            if n % 20 == 0 or n == len(items):
                print(f'[render] {n}/{len(items)}  {(time.time() - t1) / n:.2f}s/请求(6 相机)', flush=True)
    print(f'[render] 完成 {n} 个请求，总 {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
