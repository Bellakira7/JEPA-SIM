"""C 表 3DGS 臂的在线渲染服务（SplatAD venv 里运行；UniAD 在主 venv，两边不能同进程）。

    cd <neurad-studio> && SPLAT_AUTHKEY=<hex> PYTHONPATH=$PWD <splatad venv>/bin/python c_table/splat_server.py \
        --config <run>/config.yml --address /tmp/x.sock --scripts-dir <uniad_bench>/splatad/scripts

加载一次重建（与 `splatad/scripts/render_requests.py` 同一个 eval_setup、同一个相机表），然后在 AF_UNIX socket 上等一个客户端
（multiprocessing.connection，authkey 由环境变量给），循环处理：
    {'op': 'render', 'files': {CAM: nuScenes 原图文件名}, 'D': None | 4×4 list, 'out_dir': 目录}
        → 6 张 <out_dir>/<CAM>.jpg，回 {'ok': True, 'sec_render', 'sec_save'}
    {'op': 'quit'} → 退出
渲染与落盘逐行复用 render_requests.py 的做法（`moved_c2w` 位姿共轭、D=None 不做任何变换、`render_eval.save_full` 补 CAM_BACK、JPEG q95），
所以同一个 (files, D) 与 B 表 3DGS 臂的渲染图逐像素相同（test_c_eval.py 的 splat 检查）。
"""
import argparse
import os
import sys
import time
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--address', required=True)
    ap.add_argument('--scripts-dir', required=True, help='含 render_requests.py / render_eval.py 的目录')
    args = ap.parse_args()
    sys.path.insert(0, args.scripts_dir)
    from render_eval import to_uint8, save_full            # noqa: E402
    from render_requests import CAMS, moved_c2w           # noqa: E402
    from nerfstudio.utils.eval_utils import eval_setup    # noqa: E402

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
    print(f'[splat_server] 加载 {time.time() - t0:.0f}s step={step} 关键帧图像 {len(by_name)} 张 config={args.config}', flush=True)

    key = bytes.fromhex(os.environ['SPLAT_AUTHKEY'])
    listener = Listener(args.address, family='AF_UNIX', authkey=key)
    print('SPLAT_SERVER_READY', flush=True)
    conn = listener.accept()
    n = 0
    with torch.no_grad():
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break
            if msg.get('op') == 'quit':
                break
            try:
                assert msg.get('op') == 'render', msg.get('op')
                D = None if msg['D'] is None else torch.as_tensor(msg['D'], dtype=torch.float64)
                if D is not None and torch.allclose(D, torch.eye(4, dtype=torch.float64), atol=0, rtol=0):
                    D = None
                out = Path(msg['out_dir'])
                rgbs = []
                torch.cuda.synchronize()
                ta = time.perf_counter()
                for c in CAMS:
                    i = by_name[msg['files'][c]]
                    cam = cams[i:i + 1].to(model.device)
                    if D is not None:
                        cam.camera_to_worlds = moved_c2w(cams.camera_to_worlds[i], D, W).unsqueeze(0).to(model.device)
                    o = model.get_outputs_for_camera(cam)
                    rgbs.append((i, to_uint8(o['rgb'])))
                torch.cuda.synchronize()
                tb = time.perf_counter()
                for (i, rgb), c in zip(rgbs, CAMS):
                    save_full(rgb, fnames[i], out / f'{c}.jpg')
                tc = time.perf_counter()
                n += 1
                conn.send(dict(ok=True, sec_render=tb - ta, sec_save=tc - tb, n=n))
            except Exception:
                conn.send(dict(ok=False, error=traceback.format_exc()))
    conn.close()
    listener.close()
    print(f'[splat_server] 退出，共渲染 {n} 个请求', flush=True)


if __name__ == '__main__':
    main()
