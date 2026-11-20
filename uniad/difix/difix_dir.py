"""对一个渲染根目录整体做 Difix 后处理，输出到新根目录（相对路径不变），使 `--render-roots difix=<dst>` / A 表 log_root 可直接用。

按"场景目录"分片（<src>/<scene_name>/...），每个场景做完写 <dst>/<scene_name>/.difix_done_<tag>（可重入：已完成的场景跳过，
未完成的场景整场重做——单张图先写 .tmp 再 os.replace，不会留半张图）。

  --pattern  场景目录下的相对 glob（默认 '**/*.jpg'；A 表只要 δ=0：'shift_+0.0/**/*.jpg'）
  --tag      完成标记名 <dst>/<scene>/.difix_done_<tag>：同一个 dst 可以接收多个来源（如 B 表横向 δ 与航向 ψ 的渲染图，相对路径互不重叠）
  --scenes   只处理这些场景（逗号分隔）；--shard i --nshard n 按场景名排序后轮转分片
JPEG 用 quality=95（与 splatad/scripts/render_requests.py、render_eval.py 存渲染图一致）。

用法: CUDA_VISIBLE_DEVICES=0 python -I difix_dir.py --weights <difix_hf> --src <root> --dst <root> [--pattern ...] [--shard i --nshard n]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import difix_core as dc  # noqa: E402


def load(p):
    a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def save(t, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    a = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    tmp = p + '.tmp'
    Image.fromarray(a).save(tmp, format='JPEG', quality=95)
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--pattern', default='**/*.jpg')
    ap.add_argument('--scenes', default='')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--size', default='576x1024', help="native（补边到 8 的倍数，原分辨率）或 HxW（如 576x1024，Difix 训练分辨率）")
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--tag', default='all')
    a = ap.parse_args()
    assert os.path.realpath(a.src) != os.path.realpath(a.dst)
    size = 'native' if a.size == 'native' else tuple(int(v) for v in a.size.lower().split('x'))
    scenes = sorted(d for d in os.listdir(a.src) if os.path.isdir(os.path.join(a.src, d)))
    if a.scenes:
        want = set(a.scenes.split(','))
        scenes = [s for s in scenes if s in want]
    scenes = scenes[a.shard::a.nshard]
    done_name = f'.difix_done_{a.tag}'
    todo = [s for s in scenes if not os.path.exists(os.path.join(a.dst, s, done_name))]
    print(f'DIFIX_DIR shard {a.shard}/{a.nshard}: scenes={len(scenes)} todo={len(todo)} size={size} src={a.src} dst={a.dst}', flush=True)
    if not todo:
        return
    runner = dc.DifixRunner(a.weights, size=size)
    for s in todo:
        t0 = time.time()
        files = sorted(glob.glob(os.path.join(a.src, s, a.pattern), recursive=True))
        files = [f for f in files if not f.endswith('.tmp')]
        assert files, f'{s}: {a.pattern} 没有匹配的图'
        for i in range(0, len(files), a.batch):
            chunk = files[i:i + a.batch]
            imgs = [load(f) for f in chunk]
            # 同尺寸的才能拼 batch（nuScenes 6 相机都是 1600×900）
            if len({tuple(x.shape) for x in imgs}) == 1:
                ys = runner(torch.stack(imgs))
            else:
                ys = torch.cat([runner(x[None]) for x in imgs])
            for f, y in zip(chunk, ys):
                save(y, os.path.join(a.dst, os.path.relpath(f, a.src)))
        dt = time.time() - t0
        meta = {'scene': s, 'n_images': len(files), 'seconds': dt, 'size': a.size, 'pattern': a.pattern, 'src': a.src,
                'tag': a.tag, 'weights': a.weights, 'host': os.uname().nodename}
        with open(os.path.join(a.dst, s, done_name), 'w') as fh:
            json.dump(meta, fh)
        print(f'DIFIX_SCENE_DONE {s} n={len(files)} {dt:.0f}s ({dt / len(files):.3f} s/img)', flush=True)


if __name__ == '__main__':
    main()
