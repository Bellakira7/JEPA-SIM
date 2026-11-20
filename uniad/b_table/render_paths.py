"""3DGS 臂：渲染图的目录约定（生成请求 `gen_render_requests.py`、评测读图 `b_eval.py` / `a_eval.py` 共用，两边不各写一份）。

B 表（仿真位姿，按请求渲染）::
    <render_root>/<scene_name>/a<锚点:02d>_lat<δ:+.1f>/k<k>/<CAM>.jpg          横向 δ（m）；11-06 的 3DGS 作业就是这个格式
    <render_root>/<scene_name>/a<锚点:02d>_yaw<ψ:+.1f>/k<k>/<CAM>.jpg          航向偏置 ψ（deg）
A 表（日志位姿 δ=0 的训练视角，render_eval.py 的 shift_+0.0 输出）::
    <log_root>/<scene_name>/shift_+0.0/<CAM>/<原文件名 stem>.jpg
  log_root 下每个场景一个目录（可以是指向重建作业 `out/<scene>/render` 的软链）。`log_root == NUSC_REAL` 时直接返回 nuScenes 原图路径
  （自检用：与真图走完全同一条加载路径，结果必须与 identity / 参照逐位相同）。
多个 3DGS 类臂（如 Difix 后处理）：臂名 `3dgs`（--render-root）/ `3dgs_<tag>`（--render-roots tag=path），目录格式相同。
"""
import os

CAMS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')   # = uniad_port.data.CAM_ORDER
NUSC_REAL = '@nusc'


def item_dir(anchor: int, val: float, k: int, fam: str = 'lat') -> str:
    assert fam in ('lat', 'yaw'), f'3DGS 渲染只有横向 / 航向两族：{fam}'
    return f'a{anchor:02d}_{fam}{val:+.1f}/k{k}'


def image_paths(render_root: str, scene_name: str, anchor: int, val: float, k: int, fam: str = 'lat'):
    d = os.path.join(render_root, scene_name, item_dir(anchor, val, k, fam))
    return [os.path.join(d, f'{c}.jpg') for c in CAMS]


def log_image_paths(log_root: str, scene_name: str, orig_paths):
    """A 表：日志位姿渲染图。orig_paths = 该帧 6 路 nuScenes 原图路径（img_metas['filename']，即真图的加载顺序），按相机目录名 + 文件名 stem 映射。"""
    if log_root == NUSC_REAL:
        return list(orig_paths)
    out = []
    for p in orig_paths:
        cam = os.path.basename(os.path.dirname(p))
        assert cam in CAMS, p
        out.append(os.path.join(log_root, scene_name, 'shift_+0.0', cam, os.path.splitext(os.path.basename(p))[0] + '.jpg'))
    return out


def parse_render_roots(render_root: str, specs):
    """--render-root 与 --render-roots tag=path → {臂名: 根目录}（'3dgs' / '3dgs_<tag>'）。"""
    roots = {}
    if render_root:
        roots['3dgs'] = render_root
    for s in specs or []:
        tag, _, path = s.partition('=')
        assert tag and path and tag.replace('_', '').isalnum(), f'--render-roots 格式应为 tag=path：{s!r}'
        assert f'3dgs_{tag}' not in roots, f'重复的 tag：{tag}'
        roots[f'3dgs_{tag}'] = path
    return roots


def is_3dgs_arm(name: str) -> bool:
    return name == '3dgs' or name.startswith('3dgs_')
