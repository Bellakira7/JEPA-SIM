"""路径与环境引导。所有 offlog 模块先 import 本文件。

* 把 uniad_port 包加进 sys.path（只读使用，不改它）。

  这里放一个空桩模块，避免为了一个绘图依赖去改共享 venv。真正调用绘图函数会在 PolygonPatch 处报错（我们不用）。
"""
import featsim_paths as _release_paths
import os
import sys
import types


CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 数据根（权重、缓存、输出）：固定在 uniad_bench，不随代码快照移动。
UNIAD_BENCH = os.environ.get('UNIAD_BENCH', _release_paths.BENCH)
UNIAD_PORT = os.path.join(CODE_ROOT, 'uniad_port')
if UNIAD_PORT not in sys.path:
    sys.path.insert(0, UNIAD_PORT)

import nuscenes  # Install nuscenes-devkit in the evaluation environment.

if 'descartes' not in sys.modules:
    try:
        import descartes  # noqa: F401
    except ImportError:
        _stub = types.ModuleType('descartes')

        def PolygonPatch(*a, **k):  # noqa: N802
            raise RuntimeError('descartes 是 offlog/common.py 里的空桩，不支持绘图')
        _stub.PolygonPatch = PolygonPatch
        sys.modules['descartes'] = _stub

NUSC_TABLES = _release_paths.NUSC_TABLES
# maps/expansion/*.json 只在上游完整副本里有
NUSC_MAP_ROOT = _release_paths.NUSC_MAP_ROOT
CACHE_DIR = os.path.join(UNIAD_BENCH, 'offlog', 'cache')
CKPT_HF = _release_paths.CKPT_HF
CKPT_GH = _release_paths.CKPT_GH

# runner 输出的检测类别（PORTING_NOTES §7）
CLASS_NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
               'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
VEHICLE_CLASSES = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer')
