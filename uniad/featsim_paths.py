"""Portable UniAD data, checkpoint and cache paths (no I/O at import)."""
import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
def env_path(name, default):
    return str(Path(os.environ.get(name, str(default))).expanduser().resolve())

BENCH = env_path('UNIAD_BENCH', REPO_ROOT/'artifacts/uniad')
NUSC_ROOT = env_path('NUSCENES_ROOT', REPO_ROOT/'data/nuscenes')
NUSC_TABLES = env_path('NUSCENES_TABLES', Path(NUSC_ROOT)/'v1.0-trainval')
NUSC_MAP_ROOT = env_path('NUSCENES_MAP_ROOT', NUSC_ROOT)
INFOS_PKL = env_path('UNIAD_INFOS_PKL', Path(NUSC_ROOT)/'nuscenes_infos_temporal_val.pkl')
CKPT_GH = env_path('UNIAD_CKPT_GH', Path(BENCH)/'downloads/uniad_base_e2e.pth')
CKPT_HF = env_path('UNIAD_CKPT_HF', Path(BENCH)/'downloads/hf_uniad2/uniad2_base_e2e.pth')
DEPS = str(CODE_ROOT/'jepa_deps')
NEURAD_ROOT = env_path('NEURAD_ROOT', REPO_ROOT/'third_party/neurad-studio')
