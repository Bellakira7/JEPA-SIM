"""Shared, explicit camera and pose contract; never substitute real RGB."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CAMS = ('CAM_F0','CAM_L0','CAM_R0')
DRIVE_CAMS = CAMS
METHODS = ('omnire_nosmpl', 'street_gs')
PROTOCOL = 'OpenScene-LTF-OL-v1 / reconstruction-front3-pilot'


def rigid(matrix):
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError('Expected a finite 4x4 rigid transform')
    if not (np.allclose(m[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            and np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            and np.isclose(np.linalg.det(m[:3, :3]), 1, atol=1e-5, rtol=0)):
        raise ValueError('Scaled/reflected/non-rigid coordinate transforms are unsupported')
    return m


def moved_camera(c2w, D, world_from_global):
    """D = E_sim @ inv(E_log), in global metres; keep camera-time motion."""
    c = rigid(c2w)
    if D is None:
        return c.copy()
    w = rigid(world_from_global)
    return w @ rigid(D) @ np.linalg.inv(w) @ c


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get('version') != 1 or manifest.get('hz') != 2:
        raise ValueError('Expected manifest v1 with annotated 2Hz keyframes')
    by_name = {}
    for frame_idx, frame in enumerate(manifest['frames']):
        if set(frame['cameras']) != set(CAMS):
            raise ValueError(f'Incomplete three-camera frame {frame_idx}')
        for cam, item in frame['cameras'].items():
            name = Path(item['path']).name
            if name in by_name:
                raise ValueError(f'Duplicate source image: {name}')
            by_name[name] = (frame_idx, cam, item)
    return manifest, by_name


def validate_files(files, by_name):
    if set(files) != set(CAMS):
        raise ValueError('A render request must contain all three OpenScene cameras')
    frames = set()
    for cam in CAMS:
        frame_idx, channel, _ = by_name[files[cam]]
        if channel != cam:
            raise ValueError(f'{cam} requested an image from {channel}')
        frames.add(frame_idx)
    if len(frames) != 1:
        raise ValueError('A request mixes cameras from different keyframes')
    return frames.pop()


def save_rgb(rgb, path, size):
    """Full native field of view; resize model RGB, with no GT pixel fill."""
    from PIL import Image
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise ValueError(f'Invalid RGB shape/values: {rgb.shape}')
    im = Image.fromarray(np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8))
    if im.size != tuple(size):
        im = im.resize(tuple(size), Image.Resampling.BILINEAR)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    im.save(tmp, format='JPEG', quality=95)
    tmp.replace(path)
