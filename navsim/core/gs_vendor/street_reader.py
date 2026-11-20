"""nuScenes input for the official StreetGS model, without a Waymo conversion.

Tracklets and cameras share a metric scene frame. ego_pose=identity means the
official actor transform consumes scene-frame tracklets directly; moving only
the camera in closed loop leaves logged actors stationary in that world.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from .protocol import DRIVE_CAMS, load_manifest


def tracks(manifest, instances, box_scale=1.0):
    n = len(manifest['frames'])
    w = np.asarray(manifest['world_from_global'])
    out = np.full((n, max(1, len(instances)), 8), -1.0, dtype=np.float64)
    meta = {}
    times = np.array([f['timestamp_usec'] for f in manifest['frames']], dtype=np.int64)
    t0 = times[0]
    times = (times - t0) * 1e-6
    for key, obj in instances.items():
        ann = obj['frame_annotations']
        ids = np.asarray(ann['frame_idx'], dtype=int)
        if len(ids) < 2:
            continue
        poses = np.asarray(ann['obj_to_world'])
        positions = poses[:, :3, 3]
        # Official StreetGS reader also keeps only moving objects.
        if np.max(np.linalg.norm(positions - positions[0], axis=1)) < 0.5:
            continue
        track = int(key)
        size = np.asarray(ann['box_size']).max(axis=0)
        size[:2] *= box_scale
        name = obj['class_name']
        label = 0 if name == 'vehicle' else 1
        meta[track] = dict(track_id=track, **{'class': name}, class_label=label,
                           deformable=name == 'pedestrian',
                           length=float(size[0]), width=float(size[1]), height=float(size[2]),
                           start_frame=int(ids.min()), end_frame=int(ids.max()),
                           start_timestamp=float(times[ids.min()] - 0.1),
                           end_timestamp=float(times[ids.max()] + 0.1), instance_token=obj['id'])
        for i, pose in zip(ids, poses):
            p = w @ pose
            out[i, track] = [track, *p[:3, 3], *Quaternion(matrix=p[:3, :3]).elements]
    return out, times, meta


def initialize_points(root, manifest, tracklets, meta, model_path):
    """Colour LiDAR from calibrated RGB, remove actors from the background."""
    from lib.datasets.base_readers import storePly, fetchPly
    root, target = Path(root), Path(model_path) / 'input_ply'
    target.mkdir(parents=True, exist_ok=True)
    cached = target / 'points3D_bkgd.ply'
    if cached.exists():
        return fetchPly(cached), cached
    w = np.asarray(manifest['world_from_global'])
    bg, bg_rgb = [], []
    obj_points = {key: [] for key in meta}
    obj_colors = {key: [] for key in meta}
    for j, frame in enumerate(manifest['frames']):
        lidar = np.fromfile(root / 'lidar' / f'{j:03d}.bin', dtype=np.float32).reshape(-1, 6)[:, :3]
        pose = w @ np.loadtxt(root / 'lidar_pose' / f'{j:03d}.txt')
        xyz = lidar @ pose[:3, :3].T + pose[:3, 3]
        colors = np.full_like(xyz, 0.5)
        visible = np.zeros(len(xyz), dtype=bool)
        for cam_id, cam in enumerate(DRIVE_CAMS):
            item = frame['cameras'][cam]
            c2w = w @ np.asarray(item['c2w_global'])
            local = (xyz - c2w[:3, 3]) @ c2w[:3, :3]
            z = local[:, 2]
            uvw = local @ np.asarray(item['K']).T
            uv = np.rint(uvw[:, :2] / np.maximum(z[:, None], 1e-8)).astype(int)
            im = np.asarray(Image.open(item['path']).convert('RGB'))
            valid = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < im.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < im.shape[0])
            colors[valid] = im[uv[valid, 1], uv[valid, 0]] / 255.0
            visible |= valid
        foreground = np.zeros(len(xyz), dtype=bool)
        for key, obj in meta.items():
            entry = tracklets[j, key]
            if entry[0] < 0:
                continue
            r = Quaternion(entry[4:8]).rotation_matrix
            local = (xyz - entry[1:4]) @ r
            inside = (np.abs(local) <= np.array([obj['length'], obj['width'], obj['height']]) / 2).all(axis=1)
            foreground |= inside
            obj_points[key].append(local[inside & visible])
            obj_colors[key].append(colors[inside & visible])
        keep = visible & ~foreground
        bg.append(xyz[keep])
        bg_rgb.append(colors[keep])
    xyz, rgb = np.concatenate(bg), np.concatenate(bg_rgb)
    if len(xyz) < 100:
        raise ValueError('Insufficient visible background LiDAR; do not initialize a fake scene')
    rng = np.random.default_rng(0)
    ids = rng.choice(len(xyz), min(500_000, len(xyz)), replace=False)
    storePly(str(cached), xyz[ids], rgb[ids])
    for key in meta:
        if obj_points[key] and sum(map(len, obj_points[key])):
            storePly(str(target / f'points3D_obj_{key:03d}.ply'), np.concatenate(obj_points[key]), np.concatenate(obj_colors[key]))
    return fetchPly(cached), cached


def read_nuscenes(path, **kwargs):
    from lib.config import cfg
    from lib.datasets.base_readers import CameraInfo, SceneInfo, getNerfppNorm
    from lib.utils.graphics_utils import focal2fov
    from lib.utils.box_utils import bbox_to_corner3d, get_bound_2d_mask
    root = Path(path)
    manifest, _ = load_manifest(root / 'manifest.json')
    instances = json.loads((root / 'instances/instances_info.json').read_text())
    tracklets, times, obj_meta = tracks(manifest, instances, float(cfg.data.box_scale))
    w = np.asarray(manifest['world_from_global'])
    cameras, cam_times = [], {i: dict(train_timestamps=[], test_timestamps=[]) for i in range(3)}
    for j, frame in enumerate(manifest['frames']):
        for ci, cam in enumerate(DRIVE_CAMS):
            item = frame['cameras'][cam]
            c2w = w @ np.asarray(item['c2w_global'])
            w2c = np.linalg.inv(c2w)
            image = Image.open(root / 'images' / f'{j:03d}_{ci}.jpg').convert('RGB')
            k = np.asarray(item['K']).copy()
            stamp = (item['timestamp_usec'] - manifest['frames'][0]['timestamp_usec']) * 1e-6
            cam_times[ci]['train_timestamps'].append(stamp)
            guidance = {}
            p = root / 'sky_masks' / f'{j:03d}_{ci}.png'
            if p.exists():
                guidance['sky_mask'] = Image.fromarray(np.asarray(Image.open(p)) > 0)
            # Use exactly the retained moving tracks and the official box-mask
            # projection, rather than the broader DriveStudio instance masks.
            bound = np.zeros((image.height, image.width), dtype=bool)
            for key, obj in obj_meta.items():
                entry = tracklets[j, key]
                if entry[0] < 0:
                    continue
                extent = np.array([obj['length'], obj['width'], obj['height']]) * 0.5
                local = bbox_to_corner3d(np.array([-extent, extent]))
                corners = local @ Quaternion(entry[4:8]).rotation_matrix.T + entry[1:4]
                if np.max((corners @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]) <= 0:
                    continue
                bound |= get_bound_2d_mask(corners, k, w2c, image.height, image.width).astype(bool)
            guidance['obj_bound'] = Image.fromarray(bound)
            # Upstream loadCam ignores cfg.resolution and only applies its
            # resolution_scales. Supply half-resolution CameraInfo to retain
            # the official Scene/trainer's scale=1 API and full field of view.
            downscale = max(1, int(cfg.resolution))
            if downscale > 1:
                size = (image.width // downscale, image.height // downscale)
                k[0] *= size[0] / image.width
                k[1] *= size[1] / image.height
                image = image.resize(size, Image.Resampling.BILINEAR)
                guidance = {key: value.resize(size, Image.Resampling.NEAREST) for key, value in guidance.items()}
            meta = dict(frame=j, cam=ci, frame_idx=j, ego_pose=np.eye(4),
                        extrinsic=c2w, timestamp=stamp, is_val=False)
            cameras.append(CameraInfo(uid=j * 3 + ci, R=w2c[:3, :3].T, T=w2c[:3, 3],
                           FovX=focal2fov(k[0, 0], image.width), FovY=focal2fov(k[1, 1], image.height),
                           K=k, image=image, image_path=str(root / 'images' / f'{j:03d}_{ci}.jpg'),
                           image_name=f'{j:03d}_{ci}', width=image.width, height=image.height,
                           metadata=meta, guidance=guidance))
    norm = getNerfppNorm(cameras)
    norm['radius'] = max(float(norm['radius']), 10)
    cfg.data.extent = norm['radius']
    meta = dict(obj_tracklets=tracklets, tracklet_timestamps=times, obj_meta=obj_meta,
                num_images=len(cameras), num_cams=3, num_frames=len(manifest['frames']),
                camera_timestamps=cam_times, scene_center=norm['center'], scene_radius=norm['radius'],
                sphere_center=norm['center'], sphere_radius=norm['radius'])
    pcd, ply = initialize_points(root, manifest, tracklets, obj_meta, cfg.model_path) if cfg.mode == 'train' else (None, None)
    return SceneInfo(point_cloud=pcd, train_cameras=cameras, test_cameras=[], nerf_normalization=norm,
                     ply_path=str(ply) if ply else None, metadata=meta, novel_view_cameras=[])
