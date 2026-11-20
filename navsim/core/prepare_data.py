"""Build an OpenScene sensor view over existing nuPlan blobs; copy no images."""
import argparse
import json
import pickle
from pathlib import Path

from protocol import OPENSCENE_REVISION, sha256_file, write_json

CAMERAS = ('CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_L1', 'CAM_L2', 'CAM_R1', 'CAM_R2', 'CAM_B0')


def link_directory(source, target):
    source = source.resolve(strict=True)
    if target.exists() or target.is_symlink():
        if target.resolve() != source:
            raise FileExistsError(f'Existing target points elsewhere: {target}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=source.is_dir())


def audit_logs(metadata_dir, camera_root, lidar_root):
    result = []
    candidates = sorted(p for p in metadata_dir.glob('*.pkl') if (camera_root/p.stem).is_dir())
    if not candidates:
        raise ValueError('No OpenScene metadata logs overlap the existing camera blobs')
    for path in candidates:
        with path.open('rb') as f:
            frames = pickle.load(f)
        if not isinstance(frames, list) or not frames:
            raise ValueError(f'Expected a nonempty OpenScene frame list: {path}')
        missing = []
        valid_front_frames = 0
        for frame in frames:
            required = ['token', 'cams', 'anns', 'ego2global_translation', 'ego2global_rotation', 'ego_dynamic_state', 'driving_command', 'map_location']
            absent = [k for k in required if k not in frame]
            if absent:
                raise ValueError(f'{path.name}: missing metadata fields {absent}')
            front_ok = True
            for cam in CAMERAS[:3]:
                camera = frame['cams'][cam]
                for key in ('data_path', 'cam_intrinsic', 'sensor2lidar_rotation', 'sensor2lidar_translation'):
                    if key not in camera:
                        raise ValueError(f'Missing calibration: {path.name} {cam} {key}')
                # Preserve the official relative blob path; do not guess basenames.
                relative = Path(camera['data_path'])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError(f'Invalid official relative image path: {relative}')
                if not (camera_root/relative).is_file():
                    front_ok = False
                    if len(missing) < 5:
                        missing.append(str(relative))
            valid_front_frames += int(front_ok)
        result.append(dict(log_name=path.stem, metadata_file=str(path),
                           metadata_sha256=sha256_file(path), num_frames=len(frames),
                           front_camera_complete_frames=valid_front_frames,
                           missing_examples=missing,
                           lidar_directory_present=(lidar_root/path.stem/'MergedPointCloud').is_dir()))
        print(f"audit: {path.stem} front={valid_front_frames}/{len(frames)}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metadata-dir', required=True, type=Path)
    ap.add_argument('--camera-root', required=True, type=Path)
    ap.add_argument('--lidar-root', required=True, type=Path)
    ap.add_argument('--maps-root', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    print('config: '+json.dumps({k:str(v) for k,v in vars(args).items()}), flush=True)
    logs = audit_logs(args.metadata_dir, args.camera_root, args.lidar_root)
    if any(x['front_camera_complete_frames'] != x['num_frames'] for x in logs):
        write_json(args.out/'data_audit.json', dict(status='incomplete', logs=logs))
        raise ValueError('Incomplete camera coverage; inspect data_audit.json before selecting evaluation tokens')
    log_view = args.out/'navsim_logs'/'mini'
    blob_view = args.out/'sensor_blobs'/'mini'
    for log in logs:
        name = log['log_name']
        source = Path(log['metadata_file']).resolve()
        target = log_view/(name+'.pkl')
        link_directory(source, target)
        for cam in CAMERAS:
            source_cam = args.camera_root/name/cam
            if source_cam.is_dir():
                link_directory(source_cam, blob_view/name/cam)
        source_lidar = args.lidar_root/name/'MergedPointCloud'
        if source_lidar.is_dir():
            link_directory(source_lidar, blob_view/name/'MergedPointCloud')
    link_directory(args.maps_root, args.out/'maps')
    write_json(args.out/'data_audit.json', dict(status='ready_for_camera_smoke',
               dataset='OpenScene-v1.1 mini / available nuPlan blob shard 0',
               openscene_revision=OPENSCENE_REVISION, logs=logs,
               note='This subset is not the full NAVSIM navtest benchmark. LTF uses no LiDAR.'))
    print(f'done: {len(logs)} logs -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
