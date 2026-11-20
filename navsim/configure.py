"""Configure the paper's final-fused-BEV TF model and freeze official split selections."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--flowsim-root', required=True, type=Path, help='FlowSim/flowsim_navsim checkout')
    ap.add_argument('--dataset-root', required=True, type=Path)
    ap.add_argument('--train-filter', required=True, type=Path)
    ap.add_argument('--test-filter', required=True, type=Path)
    ap.add_argument('--ltf-checkpoint', required=True, type=Path)
    ap.add_argument('--flowsim-checkpoint', required=True, type=Path)
    ap.add_argument('--data-split', default='trainval')
    ap.add_argument('--gpus', type=int, default=8, help='Paper recipe: eight GPUs, one clip per GPU')
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    if args.gpus < 1:
        raise ValueError('Positive GPU count required')
    root, upstream, out = args.dataset_root.resolve(), args.flowsim_root.resolve(), args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty configuration/output directory')
    sys.path.insert(0, str(upstream))
    os.environ['NAVSIM_DEVKIT_ROOT'] = str(upstream)
    os.environ['NUPLAN_MAPS_ROOT'] = str(root/'maps')
    os.environ['NUPLAN_MAP_VERSION'] = 'nuplan-maps-v1.0'
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig
    manifest, specs = {}, {}
    for split, path in [('train', args.train_filter), ('test', args.test_filter)]:
        filt = instantiate(OmegaConf.load(path))
        if filt.num_history_frames != 4:
            raise ValueError('Official filters must use four history frames for cache extraction')
        loader = SceneLoader(root/'navsim_logs'/args.data_split, root/'sensor_blobs'/args.data_split,
                             filt, sensor_config=SensorConfig.build_no_sensors())
        tokens = sorted(loader.tokens_stage_one)
        if not tokens:
            raise ValueError(f'Empty {split} split')
        logs = sorted({loader.scene_frames_dicts[token][3]['log_name'] for token in tokens})
        manifest[split] = dict(tokens=tokens, log_names=logs, num_history_frames=filt.num_history_frames,
                               num_future_frames=filt.num_future_frames)
        specs[split] = dict(openscene_root=str(root), data_split=args.data_split,
                            scene_filter_yaml=str(path.resolve()))
    if set(manifest['train']['log_names']) & set(manifest['test']['log_names']):
        raise ValueError('Train/test logs overlap')
    out.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        (out/name).write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')
    save('filters.json', manifest)
    config = dict(protocol='OpenScene-LTF-OL-v1', device='cuda', gaps=[1, 2, 3], seed=0,
                  steps=12000, lr=.0003, batch_size=1, world_size=args.gpus, warmup=500,
                  ckpt_every=1000, identity_tolerance=1e-6, ltf_checkpoint=str(args.ltf_checkpoint.resolve()),
                  flowsim_checkpoint=str(args.flowsim_checkpoint.resolve()), flowsim_root=str(upstream),
                  **specs, filter_manifest=str(out/'filters.json'), gradient_accumulation_steps=1,
                  preparation_workers=args.gpus, evaluation_workers=args.gpus,
                  cache_root=str(out/'cache_run'), checkpoint_root=str(out/'single_step'))
    save('cache.json', config)
    # Camera-stage caches are an intermediate for frozen fusion replay, not a trained camera predictor.
    camera_sequence = dict(protocol='OpenScene-LTF-AR-v2', source_config=str(out/'cache.json'),
                           ltf_checkpoint=config['ltf_checkpoint'], device='cuda', seed=0, history_frames=4,
                           step_seconds=.5, warp_levels=[], condition_policy='all_adjacent_cached_frames',
                           world_size=args.gpus, batch_size=1, gradient_accumulation_steps=1, rollout_horizon=8,
                           tf_lr=.0003, ro_lr=.0001, ckpt_every=2000, training_stages=['tf'])
    save('camera_sequence.json', camera_sequence)
    sequence = {**camera_sequence, 'feature_cut': 'ltf/final_fused_bev_v1',
                'reference_sequences': str(out/'camera_sequence/sequences.json'),
                'conversion_batch_size': 4}
    save('sequence.json', sequence)
    save('evaluation.json', {**sequence, 'featsim_checkpoint': str(out/'sequence/tf/featsim_ltf.pt'),
                             'expected_feature_cut': 'ltf/final_fused_bev_v1',
                             'expected_checkpoint_stage': 'tf', 'expected_checkpoint_step': 45495,
                             'expected_warp_levels': []})
    print('Created configs:', out)


if __name__ == '__main__':
    main()
