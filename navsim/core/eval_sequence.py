"""AR-v2 plan/worker/merge orchestration, adapted from experiment-protocol eval_template.py."""
import argparse
import json
import os
from pathlib import Path
import sys

# Capture torchrun assignment before the independent evaluator clears DDP variables.
RANK = int(os.environ.get('RANK', 0))
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
    os.environ.pop(key, None)
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from protocol import AR_PROTOCOL, sha256_file, write_json


def evaluation_config(config, out):
    source = json.loads(Path(config['source_config']).read_text())
    evaluation = {k: source[k] for k in ('flowsim_root', 'filter_manifest', 'gaps', 'identity_tolerance',
                                        'ltf_checkpoint', 'flowsim_checkpoint', 'seed', 'device')}
    expected = {key: config[key] for key in ('expected_checkpoint_stage', 'expected_checkpoint_step',
                                           'expected_checkpoint_sha256', 'expected_warp_levels','expected_feature_cut') if key in config}
    return {**evaluation, **source['test'], **expected, 'protocol': AR_PROTOCOL, 'history_frames': 4, 'step_seconds': .5,
            'arms': ['identity', 'copy_last_bev', 'featsim_ltf', 'flowsim_vh'],
            'featsim_checkpoint': config.get('featsim_checkpoint', str(out/'ro/featsim_ltf.pt')), 'tokens_file': None, 'limit_tokens': None,
            'render_roots': {}, 'dataset': config.get('dataset','OpenScene navtest / camera-complete legal-four-history subset / frozen LTF seed0 / JEPA AR vs FlowSim VH'),
            'evaluation_workers': config['world_size']}


def plan(config, out):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from scripts.domain_gap.variational_gap_onestep_eval import build_eval_sensor_config
    from sequence_inference import select_tokens
    filt = instantiate(OmegaConf.load(config['scene_filter_yaml']))
    filt.num_history_frames = 4+max(config['gaps'])
    root = Path(config['openscene_root'])
    sensors = build_eval_sensor_config(filt.num_history_frames, list(range(1,filt.num_history_frames)), True)
    loader = SceneLoader(root/'navsim_logs'/config['data_split'], root/'sensor_blobs'/config['data_split'], filt, sensor_config=sensors)
    tokens, audit = select_tokens(loader, config)
    world = config['evaluation_workers']
    if len(tokens) < world:
        raise ValueError('Insufficient tokens for evaluation workers')
    write_json(out/'requested_tokens.json', tokens)
    write_json(out/'sequence_token_audit.json', audit)
    write_json(out/'config.json', config)
    for rank in range(world):
        shard = out/'shards'/f'rank_{rank:02d}'
        write_json(shard/'requested_tokens.json', tokens[rank::world])
        write_json(shard/'config.json', {**config, 'tokens_file': str(shard/'requested_tokens.json')})
    print('EVAL_PLAN_COMPLETE', json.dumps({k: v for k,v in audit.items() if not isinstance(v,list)}), flush=True)


def worker(out):
    import torch
    from eval_open_loop import evaluate, preflight
    shard = out/'shards'/f'rank_{RANK:02d}'
    config = json.loads((shard/'config.json').read_text())
    audit = preflight(config)
    write_json(shard/'preflight.json', audit)
    if audit['status'] != 'ready':
        raise ValueError('Sequence evaluation prerequisites failed; see preflight.json')
    torch.cuda.set_device(LOCAL_RANK)
    torch.manual_seed(config['seed'])
    if (shard/'stats.json').exists():
        existing = json.loads((shard/'stats.json').read_text())
        if (existing['status'] != 'complete' or existing['config'] != config or
            existing['artifacts']['featsim_checkpoint_sha256'] != sha256_file(config['featsim_checkpoint'])):
            raise ValueError('Existing shard differs from completed checkpoint/config')
        return
    evaluate(config, shard, torch)


def merge(out):
    from eval_open_loop import finalize_evaluation
    config = json.loads((out/'config.json').read_text())
    requested = json.loads((out/'requested_tokens.json').read_text())
    records, checks, seen = [], [], set()
    artifacts = None
    for rank in range(config['evaluation_workers']):
        shard = out/'shards'/f'rank_{rank:02d}'
        stats = json.loads((shard/'stats.json').read_text())
        tokens = json.loads((shard/'tokens.json').read_text())
        part = json.loads((shard/'records.json').read_text())
        identity = json.loads((shard/'identity_checks.json').read_text())['per_token']
        expected_config = {**config, 'tokens_file': str(shard/'requested_tokens.json')}
        if stats['status'] != 'complete' or stats['config'] != expected_config or tokens != requested[rank::config['evaluation_workers']]:
            raise ValueError('Shard config or coverage mismatch')
        if seen & set(tokens) or {r['token'] for r in part} != set(tokens) or {r['token'] for r in identity} != set(tokens):
            raise ValueError('Duplicate or incomplete shard')
        if artifacts is not None and artifacts != stats['artifacts']:
            raise ValueError('Evaluated checkpoint differs across workers')
        artifacts = stats['artifacts']
        seen.update(tokens)
        records.extend(part)
        checks.extend(identity)
    if seen != set(requested):
        raise ValueError('Final evaluation incomplete')
    write_json(out/'tokens.json', requested)
    finalize_evaluation(config, out, records, checks, artifacts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=('plan', 'worker', 'merge'))
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    config = evaluation_config(json.loads(args.config.read_text()), args.out)
    print('config: '+json.dumps(config), flush=True)
    sys.path.insert(0, config['flowsim_root'])
    os.environ['NAVSIM_DEVKIT_ROOT'] = config['flowsim_root']
    os.environ['NUPLAN_MAPS_ROOT'] = str(Path(config['openscene_root'])/'maps')
    os.environ['NUPLAN_MAP_VERSION'] = 'nuplan-maps-v1.0'
    eval_out = args.out/'open_loop'
    if args.phase == 'plan':
        plan(config, eval_out)
    elif args.phase == 'worker':
        worker(eval_out)
    else:
        merge(eval_out)


if __name__ == '__main__':
    main()
