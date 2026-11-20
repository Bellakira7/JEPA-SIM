"""Official navtrain feature training followed by a separately configured navtest evaluation."""
import argparse
from contextlib import nullcontext
import copy
import json
import math
import os
from pathlib import Path
import sys
import time


def initialize(config, split):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from scripts.domain_gap.variational_gap_onestep_eval import build_eval_sensor_config
    from eval_open_loop import load_agent
    sub = config[split]
    filt = instantiate(OmegaConf.load(sub['scene_filter_yaml']))
    loader = SceneLoader(Path(sub['openscene_root']) / 'navsim_logs' / sub['data_split'],
                         Path(sub['openscene_root']) / 'sensor_blobs' / sub['data_split'], filt,
                         sensor_config=build_eval_sensor_config(filt.num_history_frames, config['gaps'], True))
    teacher = {**config, **sub, 'arms': ['identity'], 'flowsim_checkpoint': None}
    agent = load_agent(teacher, torch)
    return loader, agent


def prepare(config, out, rank=0, world=1):
    import torch
    from protocol import write_json, sha256_file
    from ltf_features import extract_camera_stages, replay_camera_stages
    from scene_layout import condition_layout, action_between
    loader, agent = initialize(config, 'train')
    from parallel_helpers import assigned_logs
    all_tokens = sorted(loader.tokens_stage_one)
    token_logs = {token: loader.scene_frames_dicts[token][3]['log_name'] for token in all_tokens}
    owned_logs = assigned_logs(token_logs, world)[rank]
    tokens = [token for token in all_tokens if token_logs[token] in owned_logs]
    if not tokens:
        raise ValueError('Official navtrain token selection is empty')
    cache = out / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    shard = out / 'cache_shards' / f'rank_{rank:02d}'
    shard.mkdir(parents=True, exist_ok=True)
    builder = agent.get_feature_builders()[0]
    seen, pairs, logs = set(), [], set()
    sums = squares = counts = None
    identity_done = False
    reused_frames = extracted_frames = reused_conditions = new_conditions = 0
    for index, token in enumerate(tokens, 1):
        raw = loader.scene_frames_dicts[token]
        # Cached tokens need no camera loading or preprocessing. Materialize a
        # scene only when a feature or condition is missing.
        scene = inputs = None
        log = raw[3]['log_name']
        logs.add(log)
        for gap in config['gaps']:
            features = None
            keys = (raw[3-gap]['token'], raw[3]['token'])
            for key, field in zip(keys, ('camera_feature_past', 'camera_feature')):
                if key in seen:
                    continue
                path = cache / (key + '.pt')
                if path.exists():
                    tensors = torch.load(path, map_location='cpu', weights_only=False)
                    reused_frames += 1
                else:
                    if scene is None:
                        scene = loader.get_scene_from_token(token)
                    if inputs is None:
                        inputs = scene.get_agent_input()
                    if features is None:
                        features = builder.compute_features_with_scene_gap(inputs, scene, gap)
                    stages, bev = extract_camera_stages(agent, features[field].unsqueeze(0).to(config['device']))
                    if not identity_done:
                        replay = replay_camera_stages(agent, stages)
                        torch.testing.assert_close(replay, bev, atol=1e-6, rtol=0)
                        write_json(shard / 'feature_cut_identity.json', dict(max_abs=float((replay-bev).abs().max()), shapes=[list(t.shape) for t in stages]))
                        identity_done = True
                    tensors = [t[0].cpu().float() for t in stages]
                    temporary = path.with_suffix('.partial')
                    torch.save(tensors, temporary)
                    temporary.replace(path)
                    extracted_frames += 1
                if sums is None:
                    sums = [torch.zeros(t.shape[0], dtype=torch.float64) for t in tensors]
                    squares = [torch.zeros_like(t) for t in sums]
                    counts = [0] * 4
                for level, tensor in enumerate(tensors):
                    flat = tensor.double().flatten(1)
                    sums[level] += flat.sum(1)
                    squares[level] += flat.square().sum(1)
                    counts[level] += flat.shape[1]
                seen.add(key)
            condition = f'{token}_{gap}.pt'
            path = cache / condition
            if not path.exists():
                if scene is None:
                    scene = loader.get_scene_from_token(token)
                temporary = path.with_suffix('.partial')
                torch.save(dict(layout=condition_layout(scene, raw, gap), action=action_between(raw, gap)), temporary)
                temporary.replace(path)
                new_conditions += 1
            else:
                reused_conditions += 1
            pairs.append(dict(token=token, log=log, gap=gap, source=keys[0], target=keys[1], condition=condition))
        if index == 1 or index % 25 == 0 or index == len(tokens):
            write_json(shard / 'cache_progress.json', dict(rank=rank, world=world, tokens=index, total_tokens=len(tokens),
                frames=len(seen), pairs=len(pairs), reused_frames=reused_frames, extracted_frames=extracted_frames,
                reused_conditions=reused_conditions, new_conditions=new_conditions))
            print('CACHE_PROGRESS', 'rank', rank, index, '/', len(tokens), 'frames', len(seen),
                'reused_frames', reused_frames, 'extracted_frames', extracted_frames,
                'reused_conditions', reused_conditions, 'new_conditions', new_conditions, flush=True)
    # A restart may find every frame already cached: verify the cut again.
    if not identity_done:
        scene = loader.get_scene_from_token(tokens[0])
        features = builder.compute_features_with_scene_gap(scene.get_agent_input(), scene, config['gaps'][0])
        stages, bev = extract_camera_stages(agent, features['camera_feature'].unsqueeze(0).to(config['device']))
        replay = replay_camera_stages(agent, stages)
        torch.testing.assert_close(replay, bev, atol=1e-6, rtol=0)
        write_json(shard / 'feature_cut_identity.json', dict(max_abs=float((replay-bev).abs().max()), shapes=[list(t.shape) for t in stages]))
    if not (shard / 'feature_cut_identity.json').exists():
        raise ValueError('No verified feature cut identity record')
    torch.save(dict(sums=sums, squares=squares, counts=counts), shard / 'moments.pt')
    write_json(shard / 'pairs.json', pairs)
    write_json(shard / 'complete.json', dict(complete=True, rank=rank, world=world, config=config,
        tokens=tokens, logs=sorted(logs), frames=sorted(seen), ltf_sha256=sha256_file(config['ltf_checkpoint'])))
    print('CACHE_SHARD_COMPLETE', rank, 'tokens', len(tokens), flush=True)
    if world == 1:
        from parallel_helpers import merge_cache
        merge_cache(config, out, world)


def train(config, out, rank=0, world=1):
    import torch
    import torch.distributed as dist
    from protocol import write_json, sha256_file
    from ltf_features import build_predictor, predictor_kwargs
    torch.manual_seed(config['seed'])
    model = build_predictor().to(config['device']).train()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    network = torch.nn.parallel.DistributedDataParallel(model,
        device_ids=[local_rank] if config['device'].startswith('cuda') else None) if world > 1 else model
    cache_root = Path(config.get('cache_root', out))
    checkpoint_root = Path(config.get('checkpoint_root', out))
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    norm = torch.load(cache_root / 'normalization.pt', map_location='cpu', weights_only=False)
    means = [m.to(config['device']).view(1, 1, 1, -1, 1, 1) for m in norm['mean']]
    stds = [s.to(config['device']).view(1, 1, 1, -1, 1, 1) for s in norm['std']]
    pairs = json.loads((cache_root / 'pairs.json').read_text())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=.04)
    sample_rng = torch.Generator().manual_seed(config['seed'])
    per_gpu = config['batch_size']
    accumulation = config.get('gradient_accumulation_steps', 1)
    if not isinstance(accumulation, int) or accumulation < 1:
        raise ValueError('Positive integer gradient accumulation required')
    global_batch = per_gpu * world * accumulation
    last = checkpoint_root / 'last.pt'
    start = 0
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['config'] != config:
            raise ValueError('Resume configuration changed')
        if saved['world_size'] != world:
            raise ValueError('Resume world size changed')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        sample_rng.set_state(saved['sample_rng'])
        torch.set_rng_state(saved['rank_rng'][rank]['cpu'])
        if config['device'].startswith('cuda'):
            torch.cuda.set_rng_state(saved['rank_rng'][rank]['cuda'])
        start = saved['step']
    def load(key):
        filename = key if key.endswith('.pt') else key + '.pt'
        return torch.load(cache_root / 'cache' / filename, map_location='cpu', weights_only=False)
    history = json.loads((out / 'training_progress.json').read_text()) if (out / 'training_progress.json').exists() else []
    started = time.monotonic()
    print('TRAIN_START', json.dumps(dict(rank=rank, world=world, batch_per_gpu=per_gpu,
        global_batch=global_batch, gradient_accumulation_steps=accumulation,
        cache_root=str(cache_root), checkpoint_root=str(checkpoint_root),
        steps=config['steps'], resumed_step=start)), flush=True)
    for step in range(start+1, config['steps']+1):
        # All ranks draw the same global batch, then consume disjoint slots.
        indices = torch.randint(len(pairs), (global_batch,), generator=sample_rng).tolist()
        factor = min(1., step/config['warmup']) * (.01 + .99*.5*(1+math.cos(math.pi*step/config['steps'])))
        for group in optimizer.param_groups:
            group['lr'] = config['lr']*factor
        optimizer.zero_grad(set_to_none=True)
        metrics = None
        for microstep in range(accumulation):
            offset = (microstep * world + rank) * per_gpu
            samples = [pairs[i] for i in indices[offset:offset+per_gpu]]
            sources = [load(p['source']) for p in samples]
            targets = [load(p['target']) for p in samples]
            conditions = [load(p['condition']) for p in samples]
            source = [torch.stack([v[l] for v in sources]).to(config['device'])[:, None, None] for l in range(4)]
            target = [torch.stack([v[l] for v in targets]).to(config['device'])[:, None, None] for l in range(4)]
            source = [(f-m)/s for f, m, s in zip(source, means, stds)]
            target = [(f-m)/s for f, m, s in zip(target, means, stds)]
            action = torch.stack([v['action'] for v in conditions]).to(config['device'])[:, None]
            layout = torch.stack([v['layout'] for v in conditions]).to(config['device'])[:, None, None]
            sync = network.no_sync() if world > 1 and microstep+1 < accumulation else nullcontext()
            with sync:
                prediction = network(source, action, layout)
                losses = [(p-t).abs().mean() for p, t in zip(prediction, target)]
                loss = torch.stack(losses).mean()
                if not torch.isfinite(loss):
                    raise ValueError(f'Nonfinite loss at step {step}, microstep {microstep}')
                (loss / accumulation).backward()
            current = torch.stack([v.detach() for v in losses] +
                [(s-t).abs().mean().detach() for s,t in zip(source,target)]) / accumulation
            metrics = current if metrics is None else metrics + current
            del prediction, losses, loss, sources, targets, conditions, source, target, action, layout
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == start+1 or step % 25 == 0 or step == config['steps']:
            if world > 1:
                dist.all_reduce(metrics)
                metrics /= world
            if rank == 0:
                row = dict(step=step, loss=float(metrics[:4].mean()), per_level=metrics[:4].tolist(),
                    copy_last_per_level=metrics[4:].tolist(), grad_norm=float(gradient),
                    elapsed_s=time.monotonic()-started, world_size=world, global_batch=global_batch,
                    gradient_accumulation_steps=accumulation)
                history.append(row)
                write_json(out / 'training_progress.json', history)
                print('TRAIN_PROGRESS', json.dumps(row), flush=True)
        if step % config['ckpt_every'] == 0 or step == config['steps']:
            local_rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state() if config['device'].startswith('cuda') else None)
            rank_rng = [None] * world
            if world > 1:
                dist.all_gather_object(rank_rng, local_rng)
            else:
                rank_rng[0] = local_rng
            if rank == 0:
                temporary = checkpoint_root / 'last.partial'
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    sample_rng=sample_rng.get_state(), rank_rng=rank_rng, world_size=world,
                    step=step, config=config), temporary)
                temporary.replace(last)
            if world > 1:
                dist.barrier()
    if rank != 0:
        return
    checkpoint = dict(model=model.cpu().state_dict(), mean=norm['mean'], std=norm['std'], predictor_kwargs=predictor_kwargs(), feature_cut='ltf/pre_fusion_camera_stages_v1', config=config, ltf_sha256=sha256_file(config['ltf_checkpoint']), step=config['steps'])
    temporary = checkpoint_root / 'featsim_ltf.partial'
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_root / 'featsim_ltf.pt')
    evaluation = {**config, **config['test'], 'arms': ['identity', 'copy_last_bev', 'featsim_ltf'], 'featsim_checkpoint': str(checkpoint_root / 'featsim_ltf.pt'), 'tokens_file': None, 'limit_tokens': None, 'render_roots': {}, 'dataset': 'OpenScene navtest / camera-complete 131-log subset / frozen LTF seed0 / real-anchor OL'}
    if config.get('flowsim_checkpoint'):
        evaluation['arms'].append('flowsim_vh')
    write_json(out / 'eval_config.json', evaluation)
    print('TRAIN_COMPLETE_NEXT_EVAL', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['prepare', 'train'])
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--rank', type=int, default=0, help='Independent preparation shard')
    ap.add_argument('--world-size', type=int, default=1, help='Independent preparation shard count')
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ['NAVSIM_DEVKIT_ROOT'] = config['flowsim_root']
    os.environ['NUPLAN_MAPS_ROOT'] = str(Path(config['train']['openscene_root']) / 'maps')
    os.environ['NUPLAN_MAP_VERSION'] = 'nuplan-maps-v1.0'
    sys.path.insert(0, config['flowsim_root'])
    print('CONFIG', json.dumps(config), flush=True)
    if args.phase == 'prepare':
        if not 0 <= args.rank < args.world_size:
            raise ValueError('Invalid preparation rank')
        prepare(config, args.out, args.rank, args.world_size)
    else:
        if config.get('cache_root'):
            from parallel_helpers import verify_cache
            print('TRAIN_CACHE_VERIFIED', json.dumps(verify_cache(config, config['cache_root'])), flush=True)
        import torch
        import torch.distributed as dist
        from datetime import timedelta
        world = int(os.environ.get('WORLD_SIZE', 1))
        rank = int(os.environ.get('RANK', 0))
        if world != config.get('world_size', 1):
            raise ValueError('Launched world size differs from frozen configuration')
        if config['device'].startswith('cuda'):
            torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
        if world > 1:
            dist.init_process_group('nccl' if config['device'].startswith('cuda') else 'gloo', timeout=timedelta(minutes=30))
        try:
            train(config, args.out, rank, world)
            if world > 1:
                dist.barrier()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == '__main__':
    main()
