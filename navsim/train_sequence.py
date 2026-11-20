"""Four-frame TF training with optional sliding rollout fine-tuning; epoch-based data exposure."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE/'core'))
from protocol import AR_PROTOCOL, sha256_file, write_json
from sequence_data import EpochBatchSampler, validate_manifest, training_stages


def forecast(predictor, history, actions, layout, horizon):
    """No future image/features enter prediction; only the four real anchor frames are accepted."""
    from feat_vendor.pyramid_predictor import sliding_rollout
    model=getattr(predictor,'module',predictor)
    if len(history) != model.L or any(x.shape[1] != 4 for x in history):
        raise ValueError('Feature level count must match predictor; four real history frames required')
    return sliding_rollout(predictor, history, actions, layout, 4, horizon)


def losses(predictor, features, actions, layout, stage, horizon=1):
    import torch
    level_loss = lambda ps, ts: torch.stack([(p.float()-t.float()).abs().mean() for p, t in zip(ps, ts)]).mean()
    history = [x[:, :4] for x in features]
    dense, ar = forecast(predictor, history, actions, layout, horizon if stage == 'ro' else 1)
    terms = [level_loss(dense, [x[:, 1:5] for x in features])]
    if stage == 'ro':
        for k in range(1, horizon):
            terms.append(level_loss([x[:, k-1:k] for x in ar], [x[:, 4+k:5+k] for x in features]))
    loss = torch.stack(terms).mean()
    return loss, [float(t.detach()) for t in terms]


def load_batch(clips, cache, means, stds, device, condition_cache=None, legacy_condition_cache=None):
    import torch
    # Read each frame once per batch; overlapping four-frame windows are common.
    names = {name for clip in clips for name in clip['frames']+clip['conditions']}
    def path_for(name):
        path = cache/(name if name.endswith('.pt') else name+'.pt')
        if name.endswith('.pt') and not path.exists():
            for root in (condition_cache,legacy_condition_cache):
                if root and (root/name).is_file():return root/name
        return path
    blobs = {name: torch.load(path_for(name), map_location='cpu', weights_only=False) for name in names}
    features = [torch.stack([torch.stack([blobs[t][l] for t in c['frames']]) for c in clips]).to(device)[:, :, None]
                for l in range(len(means))]
    features = [(x-m)/s for x, m, s in zip(features, means, stds)]
    actions = torch.stack([torch.stack([blobs[t]['action'] for t in c['conditions']]) for c in clips]).to(device)
    layout = torch.stack([torch.stack([blobs[t]['layout'] for t in c['conditions']]) for c in clips]).to(device)[:, :, None]
    if not torch.allclose(actions[..., 3], torch.full_like(actions[..., 3], .5)):
        raise ValueError('Sequence training only accepts adjacent 0.5 s conditions')
    return features, actions, layout


def horizon_at(step, total, final=8):
    return 2 if step < total/6 else 4 if step < total/3 else final


def initialize_predictor(config):
    """Paired arms share every common initial tensor and the same post-init RNG state."""
    import torch
    from ltf_features import build_predictor, predictor_kwargs
    if config.get('feature_cut')=='ltf/final_fused_bev_v1':
        from bev_features import build_predictor,predictor_kwargs
    torch.manual_seed(config['seed'])
    kwargs = predictor_kwargs()
    kwargs.update(warp_levels=(), use_activation_checkpointing=True)
    model = build_predictor(kwargs)
    if config['warp_levels']:
        kwargs = {**kwargs, 'warp_levels': tuple(config['warp_levels'])}
        with torch.random.fork_rng(devices=[]):
            warped = build_predictor(kwargs)
        loaded = warped.load_state_dict(model.state_dict(), strict=False)
        expected = {f'flow_proj.{level}.{field}' for level in config['warp_levels'] for field in ('weight', 'bias')}
        if set(loaded.missing_keys) != expected or loaded.unexpected_keys:
            raise ValueError('Warp arm must differ only by its zero-initialized flow projections')
        model = warped
    return model, kwargs


def train_stage(config, manifest, out, stage, rank, world):
    import torch
    import torch.distributed as dist
    device = config['device']
    model, kwargs = initialize_predictor(config)
    model = model.to(device)
    stage_out = Path(out)/stage
    stage_out.mkdir(parents=True, exist_ok=True)
    recipe = manifest['stages'][stage]
    clips, total = recipe['clips'], recipe['steps']
    local_batch, accumulation = config['batch_size'], config.get('gradient_accumulation_steps', 1)
    global_batch = world*local_batch*accumulation
    sampler = EpochBatchSampler(len(clips), global_batch, config['seed'])
    lr = config['tf_lr'] if stage == 'tf' else config['ro_lr']
    # Preserve the existing OpenScene optimizer while changing history/exposure.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.04)
    complete_path = stage_out/'complete.json'
    if complete_path.exists():
        completed = json.loads(complete_path.read_text())
        if (completed['step'] != total or completed['manifest_sha256'] != sha256_file(Path(out)/'sequences.json')
                or completed['checkpoint_sha256'] != sha256_file(stage_out/'featsim_ltf.pt')):
            raise ValueError('Completed stage artifacts changed')
        if world > 1:
            dist.barrier()
        return
    last = stage_out/'last.pt'
    start = 0
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['config'] != config or saved['stage'] != stage or saved['manifest_sha256'] != sha256_file(Path(out)/'sequences.json'):
            raise ValueError('Resume recipe/manifest changed')
        model.load_state_dict(saved['model'], strict=True)
        opt.load_state_dict(saved['optimizer'])
        sampler.load_state_dict(saved['sampler'])
        torch.set_rng_state(saved['rank_rng'][rank]['cpu'])
        if device.startswith('cuda'):
            torch.cuda.set_rng_state(saved['rank_rng'][rank]['cuda'])
        start = saved['step']
    elif stage == 'ro':
        saved = torch.load(Path(out)/'tf/featsim_ltf.pt', map_location='cpu', weights_only=False)
        if saved['step'] != manifest['stages']['tf']['steps'] or saved['prediction_mode'] != 'sliding_ar':
            raise ValueError('RO requires the completed TF stage of this run')
        model.load_state_dict(saved['model'], strict=True)
    model.train()
    network = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ['LOCAL_RANK'])]
        if device.startswith('cuda') else None) if world > 1 else model
    norm = torch.load(Path(manifest['source_cache_root'])/'normalization.pt', map_location='cpu', weights_only=False)
    means = [v.to(device).view(1, 1, 1, -1, 1, 1) for v in norm['mean']]
    stds = [v.to(device).view(1, 1, 1, -1, 1, 1) for v in norm['std']]
    warmup = max(1, round(total*(500/12000 if stage == 'tf' else 300/6000)))
    started = time.monotonic()
    manifest_hash = sha256_file(Path(out)/'sequences.json')
    if rank == 0:
        print('TRAIN_START', json.dumps(dict(stage=stage, history=4, step_seconds=.5, warp_levels=config['warp_levels'],
              world=world, batch_per_gpu_clips=local_batch, global_batch_clips=global_batch,
              clips=len(clips), steps=total, resumed_step=start)), flush=True)
    for step in range(start, total):
        indices = sampler.next()
        rate = min(1., (step+1)/warmup)*(.01+.99*.5*(1+math.cos(math.pi*step/total)))
        for group in opt.param_groups:
            group['lr'] = lr*rate
        opt.zero_grad(set_to_none=True)
        average = torch.zeros((), device=device)
        horizon = horizon_at(step, total, config['rollout_horizon']) if stage == 'ro' else 1
        for micro in range(accumulation):
            offset = (micro*world+rank)*local_batch
            batch = [clips[i] for i in indices[offset:offset+local_batch]]
            features, actions, layout = load_batch(batch, Path(manifest['source_cache_root'])/'cache', means, stds,
                                                    device, Path(manifest['condition_cache_root']),
                                                    Path(manifest['legacy_condition_cache_root']) if manifest.get('legacy_condition_cache_root') else None)
            sync = network.no_sync() if world > 1 and micro+1 < accumulation else nullcontext()
            with sync:
                loss, per_horizon = losses(network, features, actions, layout, stage, horizon)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite sequence loss')
                (loss/accumulation).backward()
            average += loss.detach()/accumulation
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        completed = step+1
        if completed == start+1 or completed % 25 == 0 or completed == total:
            if world > 1:
                dist.all_reduce(average)
                average /= world
            if rank == 0:
                row = dict(stage=stage, step=completed, total_steps=total, loss=float(average), horizon=horizon,
                           sampled_clips=sampler.sampled, distinct_clips_seen=min(sampler.sampled, len(clips)),
                           pool_clips=len(clips), epochs=sampler.sampled/len(clips), grad_norm=float(gradient),
                           elapsed_s=time.monotonic()-started)
                with (stage_out/'training_progress.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                print('TRAIN_PROGRESS', json.dumps(row), flush=True)
        if completed % config['ckpt_every'] == 0 or completed == total:
            state = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state() if device.startswith('cuda') else None)
            rank_rng = [None]*world
            if world > 1:
                dist.all_gather_object(rank_rng, state)
            else:
                rank_rng[0] = state
            if rank == 0:
                tmp = last.with_suffix('.partial')
                torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), config=config, stage=stage,
                     step=completed, sampler=sampler.state_dict(), rank_rng=rank_rng, manifest_sha256=manifest_hash), tmp)
                tmp.replace(last)
    if rank == 0:
        tmp = stage_out/'featsim_ltf.partial'
        torch.save(dict(model=model.state_dict(), mean=norm['mean'], std=norm['std'], predictor_kwargs=kwargs,
                        feature_cut=config.get('feature_cut','ltf/pre_fusion_camera_stages_v1'), ltf_sha256=manifest['ltf_sha256'],
                        config=config, stage=stage, step=total, history_frames=4, step_seconds=.5,
                        prediction_mode='sliding_ar', manifest_sha256=manifest_hash), tmp)
        tmp.replace(stage_out/'featsim_ltf.pt')
        write_json(stage_out/'complete.json', dict(stage=stage, step=total, distinct_clips_seen=min(sampler.sampled,len(clips)),
                   sampled_clips=sampler.sampled, pool_clips=len(clips), manifest_sha256=manifest_hash,
                   checkpoint_sha256=sha256_file(stage_out/'featsim_ltf.pt')))
        print('TRAIN_STAGE_COMPLETE', stage, flush=True)
    if world > 1:
        dist.barrier()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    print('config: '+json.dumps(config), flush=True)
    if config['history_frames'] != 4 or config['step_seconds'] != .5 or config['warp_levels'] not in ([], [0, 1]):
        raise ValueError('Four-frame 0.5 s prediction accepts only the registered no-warp/levels-0,1 pair')
    stages = training_stages(config)
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if world != config['world_size']:
        raise ValueError('GPU world size differs from configured training budget')
    if config['device'].startswith('cuda'):
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    if world > 1:
        dist.init_process_group('nccl' if config['device'].startswith('cuda') else 'gloo', timeout=timedelta(minutes=30))
    try:
        manifest = validate_manifest(config, args.out/'sequences.json')
        for stage in stages:
            train_stage(config, manifest, args.out, stage, rank, world)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
