"""Disjoint shards and strict whole-run aggregation for OpenScene training."""
from collections import Counter
import json
import os
from pathlib import Path


def visible_gpu_identifiers(world, environment=None):
    environment = os.environ if environment is None else environment
    mask = environment.get('CUDA_VISIBLE_DEVICES')
    identifiers = [value.strip() for value in mask.split(',')] if mask is not None else [str(i) for i in range(world)]
    if len(identifiers) != world or any(not value or value == '-1' for value in identifiers):
        raise ValueError('Platform GPU visibility mask differs from the requested workers')
    return identifiers


def assigned_logs(token_logs, world):
    if world < 1:
        raise ValueError('Positive worker count required')
    counts = Counter(token_logs.values())
    result, loads = [set() for _ in range(world)], [0] * world
    for log, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        rank = min(range(world), key=lambda i: (loads[i], i))
        result[rank].add(log)
        loads[rank] += count
    return result


def expected_tokens(config, split):
    tokens = json.loads(Path(config['filter_manifest']).read_text())[split]['tokens']
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError('Frozen manifest must contain unique nonempty tokens')
    return sorted(tokens)


def merge_cache(config, out, world):
    import torch
    from protocol import write_json, sha256_file
    out = Path(out)
    expected = set(expected_tokens(config, 'train'))
    tokens, frames, logs, keys = set(), set(), set(), set()
    pairs, identities = [], []
    sums = squares = counts = None
    teacher_hash = sha256_file(config['ltf_checkpoint'])
    for rank in range(world):
        root = out / 'cache_shards' / f'rank_{rank:02d}'
        info = json.loads((root / 'complete.json').read_text())
        if not info['complete'] or info['config'] != config or info['rank'] != rank or info['world'] != world:
            raise ValueError('Cache shard execution metadata differs')
        if info['ltf_sha256'] != teacher_hash:
            raise ValueError('Cache teacher differs across shards')
        if tokens & set(info['tokens']) or frames & set(info['frames']) or logs & set(info['logs']):
            raise ValueError('Cache shards overlap: frame ownership must be log-disjoint')
        tokens.update(info['tokens'])
        frames.update(info['frames'])
        logs.update(info['logs'])
        identity = json.loads((root / 'feature_cut_identity.json').read_text())
        if identity['max_abs'] > config['identity_tolerance']:
            raise ValueError('Cache feature-cut identity check failed')
        if identities and identity['shapes'] != identities[0]['shapes']:
            raise ValueError('Cache feature shapes differ across ranks')
        identities.append(identity)
        part = json.loads((root / 'pairs.json').read_text())
        part_keys = {(row['token'], row['gap']) for row in part}
        if len(part_keys) != len(part) or part_keys != {(token, gap) for token in info['tokens'] for gap in config['gaps']}:
            raise ValueError('Cache shard has missing or duplicate token/gap pairs')
        if any(row['log'] not in info['logs'] or row['source'] not in info['frames'] or row['target'] not in info['frames'] for row in part):
            raise ValueError('Cache pair references another shard')
        if keys & part_keys:
            raise ValueError('Duplicate training pair across ranks')
        keys.update(part_keys)
        pairs.extend(part)
        moment = torch.load(root / 'moments.pt', map_location='cpu', weights_only=False)
        if sums is None:
            sums = [torch.zeros_like(value) for value in moment['sums']]
            squares = [torch.zeros_like(value) for value in moment['squares']]
            counts = [0] * len(sums)
        for level, (s, q, n) in enumerate(zip(moment['sums'], moment['squares'], moment['counts'])):
            if n <= 0 or not torch.isfinite(s).all() or not torch.isfinite(q).all():
                raise ValueError('Invalid cache normalization statistics')
            sums[level] += s
            squares[level] += q
            counts[level] += n
    if tokens != expected:
        raise ValueError(f'Cache token coverage differs: missing {len(expected-tokens)}, extra {len(tokens-expected)}')
    pairs.sort(key=lambda row: (row['token'], row['gap']))
    means = [s/n for s, n in zip(sums, counts)]
    stds = [(q/n-m.square()).clamp_min(1e-8).sqrt() for q, n, m in zip(squares, counts, means)]
    temporary = out / 'normalization.partial'
    torch.save(dict(mean=[m.float() for m in means], std=[s.float() for s in stds]), temporary)
    temporary.replace(out / 'normalization.pt')
    write_json(out / 'pairs.json', pairs)
    write_json(out / 'feature_cut_identity.json', dict(max_abs=max(row['max_abs'] for row in identities),
        shapes=identities[0]['shapes'], per_rank=identities))
    write_json(out / 'data_summary.json', dict(train_logs=sorted(logs), train_tokens=len(tokens),
        train_pairs=len(pairs), cached_training_frames=len(frames), preparation_workers=world,
        normalization='unique navtrain frames only / global channel moments', ltf_sha256=teacher_hash))
    artifacts = {name: sha256_file(out / name) for name in
        ('normalization.pt', 'pairs.json', 'feature_cut_identity.json', 'data_summary.json')}
    write_json(out / 'CACHE_COMPLETE.json', dict(complete=True, world_size=world, config=config,
        artifacts=artifacts, filter_manifest_sha256=sha256_file(config['filter_manifest'])))
    print('CACHE_COMPLETE', 'tokens', len(tokens), 'pairs', len(pairs), 'workers', world, flush=True)


def verify_cache(config, out):
    """Check the published eight-worker cache before submitting/starting training."""
    from protocol import sha256_file
    out = Path(out)
    marker = json.loads((out / 'CACHE_COMPLETE.json').read_text())
    if not marker.get('complete') or marker['config'] != config:
        raise ValueError('Completed cache belongs to a different configuration')
    if marker['world_size'] != config.get('preparation_workers', config.get('world_size', 1)):
        raise ValueError('Completed cache worker count differs')
    if marker['filter_manifest_sha256'] != sha256_file(config['filter_manifest']):
        raise ValueError('Frozen token manifest changed after extraction')
    names = {'normalization.pt', 'pairs.json', 'feature_cut_identity.json', 'data_summary.json'}
    if set(marker.get('artifacts', {})) != names:
        raise ValueError('Completed cache artifact manifest is incomplete')
    for name, digest in marker['artifacts'].items():
        if sha256_file(out / name) != digest:
            raise ValueError('Completed cache artifact changed: ' + name)
    summary = json.loads((out / 'data_summary.json').read_text())
    if summary['ltf_sha256'] != sha256_file(config['ltf_checkpoint']):
        raise ValueError('Cache teacher changed after extraction')
    expected = {(token, gap) for token in expected_tokens(config, 'train') for gap in config['gaps']}
    pairs = json.loads((out / 'pairs.json').read_text())
    if len(pairs) != len(expected) or {(row['token'], row['gap']) for row in pairs} != expected:
        raise ValueError('Completed cache token/gap coverage differs')
    return dict(complete=True, cache_root=str(out), workers=marker['world_size'],
        train_tokens=summary['train_tokens'], train_pairs=len(pairs), artifacts=marker['artifacts'])


def merge_evaluation(config, out, world):
    from protocol import write_json
    from eval_open_loop import finalize_evaluation
    out = Path(out)
    expected = expected_tokens(config, 'test')
    seen, records, identities = set(), [], []
    artifacts = None
    for rank in range(world):
        root = out / 'shards' / f'rank_{rank:02d}'
        summary = json.loads((root / 'stats.json').read_text())
        part = json.loads((root / 'records.json').read_text())
        token_list = json.loads((root / 'tokens.json').read_text())
        part_identity = json.loads((root / 'identity_checks.json').read_text())
        expected_part = expected[rank::world]
        if summary['status'] != 'complete' or token_list != expected_part:
            raise ValueError('Evaluation shard is incomplete or has unexpected tokens')
        shard_config = dict(summary['config'])
        shard_config['tokens_file'] = config.get('tokens_file')
        if shard_config != config:
            raise ValueError('Evaluation configurations differ across shards')
        if seen & set(token_list):
            raise ValueError('Overlapping evaluation shards')
        if {row['token'] for row in part} != set(token_list) or {row['token'] for row in part_identity['per_token']} != set(token_list):
            raise ValueError('Evaluation record/identity coverage differs')
        if len(part_identity['per_token']) != len(token_list):
            raise ValueError('Duplicate identity checks')
        if artifacts is None:
            artifacts = summary['artifacts']
        elif artifacts != summary['artifacts']:
            raise ValueError('Evaluated checkpoint hashes differ across shards')
        seen.update(token_list)
        records.extend(part)
        identities.extend(part_identity['per_token'])
    if seen != set(expected):
        raise ValueError('Final evaluation token coverage is incomplete')
    records.sort(key=lambda row: (row['token'], row['gap_frames'], row['arm']))
    identities.sort(key=lambda row: row['token'])
    write_json(out / 'tokens.json', expected)
    # Recompute means from all records, including log means; do not average shard means.
    finalize_evaluation(config, out, records, identities, artifacts)
