"""Four-frame clips from verified adjacent cached transitions, without replacement sampling."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import pickle
import sys

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / 'core'))
from protocol import AR_PROTOCOL, sha256_file, write_json


def training_stages(config):
    """The paper recipe trains TF only; older camera experiments can retain TF then RO."""
    stages = tuple(config.get('training_stages', ('tf', 'ro')))
    if stages not in (('tf',), ('tf', 'ro')):
        raise ValueError('training_stages must be [tf] or [tf, ro] in that order')
    return stages


def regular_timestamps(timestamps, step_seconds=.5, tolerance=.1):
    return all(abs((b-a)/1e6-step_seconds) <= tolerance
               for a, b in zip(timestamps, timestamps[1:]))


def make_clips(pairs, frame_count, timestamps):
    """Every edge is a verified same-log, adjacent 0.5 s transition."""
    edges = {}
    for pair in pairs:
        if pair['gap'] != 1:
            continue
        key = (pair['log'], pair['source'])
        if key in edges:
            raise ValueError('Ambiguous next frame in the cache')
        edges[key] = pair
    clips = []
    rejected = Counter()
    for key in sorted(edges):
        log, source = key
        frames, conditions = [source], []
        for _ in range(frame_count-1):
            edge = edges.get((log, frames[-1]))
            if edge is None:
                rejected['missing_adjacent_cached_edge'] += 1
                break
            frames.append(edge['target'])
            conditions.append(edge['condition'])
        else:
            if len(set(frames)) != len(frames):
                raise ValueError('Cyclic frame chain')
            times = [timestamps[(log, token)] for token in frames]
            if not regular_timestamps(times):
                rejected['timestamp_gap'] += 1
                continue
            clips.append(dict(log=log, frames=frames, conditions=conditions))
    return clips, dict(rejected)


def aligned_steps(clip_count, global_batch, stage):
    # Match the sampled-clips/base-clips ratio of the verified nuScenes run.
    reference_clips, reference_samples = {'tf': (25330, 96000), 'ro': (20430, 48000)}[stage]
    if clip_count < 1 or global_batch < 1:
        raise ValueError('Nonempty clips and positive batch required')
    return math.ceil(clip_count*reference_samples/reference_clips/global_batch)


class EpochBatchSampler:
    """Global batches traverse shuffled epochs; a final short epoch wraps without dropping clips."""
    def __init__(self, size, batch_size, seed=0):
        if size < 1 or batch_size < 1:
            raise ValueError('Invalid sampler size/batch')
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch, self.cursor, self.sampled = 0, 0, 0
        self._order = None

    def _permutation(self):
        import torch
        if self._order is None:
            g = torch.Generator().manual_seed(self.seed+self.epoch)
            self._order = torch.randperm(self.size, generator=g).tolist()
        return self._order

    def next(self):
        batch = []
        while len(batch) < self.batch_size:
            take = min(self.batch_size-len(batch), self.size-self.cursor)
            batch.extend(self._permutation()[self.cursor:self.cursor+take])
            self.cursor += take
            self.sampled += take
            if self.cursor == self.size:
                self.epoch += 1
                self.cursor, self._order = 0, None
        return batch

    def state_dict(self):
        return {k: getattr(self, k) for k in ('size', 'batch_size', 'seed', 'epoch', 'cursor', 'sampled')}

    def load_state_dict(self, state):
        if any(state[k] != getattr(self, k) for k in ('size', 'batch_size', 'seed')):
            raise ValueError('Sampler recipe changed on resume')
        if state['sampled'] != state['epoch']*self.size+state['cursor']:
            raise ValueError('Invalid sampler cursor')
        self.epoch, self.cursor, self.sampled = (state[k] for k in ('epoch', 'cursor', 'sampled'))
        self._order = None


def build_manifest(config, out):
    from parallel_helpers import verify_cache
    source = json.loads(Path(config['source_config']).read_text())
    cache = Path(source['cache_root'])
    verify_cache(source, cache)
    if config['history_frames'] != 4 or config['step_seconds'] != .5:
        raise ValueError('This experiment fixes four-frame history and 0.5 s steps')
    if config['ltf_checkpoint'] != source['ltf_checkpoint']:
        raise ValueError('Sequence teacher differs from cached teacher')
    cached_pairs = json.loads((cache / 'pairs.json').read_text())
    logs = sorted({p['log'] for p in cached_pairs})
    # Older variable-gap sources contain useful frames that were never selected
    # as a target. Keep them; only their adjacent symbolic conditions are missing.
    wanted = {(p['log'], p[field]) for p in cached_pairs for field in ('source', 'target')}
    existing_conditions = {p['condition'] for p in cached_pairs if p['gap'] == 1}
    timestamps, metadata_hashes, pairs, extra_conditions = {}, {}, [], []
    log_root = Path(source['train']['openscene_root']) / 'navsim_logs' / source['train']['data_split']
    for i, log in enumerate(logs, 1):
        path = log_root / (log+'.pkl')
        payload = path.read_bytes()
        import hashlib
        metadata_hashes[log] = hashlib.sha256(payload).hexdigest()
        raw = pickle.loads(payload)
        for frame in raw:
            key = (log, frame['token'])
            if key in wanted:
                timestamps[key] = int(frame['timestamp'])
        for previous, frame in zip(raw, raw[1:]):
            if (log, previous['token']) in wanted and (log, frame['token']) in wanted:
                condition = frame['token']+'_1.pt'
                edge = dict(log=log, source=previous['token'], target=frame['token'], gap=1, condition=condition)
                pairs.append(edge)
                if condition not in existing_conditions:
                    extra_conditions.append(edge)
        if i == 1 or i % 50 == 0 or i == len(logs):
            print('SEQUENCE_INDEX_PROGRESS', i, len(logs), flush=True)
    if set(timestamps) != wanted:
        raise ValueError('Cached frame missing from original metadata')
    stages, audit = {}, {}
    batch = config['world_size']*config['batch_size']*config.get('gradient_accumulation_steps', 1)
    for stage in training_stages(config):
        length = 5 if stage == 'tf' else 4+config['rollout_horizon']
        clips, rejected = make_clips(pairs, length, timestamps)
        steps = aligned_steps(len(clips), batch, stage)
        stages[stage] = dict(clips=clips, steps=steps)
        audit[stage] = dict(clip_count=len(clips), clip_frames=length, steps=steps,
                            sampled_clips=steps*batch, epoch_equivalents=steps*batch/len(clips),
                            rejected_starts=rejected)
    used_conditions = {name for stage in stages.values() for clip in stage['clips'] for name in clip['conditions']}
    extra_conditions = [edge for edge in extra_conditions if edge['condition'] in used_conditions]
    manifest = dict(protocol=AR_PROTOCOL, config=config, source_cache_root=str(cache),
                    condition_cache_root=config.get('condition_cache_root', str(Path(out)/'conditions')), extra_conditions=extra_conditions,
                    source_artifacts={name: sha256_file(cache/name) for name in
                                      ('pairs.json', 'normalization.pt', 'CACHE_COMPLETE.json')},
                    ltf_sha256=sha256_file(config['ltf_checkpoint']), stages=stages,
                    metadata_sha256=metadata_hashes, audit=audit)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    destination = out/'sequences.json'
    if destination.exists():
        if json.loads(destination.read_text()) != manifest:
            raise ValueError('Existing sequence manifest differs; use a new run')
    else:
        write_json(destination, manifest)
    write_json(out/'sequence_audit.json', dict(protocol=AR_PROTOCOL, stages=audit,
               manifest_sha256=sha256_file(destination), training_logs=len(logs),
               cached_frames=len(wanted), adjacent_transitions=len(pairs), new_conditions=len(extra_conditions),
               coverage_policy='shuffle each epoch, no replacement within epoch, no dropped clips',
               reuse='All originally cached frames retained; missing adjacent symbolic conditions use a new directory'))
    print('SEQUENCE_INDEX_COMPLETE', json.dumps(audit), flush=True)
    return manifest


def validate_manifest(config, path):
    manifest = json.loads(Path(path).read_text())
    if manifest['config'] != config or manifest['protocol'] != AR_PROTOCOL:
        raise ValueError('Sequence manifest recipe mismatch')
    cache = Path(manifest['source_cache_root'])
    for name, digest in manifest['source_artifacts'].items():
        if sha256_file(cache/name) != digest:
            raise ValueError('Source cache artifact changed: '+name)
    if sha256_file(config['ltf_checkpoint']) != manifest['ltf_sha256']:
        raise ValueError('Frozen LTF changed')
    marker = json.loads((Path(path).parent/'CONDITIONS_COMPLETE.json').read_text())
    if marker['manifest_sha256'] != sha256_file(path) or marker['conditions'] != len(manifest['extra_conditions']):
        raise ValueError('Adjacent symbolic conditions are not complete')
    return manifest


def prepare_conditions(config, out, rank=0, world=1, limit=None):
    """CPU-only map/box conditioning; reads JPEG dimensions, never extracts image features."""
    import torch
    from PIL import Image
    from navsim.common.dataclasses import Scene, SensorConfig
    from scene_layout import action_between, condition_layout, CAMS
    manifest = json.loads((Path(out)/'sequences.json').read_text())
    source = json.loads(Path(config['source_config']).read_text())
    root = Path(source['train']['openscene_root'])
    log_root = root/'navsim_logs'/source['train']['data_split']
    sensor_root = root/'sensor_blobs'/source['train']['data_split']
    directory = Path(manifest['condition_cache_root'])
    directory.mkdir(parents=True, exist_ok=True)
    by_log = {}
    for edge in manifest['extra_conditions']:
        by_log.setdefault(edge['log'], []).append(edge)
    done = 0
    for log in sorted(by_log)[rank::world]:
        path = log_root/(log+'.pkl')
        if sha256_file(path) != manifest['metadata_sha256'][log]:
            raise ValueError('Training metadata changed after sequence indexing')
        raw = pickle.loads(path.read_bytes())
        frames = {f['token']: f for f in raw}
        for edge in by_log[log]:
            path = directory/edge['condition']
            if not path.exists():
                selected = [frames[edge['source']], frames[edge['target']]]
                scene = Scene.from_scene_dict_list(selected, sensor_root, num_history_frames=2,
                                                    num_future_frames=0, sensor_config=SensorConfig.build_no_sensors())
                shapes = {}
                for cam in CAMS:
                    with Image.open(sensor_root/selected[1]['cams'][cam]['data_path']) as image:
                        shapes[cam] = (image.height, image.width)
                value = dict(action=action_between(selected, 1, target_index=1),
                             layout=condition_layout(scene, selected, 1, target_index=1, image_shapes=shapes))
                temp = path.with_suffix('.partial')
                torch.save(value, temp)
                temp.replace(path)
            done += 1
            if done == 1 or done % 50 == 0:
                print('CONDITION_PROGRESS', rank, done, flush=True)
            if limit is not None and done >= limit:
                return
    write_json(Path(out)/f'conditions_rank_{rank:02d}.json', dict(rank=rank, world=world, conditions=done,
               manifest_sha256=sha256_file(Path(out)/'sequences.json')))


def verify_conditions(out, world):
    out = Path(out)
    manifest_path = out/'sequences.json'
    manifest = json.loads(manifest_path.read_text())
    digest = sha256_file(manifest_path)
    total = 0
    for rank in range(world):
        receipt = json.loads((out/f'conditions_rank_{rank:02d}.json').read_text())
        if receipt['rank'] != rank or receipt['world'] != world or receipt['manifest_sha256'] != digest:
            raise ValueError('Condition worker recipe differs')
        total += receipt['conditions']
    if total != len(manifest['extra_conditions']):
        raise ValueError('Condition coverage count differs')
    directory = Path(manifest['condition_cache_root'])
    for edge in manifest['extra_conditions']:
        if not (directory/edge['condition']).is_file():
            raise ValueError('Missing adjacent condition '+edge['condition'])
    write_json(out/'CONDITIONS_COMPLETE.json', dict(conditions=total, manifest_sha256=digest))
    print('CONDITIONS_COMPLETE', total, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--phase', choices=('index', 'conditions', 'verify-conditions'), default='index')
    ap.add_argument('--limit', type=int, help='CPU interface check only; does not publish a completion receipt')
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    print('config: '+json.dumps(config), flush=True)
    if args.phase == 'index':
        build_manifest(config, args.out)
    elif args.phase == 'conditions':
        import os
        source = json.loads(Path(config['source_config']).read_text())
        sys.path.insert(0, source['flowsim_root'])
        os.environ['NUPLAN_MAPS_ROOT'] = str(Path(source['train']['openscene_root'])/'maps')
        os.environ['NUPLAN_MAP_VERSION'] = 'nuplan-maps-v1.0'
        prepare_conditions(config, args.out, int(os.environ.get('RANK',0)), int(os.environ.get('WORLD_SIZE',1)), args.limit)
    else:
        verify_conditions(args.out, config['world_size'])


if __name__ == '__main__':
    main()
