"""Merge on-log evaluation shards; report only locally evaluated arms.

Historical BEV-warp result archives are not required or mixed into this summary.
"""
import argparse
import glob
import json
import os
import pickle
import re
from pathlib import Path
import numpy as np
from aggregation import cluster_boot

KS = (1, 3, 5, 8)
DT = {k: k*.5 for k in KS}

def load_eval(d):
    shards = sorted(glob.glob(os.path.join(d, 'shard*of*.pkl')))
    assert shards, f'{d} 下没有 shard*.pkl'
    parts = [pickle.load(open(p, 'rb')) for p in shards]
    ns = {int(os.path.basename(p).split('of')[1].split('.')[0]) for p in shards}
    assert len(ns) == 1 and len(shards) == ns.pop(), f'分片不全: {shards}'
    indices = [int(re.fullmatch(r'shard(\d+)of\d+\.pkl', Path(p).name).group(1)) for p in shards]
    if sorted(indices) != list(range(len(shards))):
        raise ValueError('Missing or repeated shard index')
    arms = parts[0]['arms']
    assert all(p['arms'] == arms and p['ks'] == parts[0]['ks'] for p in parts)
    if tuple(parts[0]['ks']) != KS or any(p['ckpt'] != parts[0]['ckpt'] for p in parts):
        raise ValueError('Checkpoint family or horizon protocol differs across shards')
    scenes = [scene for part in parts for scene in part['scenes']]
    if len(scenes) != len(set(scenes)):
        raise ValueError('Scene ownership overlaps across shards')
    rec = {n: {k: {m: np.concatenate([p['rec'][n][k][m] for p in parts]) for m in parts[0]['rec'][n][k]} for k in KS} for n in arms}
    meta = {k: {f: np.concatenate([p['meta'][k][f] for p in parts]) for f in parts[0]['meta'][k]} for k in KS}
    ref = {f: np.concatenate([p['ref'][f] for p in parts]) for f in parts[0]['ref']}
    info = dict(ckpt=parts[0]['ckpt'], ctx=parts[0]['ctx'], n_shards=len(parts), preds=parts[0]['preds'],
                scenes=sum((p['scenes'] for p in parts), []),
                sec_rollout={k: float(np.mean([p['sec_rollout'][k] for p in parts if k in p['sec_rollout']])) for k in parts[0]['sec_rollout']},
                sec_uniad=float(np.mean([p['sec_per_uniad_step'] for p in parts])),
                wall_s_max=float(max(p['wall_s'] for p in parts)), wall_s_sum=float(sum(p['wall_s'] for p in parts)))
    for k in KS:  # 各臂行顺序必须与 meta 一致
        if len(meta[k]['target_token']) != len(set(meta[k]['target_token'])):
            raise ValueError('Duplicate target token at the same horizon')
        for n in arms:
            assert all(len(value) == len(meta[k]['target_token']) for value in rec[n][k].values()), (n, k)
    return arms, rec, meta, ref, info


def summarize(arms, records, meta, boot=1000):
    result = {subset: {} for subset in ('geom', 'refdet')}
    result['paired_differences'] = {subset: {} for subset in ('geom', 'refdet')}
    for subset in ('geom', 'refdet'):
        for arm in arms:
            result[subset][arm] = {}
            for k in KS:
                groups = np.asarray(meta[k]['scene'])
                mask = np.ones(len(groups), dtype=bool) if subset == 'geom' else np.asarray(records[arms[0]][k]['det_n_ref']) > 0
                result[subset][arm][str(k)] = {}
                for name, values in records[arm][k].items():
                    values = np.where(mask, np.asarray(values, dtype=float), np.nan)
                    mean, lower, upper = cluster_boot(values, groups, boot, np.random.default_rng(0))
                    result[subset][arm][str(k)][name] = dict(mean=mean, lower95=lower, upper95=upper,
                                                         finite_samples=int(np.isfinite(values).sum()))
                    if arm.startswith('pred:') and 'copy_last' in records and name in records['copy_last'][k]:
                        differences = values-np.asarray(records['copy_last'][k][name], dtype=float)
                        delta, lo, hi = cluster_boot(differences, groups, boot, np.random.default_rng(0))
                        result['paired_differences'][subset].setdefault(arm, {}).setdefault(str(k), {})[name] = dict(
                            diff=delta, lower95=lo, upper95=hi, finite_samples=int(np.isfinite(differences).sum()))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('eval_dir', type=Path)
    ap.add_argument('--boot', type=int, default=1000)
    args = ap.parse_args()
    arms, records, meta, reference, info = load_eval(args.eval_dir)
    summary = summarize(arms, records, meta, args.boot)
    def finite_json(value):
        if isinstance(value, dict):
            return {str(k): finite_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    payload = finite_json(dict(info=info, aggregation='sample mean; scene-cluster bootstrap', summary=summary))
    (args.eval_dir/'summary.json').write_text(json.dumps(payload, indent=2, default=str, allow_nan=False)+'\n')
    lines = ['# On-log agreement', '', '| Subset | Arm | Horizon (s) | Metric | Mean | 95% CI |',
             '|---|---|---:|---|---:|---|']
    for subset in ('geom', 'refdet'):
        for arm, horizons in summary[subset].items():
            for k, metrics in horizons.items():
                for metric, row in metrics.items():
                    lines.append(f"| {subset} | {arm} | {DT[int(k)]} | {metric} | {row['mean']:.6g} | [{row['lower95']:.6g}, {row['upper95']:.6g}] |")
    (args.eval_dir/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
