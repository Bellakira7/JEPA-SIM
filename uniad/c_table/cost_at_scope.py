"""Summarize saved C timings; sum components per step before taking medians."""
import json
from pathlib import Path
from statistics import median

run = Path(__file__).resolve().parents[2]
cfg = json.loads((run / 'config.json').read_text())
out = {}
for scope, scenes in [('val150', cfg['scene_indices']), ('recon30', cfg['recon30_scene_indices'])]:
    table = []
    for arm in cfg['arms']:
        rows = []
        for si in scenes:
            path = run / 'eval' / f'scene{si:03d}__{arm}.jsonl'
            assert path.is_file(), path
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        def value(row, key):
            return float(row.get(key, 0))
        def med(key):
            return median(value(row, key) for row in rows)
        obs = [sum(value(row, key) for key in ['t_cond', 't_pred', 't_render', 't_difix', 't_backbone']) for row in rows]
        step = [t + value(row, 't_uniad') for t, row in zip(obs, rows)]
        table.append(dict(arm=arm, layout=med('t_cond'), prediction=med('t_pred'), difix=med('t_difix'),
                          backbone=med('t_backbone'), obs=median(obs), step=median(step), uniad=med('t_uniad'),
                          n_steps=len(rows), n_scenes=len(scenes), fitting_gpu_hours=0 if arm != 'log' else None))
    out[scope] = table
out['protocol'] = 'nuScenes / C-egostat-sim-v1 / 4 real-image warmup frames / PPU / seconds / pooled-step medians; Obs and Step summed per step; JPEG and offline detection excluded'
dest = run / 'merged'
dest.mkdir(exist_ok=True)
(dest / 'cost.json').write_text(json.dumps(out, indent=2))
lines = [out['protocol']]
for scope in ['val150', 'recon30']:
    lines += ['', '## ' + scope, '', '| 来源 | Layout | 预测/渲染 | Difix | 编码器 | Obs | Step | 逐场景拟合卡时 | 步数 |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in out[scope]:
        cells = [r['arm']] + [f"{r[k]:.4f}" for k in ['layout', 'prediction', 'difix', 'backbone', 'obs', 'step']]
        cells += [str(r['fitting_gpu_hours']) if r['fitting_gpu_hours'] is not None else '--', str(r['n_steps'])]
        lines.append('| ' + ' | '.join(cells) + ' |')
(dest / 'cost.md').write_text('\n'.join(lines) + '\n')
print('C_COST_OK', dest / 'cost.json')
