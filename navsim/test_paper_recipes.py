"""Paper recipe regressions: configuration, BEV cache handoff and TF-only dispatch."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(HERE/'core')]

from protocol import sha256_file, write_json
from sequence_data import build_manifest, training_stages, validate_manifest, verify_conditions


def test_configure_defaults_to_paper_bev_tf(tmp_path):
    import configure
    class Omega:
        @staticmethod
        def load(path):
            return SimpleNamespace(split=Path(path).stem, num_history_frames=4, num_future_frames=8)
    class Loader:
        def __init__(self, logs, sensors, filt, sensor_config):
            token = filt.split+'-token'
            self.tokens_stage_one = [token]
            self.scene_frames_dicts = {token: [dict(log_name=filt.split+'-log')]*4}
    modules = {
        'hydra.utils': SimpleNamespace(instantiate=lambda value: value),
        'omegaconf': SimpleNamespace(OmegaConf=Omega),
        'navsim.common.dataloader': SimpleNamespace(SceneLoader=Loader),
        'navsim.common.dataclasses': SimpleNamespace(SensorConfig=SimpleNamespace(build_no_sensors=lambda: None)),
    }
    out = tmp_path/'recipe'
    argv = ['configure.py', '--flowsim-root', str(tmp_path/'upstream'), '--dataset-root', str(tmp_path/'data'),
            '--train-filter', str(tmp_path/'train.yaml'), '--test-filter', str(tmp_path/'test.yaml'),
            '--ltf-checkpoint', str(tmp_path/'ltf.ckpt'), '--flowsim-checkpoint', str(tmp_path/'flowsim.ckpt'),
            '--out', str(out)]
    with patch.dict(sys.modules, modules), patch.object(sys, 'argv', argv):
        configure.main()
    camera = json.loads((out/'camera_sequence.json').read_text())
    sequence = json.loads((out/'sequence.json').read_text())
    evaluation = json.loads((out/'evaluation.json').read_text())
    assert camera['training_stages'] == sequence['training_stages'] == ['tf']
    assert sequence['world_size'] == 8 and sequence['batch_size'] == 1
    assert sequence['feature_cut'] == 'ltf/final_fused_bev_v1'
    assert sequence['warp_levels'] == []
    assert Path(sequence['reference_sequences']) == out/'camera_sequence/sequences.json'
    assert Path(evaluation['featsim_checkpoint']) == out/'sequence/tf/featsim_ltf.pt'
    assert evaluation['expected_feature_cut'] == sequence['feature_cut']
    assert evaluation['expected_checkpoint_stage'] == 'tf'
    assert evaluation['expected_checkpoint_step'] == 45495
    assert evaluation['expected_warp_levels'] == []
    from eval_sequence import evaluation_config
    effective = evaluation_config(evaluation, out/'sequence')
    for field in ('expected_feature_cut', 'expected_checkpoint_stage', 'expected_checkpoint_step', 'expected_warp_levels'):
        assert effective[field] == evaluation[field]
    assert effective['featsim_checkpoint'] == evaluation['featsim_checkpoint']


def camera_manifest(tmp_path):
    source_cache = tmp_path/'camera_cache'
    source_cache.mkdir()
    teacher = tmp_path/'teacher'
    teacher.write_bytes(b'fixture teacher')
    metadata = tmp_path/'dataset/navsim_logs/trainval'
    metadata.mkdir(parents=True)
    # Only five frames: sufficient for TF, deliberately too short for RO8.
    frames = [dict(token=str(i), timestamp=i*500000) for i in range(5)]
    (metadata/'log.pkl').write_bytes(pickle.dumps(frames))
    pairs = [dict(log='log', source=str(i), target=str(i+1), gap=1, condition=f'{i+1}_1.pt') for i in range(4)]
    write_json(source_cache/'pairs.json', pairs)
    (source_cache/'normalization.pt').write_bytes(b'camera normalization fixture')
    write_json(source_cache/'CACHE_COMPLETE.json', dict(complete=True))
    source = dict(cache_root=str(source_cache), ltf_checkpoint=str(teacher),
                  train=dict(openscene_root=str(tmp_path/'dataset'), data_split='trainval'))
    source_path = tmp_path/'source.json'
    write_json(source_path, source)
    config = dict(source_config=str(source_path), ltf_checkpoint=str(teacher), history_frames=4,
                  step_seconds=.5, world_size=1, batch_size=1, rollout_horizon=8, training_stages=['tf'],
                  warp_levels=[], device='cpu', seed=0, tf_lr=.0003, ro_lr=.0001, ckpt_every=2000)
    out = tmp_path/'camera_sequence'
    with patch('parallel_helpers.verify_cache', return_value={}):
        manifest = build_manifest(config, out)
    write_json(out/'conditions_rank_00.json', dict(rank=0, world=1, conditions=0,
               manifest_sha256=sha256_file(out/'sequences.json')))
    verify_conditions(out, 1)
    return config, manifest, out


def test_tf_index_does_not_require_rollout_clips(tmp_path):
    config, manifest, out = camera_manifest(tmp_path)
    assert set(manifest['stages']) == {'tf'}
    assert len(manifest['stages']['tf']['clips']) == 1
    assert len(manifest['stages']['tf']['clips'][0]['frames']) == 5
    assert validate_manifest(config, out/'sequences.json') == manifest


def test_bev_handoff_and_training_entrypoint_remain_tf_only(tmp_path):
    config, reference, camera_out = camera_manifest(tmp_path)
    config = {**config, 'feature_cut': 'ltf/final_fused_bev_v1',
              'reference_sequences': str(camera_out/'sequences.json')}
    out = tmp_path/'bev_sequence'
    cache = out/'bev_features'
    directory = cache/'cache'
    directory.mkdir(parents=True)
    for index in range(5):
        torch.save([torch.zeros(512, 8, 8)], directory/f'{index}.pt')
    count = 5*64
    torch.save(dict(sums=torch.zeros(512), squares=torch.ones(512)*count, count=count), cache/'moments_00.pt')
    write_json(cache/'rank_00.json', dict(rank=0, world=1, frames=5,
               reference_hash=sha256_file(config['reference_sequences']),
               ltf_sha256=sha256_file(config['ltf_checkpoint']), moments_sha256=sha256_file(cache/'moments_00.pt')))
    import prepare_bev_cache
    prepare_bev_cache.merge(config, out)
    manifest = validate_manifest(config, out/'sequences.json')
    assert set(manifest['stages']) == {'tf'}
    assert Path(manifest['source_cache_root']) == cache
    assert manifest['legacy_condition_cache_root'] == str(Path(reference['source_cache_root'])/'cache')
    norm = torch.load(cache/'normalization.pt', weights_only=False)
    assert len(norm['mean']) == 1 and norm['mean'][0].shape == (512,)
    torch.testing.assert_close(norm['std'][0], torch.ones(512))
    config_path = tmp_path/'bev.json'
    write_json(config_path, config)
    import train_sequence
    called = []
    with patch.object(sys, 'argv', ['train_sequence.py', '--config', str(config_path), '--out', str(out)]), \
         patch.dict(os.environ, {'RANK': '0', 'WORLD_SIZE': '1', 'LOCAL_RANK': '0'}), \
         patch.object(train_sequence, 'train_stage', side_effect=lambda cfg, data, dest, stage, rank, world: called.append(stage)):
        train_sequence.main()
    assert called == ['tf']
    assert not (out/'ro').exists()


@pytest.mark.parametrize('stages', [[], ['ro'], ['ro', 'tf'], ['unknown']])
def test_invalid_training_stage_selection_is_rejected(stages):
    with pytest.raises(ValueError, match='training_stages'):
        training_stages({'training_stages': stages})


def test_legacy_tf_then_rollout_configuration_is_preserved():
    assert training_stages({}) == ('tf', 'ro')
    assert training_stages({'training_stages': ['tf', 'ro']}) == ('tf', 'ro')


@pytest.mark.parametrize('stage', [0, 1, 2])
def test_uniad_documented_training_commands_match_paper(stage):
    text = (REPO/'uniad/README.md').read_text(encoding='utf8')
    commands = [block.replace('\\\n', ' ') for block in re.findall(r'```bash\n(.*?)\n```', text, re.S)
                if block.startswith('torchrun')]
    assert len(commands) == 3
    tokens = shlex.split(commands[stage])
    assert '--nproc_per_node=8' in tokens
    tree = ast.parse((REPO/'uniad/feat/train_pred.py').read_text(encoding='utf8'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('get_args', 'parse_F_schedule')]
    namespace = {'argparse': argparse}
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<UniAD arguments>', 'exec'), namespace)
    args = namespace['get_args'](tokens[tokens.index('uniad/feat/train_pred.py')+1:])
    assert args.residual and args.warp_levels == [0, 1] and args.map_layout
    assert not args.assoc_layout
    assert args.bs == 1 and args.depth == 8 and args.ctx == 4 and args.wd == .04
    assert args.steps == (12000 if stage == 0 else 6000)
    assert args.lr == (.0003 if stage == 0 else .0001)
    assert args.warmup == (500 if stage == 0 else 300)
    assert args.sweeps == (stage != 0)
    if stage == 0:
        assert args.init is None and args.rollout == 'tf' and args.T == 5
    else:
        assert args.init == '$UNIAD_BENCH/train_tf_init/iter_12000.pt'
        assert args.rollout == ('tf' if stage == 1 else 'sliding')
    if stage == 2:
        assert args.T == 12 and args.F == 8
        assert args.sched == [(0, 2), (1000, 4), (2000, 8)]
