"""The review candidate's standalone merger consumes complete, disjoint shards."""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from a_merge import KS, load_eval, summarize


def fixture(root, rank, scene=None):
    scene = rank if scene is None else scene
    data = dict(arms=['copy_last'], ks=KS, ckpt='hf', ctx=4, preds={}, scenes=[scene],
                sec_rollout={}, sec_per_uniad_step=.1, wall_s=1.,
                rec={'copy_last': {k: {'plan_ade': np.array([float(rank)]), 'det_n_ref': np.array([rank])} for k in KS}},
                meta={k: {'scene': np.array([scene]), 'target_token': np.array([f'token-{rank}'])} for k in KS},
                ref={'sample_token': np.array([f'token-{rank}'])})
    (root/f'shard{rank}of2.pkl').write_bytes(pickle.dumps(data))


def test_merger_cli_and_sample_mean(tmp_path):
    fixture(tmp_path, 0)
    fixture(tmp_path, 1)
    completed = subprocess.run([sys.executable, str(Path(__file__).with_name('a_merge.py')), str(tmp_path),
                                '--boot', '20'], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    data = json.loads((tmp_path/'summary.json').read_text())
    assert data['summary']['geom']['copy_last']['1']['plan_ade']['mean'] == .5
    assert data['summary']['refdet']['copy_last']['1']['plan_ade']['mean'] == 1.
    assert (tmp_path/'summary.md').exists()


def test_incomplete_and_overlapping_shards_fail(tmp_path):
    fixture(tmp_path, 0)
    with pytest.raises(AssertionError):
        load_eval(tmp_path)
    fixture(tmp_path, 1, scene=0)
    with pytest.raises(ValueError, match='overlap'):
        load_eval(tmp_path)


def test_paired_differences_keep_the_reference_detection_mask():
    meta = {k: {'scene': np.array(['a', 'b'])} for k in KS}
    records = {
        'copy_last': {k: {'plan_ade': np.array([5., 9.]), 'det_n_ref': np.array([0, 1])} for k in KS},
        'pred:test': {k: {'plan_ade': np.array([1., 7.]), 'det_n_ref': np.array([0, 1])} for k in KS},
    }
    result = summarize(list(records), records, meta, boot=20)
    assert result['paired_differences']['geom']['pred:test']['1']['plan_ade']['diff'] == -3.
    assert result['paired_differences']['refdet']['pred:test']['1']['plan_ade']['diff'] == -2.
