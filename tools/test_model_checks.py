"""Integrity failures must be caught before loading released model weights."""
import json

import pytest
import torch

from check_models import check_metadata, normalization_check, state_sha256
from check_release import inspect


def test_state_hash_detects_tensor_changes_but_not_dictionary_order():
    first = {'a': torch.tensor([1., 2.]), 'b': torch.tensor([3.])}
    assert state_sha256(first) == state_sha256(dict(reversed(list(first.items()))))
    second = {key: value.clone() for key, value in first.items()}
    second['a'][0] += 1
    assert state_sha256(first) != state_sha256(second)


def test_nonpositive_or_nonfinite_normalization_is_rejected():
    normalization_check([torch.zeros(2)], [torch.ones(2)], [2])
    for bad in (torch.zeros(2), torch.tensor([1., float('nan')])):
        with pytest.raises(ValueError, match='positive standard deviation'):
            normalization_check([torch.zeros(2)], [bad], [2])
    with pytest.raises(ValueError, match='Unsupported'):
        check_metadata(object())


def test_source_audit_accepts_model_manifest_and_ignores_runtime_weights(tmp_path):
    (tmp_path/'models').mkdir()
    manifest = tmp_path/'models/manifest.json'
    manifest.write_text(json.dumps({'models': {}}))
    (tmp_path/'artifacts/models').mkdir(parents=True)
    (tmp_path/'artifacts/models/weights.pt').write_bytes(b'runtime binary')
    assert inspect(tmp_path)['status'] == 'pass'
    manifest.write_text('{invalid json')
    assert inspect(tmp_path)['findings'][0]['check'] == 'json_syntax'
