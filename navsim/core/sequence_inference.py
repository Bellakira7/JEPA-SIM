"""Causal real-anchor four-frame forecasting on the original logged timeline."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sequence_data import regular_timestamps


def select_tokens(loader, config):
    original = json.loads(Path(config['filter_manifest']).read_text())['test']['tokens']
    if len(set(original)) != len(original):
        raise ValueError('Duplicate original test token')
    absent = sorted(set(original)-set(loader.tokens_stage_one))
    gap_tokens, selected = [], []
    end = config['history_frames']+max(config['gaps'])-1
    for token in sorted(set(original)&set(loader.tokens_stage_one)):
        raw = loader.scene_frames_dicts[token]
        if len(raw) < end+9 or raw[end]['token'] != token:
            raise ValueError('Sequence loader target is misaligned')
        if not regular_timestamps([int(f['timestamp']) for f in raw[:end+9]]):
            gap_tokens.append(token)
        else:
            selected.append(token)
    if not selected:
        raise ValueError('No camera-complete, temporally valid sequence evaluation tokens')
    return selected, dict(original_tokens=len(original), selected_tokens=len(selected),
                          insufficient_history_or_future=absent, timestamp_gaps=gap_tokens)


def predict_stages(agent, predictor, builder, inputs, scene, raw, gap, mean, std, device,
                   feature_cache, condition_cache):
    import torch
    from ltf_features import extract_camera_stages
    from scene_layout import action_between, condition_layout
    from train_sequence import forecast
    target = scene.scene_metadata.num_history_frames-1
    anchor = target-gap
    first = anchor-3
    if first < 0:
        raise ValueError('Four real history frames must end at or before the anchor')
    for i in range(first, anchor+1):
        if i not in feature_cache:
            image = builder._get_camera_feature(inputs, idx=i).unsqueeze(0).to(device)
            feature_cache[i], _ = extract_camera_stages(agent, image)
    # Each transition carries its own motion and next-frame logged layout.
    for i in range(first, target):
        if i not in condition_cache:
            condition_cache[i] = (action_between(raw, 1, target_index=i+1).to(device),
                                  condition_layout(scene, raw, 1, target_index=i+1).to(device))
    history = [torch.stack([((feature_cache[i][l]-mean[l])/std[l])[0]
                            for i in range(first, anchor+1)])[None, :, None] for l in range(4)]
    actions = torch.stack([condition_cache[i][0] for i in range(first, target)])[None]
    layout = torch.stack([condition_cache[i][1] for i in range(first, target)])[None, :, None]
    dense, ar = forecast(predictor, history, actions, layout, gap)
    generated = dense if gap == 1 else ar
    return [p[:, -1, 0]*s+m for p, m, s in zip(generated, mean, std)]
