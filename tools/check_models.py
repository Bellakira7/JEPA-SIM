"""Check released predictor hashes, paper configuration, strict loading and CPU forward finiteness."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8*1024*1024):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state):
    """Hash sorted tensor names, dtypes, shapes and exact contiguous CPU bytes."""
    import torch
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode()+b'\0')
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def check_metadata(value):
    import torch
    if isinstance(value, str):
        if re.search(r'/cpfs\d*/|[A-Za-z]:[\\/](?:Users|workspace)[\\/]|https?://[^/\s]+\.(?:internal|corp)\b', value, re.I):
            raise ValueError('Private location in checkpoint metadata')
    elif isinstance(value, dict):
        for key, item in value.items():
            check_metadata(key)
            check_metadata(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            check_metadata(item)
    elif not isinstance(value, (torch.Tensor, int, float, bool, type(None))):
        raise ValueError('Unsupported checkpoint metadata type')


def require(condition, description):
    if not condition:
        raise ValueError(description)


def normalization_check(means, stds, channels):
    import torch
    require(len(means) == len(stds) == len(channels), 'Normalization level count differs')
    for mean, std, channel in zip(means, stds, channels):
        require(mean.shape == std.shape == (channel,), 'Normalization channel count differs')
        require(bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()),
                'Normalization must be finite with positive standard deviation')


def architecture_check(model):
    import torch
    require(model.D == 384 and len(model.blocks) == 8, 'Paper requires width 384 and eight blocks')
    for block in model.blocks:
        require(block.attn.num_heads == 6, 'Paper requires six attention heads')
        require(block.norm1.eps == block.norm2.eps == 1e-6, 'LayerNorm epsilon differs')
        require(isinstance(block.mlp.act, torch.nn.GELU), 'Paper requires GELU MLPs')
        require(block.mlp.fc1.out_features == 1536, 'MLP ratio differs')
        require(block.ada_ln[-1].out_features == 4*384, 'AdaLN modulation differs')
    adapter = model.layout_adapter
    require(adapter is not None and adapter.replace_main, 'Paper layout detail path is required')
    convs = [layer for layer in adapter.detail if isinstance(layer, torch.nn.Conv2d)]
    require(len(convs) == 3 and [layer.dilation for layer in convs] == [(1, 1), (2, 2), (4, 4)],
            'Layout convolution dilation differs')
    require(all(layer.out_channels == 64 and layer.kernel_size == (3, 3) for layer in convs),
            'Layout convolution shape differs')


def validate(model_name, weights_dir, manifest):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(0)
    repo = Path(__file__).resolve().parents[1]
    entry = manifest['models'][model_name]
    path = Path(weights_dir)/entry['filename']
    require(file_sha256(path) == entry['sha256'], 'Checkpoint file checksum mismatch')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    check_metadata(checkpoint)
    require('opt' not in checkpoint and 'optimizer' not in checkpoint, 'Optimizer must not be included')
    state = checkpoint['model']
    require(all(bool(torch.isfinite(tensor).all()) for tensor in state.values()), 'Nonfinite model tensor')
    require(state_sha256(state) == entry['state_dict_sha256'] == checkpoint['release']['state_dict_sha256'],
            'Model tensors differ from the exported source checkpoint')
    require(checkpoint['release']['source_checkpoint_sha256'] == entry['source_checkpoint_sha256'],
            'Source checkpoint identity differs')
    if model_name == 'uniad':
        sys.path[:0] = [str(repo/'uniad'), str(repo/'uniad/feat')]
        from pyramid_predictor import PyramidPredictor, load_predictor
        model, _ = load_predictor(str(path), 'cpu')
        require(model.L == 4 and model.n_cams == 6 and model.C == 256, 'UniAD feature interface differs')
        require(model.level_grids == [(116, 200), (58, 100), (29, 50), (15, 25)], 'UniAD native grids differ')
        require(model.level_pools == [8, 4, 2, 1] and model.tokens_per_frame == 9000, 'UniAD patch factors differ')
        require(model.residual and model.warp_levels == (0, 1) and model.flow_px == 32., 'UniAD residual/flow differs')
        require(model.layout_assoc is None and model.action_encoder.in_features == 3, 'UniAD conditioning differs')
        architecture_check(model)
        args = checkpoint['args']
        require(checkpoint['step'] == 6000 and args['rollout'] == 'sliding', 'UniAD training stage differs')
        require(args['ctx'] == 4 and args['F'] == 8 and args['T'] == 12, 'UniAD history or rollout horizon differs')
        require(args['F_schedule'] == '0:2,1000:4,2000:8', 'UniAD rollout curriculum differs')
        require(args['map_layout'] and args['sweeps'] and not args.get('assoc_layout', False), 'UniAD training conditions differ')
        require(args['lr'] == 1e-4 and args['warmup'] == 300 and args['wd'] == .04, 'UniAD fine-tuning optimizer differs')
        require(checkpoint['teacher_sha256'] == entry['teacher_sha256'], 'UniAD teacher differs')
        stats_entry = entry['normalization']
        stats_path = Path(weights_dir)/stats_entry['filename']
        require(file_sha256(stats_path) == stats_entry['sha256'], 'UniAD normalization checksum differs')
        stats = torch.load(stats_path, map_location='cpu', weights_only=True)
        check_metadata(stats)
        require(stats['ckpt'] == 'hf' and stats['scale'] == 1., 'UniAD feature scale/family differs')
        normalization_check(stats['mean'], stats['std'], [256]*4)
        for key in ('mean', 'std'):
            require(all(torch.equal(a, b) for a, b in zip(stats[key], checkpoint['normalization'][key])),
                    'Embedded and separate UniAD normalization differ')
        # Same learned tensors, reduced spatial grids: avoids full-resolution attention on a CPU.
        kwargs = {**checkpoint['model_kwargs'], 'level_grids': ((8, 8), (4, 4), (2, 2), (1, 1))}
        forward_model = PyramidPredictor(**kwargs).eval()
        forward_model.load_state_dict(state, strict=True)
        features = [torch.randn(1, 4, 6, 256, h, w) for h, w in kwargs['level_grids']]
        actions = torch.randn(1, 4, 3)
        layout = torch.rand(1, 4, 6, 3, 8, 8)
        forward_scope = 'CPU FP32, all released parameters, six cameras/four levels/four frames, reduced spatial grids'
    else:
        sys.path[:0] = [str(repo/'navsim'), str(repo/'navsim/core')]
        from ltf_features import load_predictor
        model, _, _, _ = load_predictor(path, 'cpu')
        require(checkpoint['feature_cut'] == 'ltf/final_fused_bev_v1', 'NAVSIM interface must be final fused BEV')
        require(model.L == model.n_cams == 1 and model.channels == (512,), 'NAVSIM BEV channels differ')
        require(model.level_grids == [(8, 8)] and model.level_pools == [1] and model.tokens_per_frame == 64,
                'NAVSIM token grid differs')
        require(model.residual and not model.warp_levels and model.action_encoder.in_features == 4,
                'NAVSIM residual/flow/action differs')
        require(model.layout_assoc is not None, 'NAVSIM six-channel associated layout is required')
        architecture_check(model)
        require(checkpoint['stage'] == 'tf' and checkpoint['step'] == 45495, 'NAVSIM checkpoint must be TF45495')
        require(checkpoint['history_frames'] == 4 and checkpoint['step_seconds'] == .5, 'NAVSIM cadence/history differs')
        require(checkpoint['prediction_mode'] == 'sliding_ar', 'NAVSIM inference must be autoregressive')
        require(checkpoint['ltf_sha256'] == entry['teacher_sha256'], 'NAVSIM teacher differs')
        config = checkpoint['config']
        require(config['world_size']*config['batch_size']*config['gradient_accumulation_steps'] == 8,
                'NAVSIM global batch differs')
        require(config['tf_lr'] == 3e-4 and config['warp_levels'] == [], 'NAVSIM TF recipe differs')
        normalization_check(checkpoint['mean'], checkpoint['std'], [512])
        forward_model = model
        features = [torch.randn(1, 4, 1, 512, 8, 8)]
        actions = torch.randn(1, 4, 4)
        actions[..., 3] = .5
        layout = torch.rand(1, 4, 1, 6, 64, 256)
        forward_scope = 'CPU FP32, native 512x8x8 BEV, four frames, released parameters'
    with torch.inference_mode():
        prediction = forward_model(features, actions, layout)
    require(all(bool(torch.isfinite(value).all()) for value in prediction), 'Nonfinite forward output')
    require(all(a.shape == b.shape for a, b in zip(prediction, features)), 'Forward output shape differs')
    parameters = sum(parameter.numel() for parameter in model.parameters())
    require(parameters == entry['parameters'], 'Parameter count differs')
    return dict(status='pass', model=model_name, filename=path.name, parameters=parameters,
                checkpoint_sha256=entry['sha256'], state_dict_sha256=entry['state_dict_sha256'],
                strict_native_loading=True, paper_architecture_and_stage=True, normalization='pass',
                metadata_scan='pass', forward_finite=True, forward_scope=forward_scope,
                limitation='No frozen driving-model or full-dataset quality/latency rerun.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('uniad', 'navsim'), required=True)
    parser.add_argument('--weights-dir', required=True, type=Path)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1]/'models/manifest.json')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = validate(args.model, args.weights_dir, json.loads(args.manifest.read_text(encoding='utf8')))
    if args.report:
        args.report.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
