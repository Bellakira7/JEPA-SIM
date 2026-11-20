"""Small six-camera checks of the exported UniAD predictor and bundled dependencies."""
import torch
from pyramid_predictor import PyramidPredictor, sliding_rollout


def sample():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    grids, pools = ((8, 8), (4, 4), (2, 2), (1, 1)), (8, 4, 2, 1)
    model = PyramidPredictor(level_grids=grids, level_pools=pools, n_cams=6, in_chans=2,
                             embed_dim=48, depth=2, num_heads=2, residual=True,
                             warp_levels=(0, 1), assoc_layout=True)
    features = [torch.randn(1, 5, 6, 2, *grid) for grid in grids]
    actions = torch.randn(1, 5, 3)
    layout = torch.rand(1, 5, 6, 6, 8, 8)
    return model, features, actions, layout


def test_six_camera_zero_residual_and_gradients():
    model, features, actions, layout = sample()
    result = model(features, actions, layout)
    for actual, expected in zip(result, features):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    sum(value.square().mean() for value in result).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.proj[0].weight.grad.abs().sum() > 0


def test_causality_and_kv_prefix_equivalence():
    model, features, actions, layout = sample()
    with torch.no_grad():
        for projection in model.proj:
            projection.weight.normal_(std=.02)
        dense = model(features, actions, layout)
        changed = [value.clone() for value in features]
        for value in changed:
            value[:, 3:] += 20
        later = model(changed, actions, layout)
        for first, second in zip(dense, later):
            torch.testing.assert_close(first[:, :3], second[:, :3], atol=1e-6, rtol=1e-6)
        cache, steps = None, []
        for index in range(5):
            result, cache = model.forward_step([f[:, index:index+1] for f in features],
                actions[:, index:index+1], layout[:, index:index+1], pos=index, kv_cache=cache)
            steps.append(result)
        for level, full in enumerate(dense):
            streamed = torch.cat([step[level] for step in steps], dim=1)
            torch.testing.assert_close(full, streamed, atol=1e-5, rtol=1e-5)


def test_sliding_rollout_shapes():
    model, features, actions, layout = sample()
    dense, rollout = sliding_rollout(model, [value[:, :4] for value in features], actions, layout, 4, 2)
    assert all(value.shape[1] == 4 for value in dense)
    assert all(value.shape[1] == 1 for value in rollout)
