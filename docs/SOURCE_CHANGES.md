# Implementation notes

- UniAD predicts six-camera, four-level FPN features and advances the frozen UniAD streaming state.
- NAVSIM predicts the final fused BEV and evaluates its TF checkpoint autoregressively.
- Predictor attention, layout conditioning and projection modules are bundled with each integration.
- UniAD data and checkpoint locations are configured through `uniad/featsim_paths.py`.
- NAVSIM data, cache and checkpoint locations are configured through JSON files.
- NAVSIM helpers live in `navsim/core/`; the upstream NAVSIM runtime is installed through FlowSim.
- Predictor modules use timm DropPath; frozen UniAD inference uses MMCV CUDA operators.
- The UniAD on-log merger reports geom/refdet summaries, scene-cluster bootstrap intervals and paired differences against copy-last.
- Dataset splits come from the official nuScenes SDK.
- Training guides specify the three UniAD stages and the final-BEV TF recipe for NAVSIM.
- Model cards document feature interfaces, normalization, inference commands and planning results.
- Model-check tools verify checkpoint hashes, configuration, normalization, strict loading and CPU forwards.
