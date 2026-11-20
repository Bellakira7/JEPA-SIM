# Changes made for the source review candidate

- Selected Python implementation and test files for UniAD and NAVSIM. Runtime products, dataset samples,
  frozen environment copies, platform clients, deployment scripts and historical result archives are excluded.
- Bundled the predictor's attention, layout and projection dependencies so the code does not import another project.
- Centralized UniAD external locations in `uniad/featsim_paths.py`; NAVSIM locations are supplied through JSON configs.
- Moved NAVSIM integration helpers to `navsim/core/` so the directory does not shadow the upstream `navsim` package.
- Removed shared cache overrides, internal prose and hardcoded experiment-checkpoint defaults.
- Replaced the UniAD dependency's MMCV DropPath import with the timm implementation also used by the NAVSIM dependency.
  This removes the MMCV requirement from predictor-only checks; the frozen UniAD runner still requires MMCV CUDA ops.
- Replaced the incomplete on-log merger with a standalone merger for evaluated arms and a tested scene-cluster bootstrap.
  It retains geom/refdet masks and paired differences against copy-last. It no longer imports an unavailable
  historical merge helper or injects archived BEV-warp rows. Its JSON summary schema is new.
- Reused the official nuScenes SDK split definitions instead of shipping a separate split JSON or SDK source copy.
- Added a config generator, external motion-anchor conversion, path-safe source checks and setup instructions.
- Aligned the default NAVSIM recipe with the paper: final fused BEV, TF-only training, explicit BEV-cache conversion,
  and autoregressive evaluation of the TF checkpoint with interface/stage/update guards.
- Aligned the UniAD training examples with the paper's three-channel layouts, residual/flow heads,
  eight-GPU batch and separate TF initialization, TF control and rollout fine-tuning schedules.
- Added model cards and an integrity manifest for the two main-accuracy predictor checkpoints.
  Separate inference-weight copies omit optimizer state and private metadata while retaining exact model tensors.
- Added strict native loading, normalization, architecture/stage and CPU forward checks for those weights.

Model forward/training math and checkpoint tensor names are otherwise retained.
No project-wide open-source license has been selected. Full CUDA environment locking and dataset-based reproduction
are follow-up work; passing CPU tests does not establish reproduction of paper results.
