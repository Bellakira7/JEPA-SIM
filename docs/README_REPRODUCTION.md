# README verification on DSW — 2026-11-23

The public inference pipelines produced finite real-data planning-consistency results in a
prepared Linux/PPU environment. This checks that the commands, released predictors and frozen
hosts work together; it does **not** reproduce the full paper tables or timings.
The evaluated local documentation snapshot was `fc2669481b687ada0a9231f867fcc7235759b013`;
predictor, frozen-host and evaluation Python sources match the public upstream base
`1035269bccec4395a0a0753d39b3b77a9579a84e` and are unchanged by this PR.
See the [machine-readable readout](README_REPRODUCTION_20261123.json) for all smoke planning rows,
test counts, model hashes, identity checks and aggregation labels.

## What passed

- All documented CPU suites: **44 pytest cases and 23 agreement checks**.
- Both v1.0.0 released predictors: file/tensor hashes, configuration, normalization,
  strict loading and finite CPU forwards. UniAD uses reduced spatial grids for this model-only check;
  NAVSIM uses its native 512×8×8 BEV.
- Both frozen-host file hashes match the released predictors' `teacher_sha256` values.
- UniAD: nuScenes validation **scene-0003**, all **33 legal anchors**, identity/copy-last/JEPA arms,
  **26 common refdet targets per offset** at 0.5/1.5/2.5/4.0 s. Image/reference inference,
  autoregressive feature injection and `a_merge.py` completed. Identity Plan ADE is zero.
- NAVSIM: official navtrain/navtest filters on the available **seven-log OpenScene mini split**,
  then **one navtest target in one log**. Identity/copy-last-BEV/JEPA/FlowSim-VH arms completed
  at 0.5/1.0/1.5 s under **OpenScene-LTF-AR-v2 / `per_token_mean`**. Frozen-interface identity
  differences are zero with tolerance `1e-6`; all planning rows are finite.
- The final release source check accepts valid paper PNGs in `docs/assets/` and retains its
  source syntax and private-reference checks.

## Commands exercised

Use the integration guides to set dataset, frozen-host and import paths. With the model bundle
verified, copy the released UniAD normalization into a fresh `$UNIAD_BENCH/feat/` directory and run:

```bash
CUDA_VISIBLE_DEVICES=0 python uniad/a_table/a_eval.py --ckpt hf --scenes 0 --ctx 4 \
  --pred "jepasim=$PWD/artifacts/models/weights/jepasim_uniad_rollout_6000.pt" \
  --pred-opt jepasim=dtype:bf16,kv:0 --out "$UNIAD_BENCH/inference_smoke"
python uniad/a_table/a_merge.py "$UNIAD_BENCH/inference_smoke"
```

For NAVSIM, run `configure.py` with `--data-split mini --gpus 1` and the official filters,
then apply the released-checkpoint settings from the model card. Resolve the evaluation config,
set `limit_tokens=1`, and execute both `eval_open_loop.py --preflight` and the actual evaluation
as documented in the [NAVSIM inference guide](../navsim/README.md#inference).
The locked FlowSim source revision was verified through an archive file-hash manifest.

## Environment and repairs

The host exposes eight PPU-ZW810E accelerators; smoke inference used one at a time. Python was
3.12.3, PyTorch/torchvision 2.9.0/0.24.0, NumPy 1.26.4, timm 1.0.27, MMDetection 3.3.0,
and vendor MMCV 2.2.0 / MMDetection3D 1.4.0 builds. NAVSIM used Lightning 2.2.1,
Hydra 1.3.2 and the existing nuPlan SDK source. These differ from the NVIDIA installation recipes.

Initial attempts exposed missing `pyquaternion`/`casadi`, geospatial dependencies and
`pkg_resources` in the prepared runtime. The successful attempts supplied those dependencies
in integration environments and initialized the installed vendor SDK before inference.
On PPU, source the vendor's `envsetup.sh` in the execution shell; otherwise runtime compilation
can abort with “Both PPU_SDK and PPU_HOME are not exist” even after strict model loading succeeds.
No shared host packages or other jobs were modified.

The original UniAD `--max-anchors 1` startup example yielded only the 4.0 s offset for its first
anchor. The guide now uses a complete scene so the merger has common targets at every offset.
Keep the single-anchor option for startup checks, and use fresh output directories for retries.

## Evidence boundary

No predictor training, renderer reconstruction or queued platform jobs were launched. The full
150-scene nuScenes comparison, full OpenScene navtest comparison, proprietary in-house execution
benchmark and paper observation-generation timings remain unverified by this smoke run.
Smoke scores are kept separate from the paper results in the README. With only one scene/log,
bootstrap intervals do not establish significance, and no performance or speed ranking is inferred.
