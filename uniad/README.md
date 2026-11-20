# UniAD integration

This implementation predicts four scales of six-camera FPN features and injects them into a frozen UniAD runner.
The inference port uses the MMDetection 3 / MMDetection3D 1 APIs; it must not be combined with an older UniAD training environment without checking compatibility.

Prepared rollout-6000 weights and matching normalization are described in the [model card](../models/README.md).
When using them, skip predictor training/statistics estimation and follow the pretrained evaluation commands there.

## Installation

All commands below run from the repository root in Bash on Linux. Use a separate environment for this integration:

```bash
export REPO_ROOT="$(pwd)"
conda create -n jepasim-uniad python=3.10 pip -y
conda activate jepasim-uniad
```

Install PyTorch 2.x and its matching torchvision/CUDA build using the
[PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the inference stack:

```bash
python -m pip install -U openmim
mim install 'mmengine>=0.8.0,<1.0.0'
mim install 'mmcv==2.1.0'
mim install 'mmdet==3.3.0'
mim install 'mmdet3d==1.4.0'
python -m pip install 'numpy<2' 'opencv-python<4.12' nuscenes-devkit \
  casadi einops timm scipy shapely pyquaternion pytest
```

These API versions follow the [MMDetection3D compatibility table](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/docs/en/notes/faq.md).
MMCV must include CUDA operators; `mmcv-lite` is insufficient. If no wheel matches the chosen PyTorch/CUDA pair,
follow the [official MMCV build instructions](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).
This is an installation recipe, not a validated GPU lockfile. This port uses MMDetection 3 APIs rather than
the older MMDetection 2 environment used by some UniAD/FlowSim codebases.

Verify the environment before preparing data:

```bash
python -c "import torch, mmcv, mmdet, mmdet3d; from mmcv.ops import ModulatedDeformConv2dPack; assert torch.cuda.is_available(); print(torch.__version__, mmcv.__version__, mmdet.__version__, mmdet3d.__version__)"
```

Optional SplatAD and Difix adapters use separately installed upstream models and environments.

## Data preparation

Download nuScenes v1.0 trainval, including camera samples, camera sweeps, JSON tables and map expansion files,
from [nuScenes](https://www.nuscenes.org/nuscenes#download).
Obtain the temporal validation infos and motion anchors from the
[official UniAD 2.0 data guide](https://github.com/OpenDriveLab/UniAD/blob/v2.0/docs/DATA_PREP.md),
and the frozen checkpoint from [UniAD 2.0 R101](https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes).
This implementation builds its training-frame index from the nuScenes JSON tables; a separate temporal training-info pickle is not required.
If regenerating validation infos, follow the upstream preprocessing instructions, including CAN bus preparation.

Arrange the inputs as follows; the raw dataset may also live outside this repository:

```text
data/
  nuscenes/
    samples/<CAMERA>/<image>.jpg
    sweeps/<CAMERA>/<image>.jpg
    maps/expansion/*.json
    v1.0-trainval/*.json
  uniad/
    nuscenes_infos_temporal_val.pkl
    motion_anchor_infos_mode6.pkl
```

From the repository root, add the integration's code directories to the import path:

```bash
export PYTHONPATH="$PWD/uniad:$PWD/uniad/uniad_port:$PWD/uniad/feat${PYTHONPATH:+:$PYTHONPATH}"
export NUSCENES_ROOT=/path/to/nuscenes
export UNIAD_INFOS_PKL="$PWD/data/uniad/nuscenes_infos_temporal_val.pkl"
export UNIAD_BENCH="$PWD/artifacts/uniad"
export UNIAD_CKPT_HF=/path/to/UniAD2.0/uniad_base_e2e.pth
mkdir -p "$UNIAD_BENCH/feat"
```

`featsim_paths.py` defines all data/checkpoint defaults. `NUSCENES_TABLES` and `NUSCENES_MAP_ROOT`
can override metadata and map roots; both default to the standard nuScenes directory layout.
`JEPA_DEPS_DIR` can override the bundled `jepa_deps/`, but is not required.

The two UniAD checkpoint families have different numerical conventions; `uniad_port/uniad_port/compat.py`
records them and the runner selects the family from checkpoint metadata. The paper commands below use `--ckpt hf`.
Only for the legacy v1.0.1 model, set `UNIAD_CKPT_GH` and use `--ckpt gh` consistently.

The motion-anchor NPZ assets are external and are not bundled. Obtain the official anchor pickle matching
each checkpoint family, then convert it without changing coordinates:

```bash
python tools/convert_motion_anchors.py --input data/uniad/motion_anchor_infos_mode6.pkl \
  --output uniad/uniad_port/uniad_port/assets/motion_anchors_mode6_v2.0.npz
```

For v1.0.1, use its own official pickle and `motion_anchors_mode6_v1.0.1.npz`.
Do not reuse one family's anchors for the other. Weights and anchor files remain excluded from Git.

### Build indices and normalization

Compute feature normalization on the training split with the same frozen checkpoint used for evaluation:

```bash
python uniad/feat/nusc_sweeps.py
python uniad/feat/nusc_map.py --workers 8
python uniad/feat/compute_stats.py --ckpt hf --n 2000 \
  --out "$UNIAD_BENCH/feat/level_stats_1x_hf.pt"
python uniad/feat/extractor.py --selfcheck --ckpt hf
```

The index, sweep index, map cache and normalization are written below `$UNIAD_BENCH/feat/`.
The extractor self-check compares image preprocessing and all four feature levels with the frozen runner.
The normalization command illustrates training-only statistics estimation with 2,000 sampled clips;
reuse the fixed statistics matching the frozen checkpoint and evaluated predictor for a reproduction.

## Training

All three training stages below use eight GPUs and one clip per GPU. The paper's UniAD model uses
three-channel map/box layouts, residual heads at all levels, and flow heads only at P3/P4;
`--assoc_layout` is omitted. Predictor forwards use BF16; backbone extraction and feature-loss reductions use FP32.

First train the shared TF initialization on five-frame clips (four input positions):

```bash
torchrun --standalone --nproc_per_node=8 uniad/feat/train_pred.py \
  --out "$UNIAD_BENCH/train_tf_init" --ckpt hf --bs 1 --depth 8 \
  --map_layout --residual --warp_levels 0 1 \
  --rollout tf --T 5 --ctx 4 --steps 12000 --lr 3e-4 --warmup 500 --wd 0.04
```

The TF-only control receives 6,000 additional updates from that initialization, with phase-shifted camera sweeps:

```bash
torchrun --standalone --nproc_per_node=8 uniad/feat/train_pred.py \
  --out "$UNIAD_BENCH/train_tf_control" --ckpt hf --bs 1 --depth 8 \
  --map_layout --residual --warp_levels 0 1 --sweeps \
  --rollout tf --T 5 --ctx 4 --steps 6000 --lr 1e-4 --warmup 300 --wd 0.04 \
  --init "$UNIAD_BENCH/train_tf_init/iter_12000.pt"
```

The Full (rollout) model starts independently from the same TF initialization, with the same fine-tuning budget:

```bash
torchrun --standalone --nproc_per_node=8 uniad/feat/train_pred.py \
  --out "$UNIAD_BENCH/train_ro" --ckpt hf --bs 1 --depth 8 \
  --map_layout --residual --warp_levels 0 1 --sweeps \
  --rollout sliding --ctx 4 --F 8 --F_schedule 0:2,1000:4,2000:8 \
  --steps 6000 --lr 1e-4 --warmup 300 --wd 0.04 \
  --init "$UNIAD_BENCH/train_tf_init/iter_12000.pt"
```

Both fine-tuning stages reset the optimizer through `--init`; the rollout model does not initialize from the TF control.
AdamW uses betas `(0.9,0.999)`, gradient clipping at 5, linear warm-up and cosine decay to 1% of the peak.
The final rollout horizon uses 12-frame clips (`ctx + F`), with stop-gradient between sliding prediction calls.
The exported training algorithm and checkpoint tensor keys are preserved.

## Inference

After training, run one logged scene and one anchor on a single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python uniad/a_table/a_eval.py \
  --ckpt hf --scenes 0 --max-anchors 1 --ctx 4 \
  --pred "featsim=$UNIAD_BENCH/train_ro/iter_6000.pt" \
  --out "$UNIAD_BENCH/inference_smoke"
```

The script initializes real feature history and UniAD memory at each anchor, generates the following observations,
and feeds them through the frozen planner. It writes `shard0of1.pkl` with metric records and run metadata.
Use a new output directory for independent runs. This smoke run does not replace full validation.

### Predictor interface

For integration into another driver, `feat/pyramid_predictor.py::load_predictor(path, device)` returns the model
and checkpoint metadata. `sliding_rollout(model, features, actions, layout, L, F)` uses only the first `L` real
features and returns a dense TF prefix plus the remaining autoregressive predictions.

| Input | Contract |
|---|---|
| `features` | Four standardized tensors `[B,K,6,256,H_l,W_l]`; grids `116×200`, `58×100`, `29×50`, `15×25` |
| `actions` | `[B,K+F-1,3]`; relative ego `(dx,dy,dyaw)`, normalized by `ACT_MEAN/ACT_STD` in `clip_dataset.py` |
| `layout` | `[B,K+F-1,6,3,120,200]`, float in `[0,1]`; target-time box/map projection |

Condition position `i` describes the transition from feature frame `i` to `i+1`.
The paper model uses `K=4`; the first forecast is the final position of the dense TF output.
Keep recurrent predictions standardized. Apply the inverse fixed channel normalization before injecting
each generated FPN into `UniADRunner.step_frame`, and advance the runner's streaming state with generated features.
`a_table/a_eval.py::ARSource` implements this path, including shorter context at scene starts.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python uniad/a_table/a_eval.py --ckpt hf --scene-set all \
  --pred "featsim=$UNIAD_BENCH/train_ro/iter_6000.pt" \
  --out "$UNIAD_BENCH/onlog"
python uniad/a_table/a_merge.py "$UNIAD_BENCH/onlog"
```

The default command produces one shard for all validation scenes. For manual multi-GPU partitioning,
run a distinct `--shard i --nshards N` on each GPU into the same output directory, then merge after every shard completes.
`summary.json` and `summary.md` separate **geom** and **refdet** target sets; the main-paper planning comparison uses refdet.
ADE/FDE are in meters against real-image plans from the same frozen model, at observation offsets 0.5/1.5/2.5/4.0 seconds.
These offsets are not the downstream planner's waypoint horizon.

`b_table/b_eval.py` and `c_table/c_eval.py` expose off-log and closed-loop options through `--help`.
Provide predictors explicitly with `--pred name=/path/to/checkpoint.pt`; no historical run is selected implicitly.
The on-log merger summarizes evaluated arms with both geom and refdet masks, and paired differences against copy-last.
It does not require archived BEV-warp experiment files; its JSON schema differs from the historical merger.
Optional renderer backends require user-supplied reconstructions; `NEURAD_ROOT` locates neurad-studio.

For the original full-resolution CUDA checks, run `feat/test_pyramid_predictor.py`, `feat/test_pyramid_kv.py`,
and the dataset-dependent self-checks after preparing the environment and assets.
