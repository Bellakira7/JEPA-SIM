# NAVSIM / OpenScene / frozen LTF

The default paper recipe predicts LTF's final fused BEV, `[B,512,8,8]`, with four real history frames
and 0.5-second transitions. It trains with teacher forcing only and evaluates that TF checkpoint autoregressively.
The model has a residual output head and no flow head; target image encoding and camera fusion are bypassed at generated steps.
Camera-stage prediction and rollout fine-tuning remain available as separate implementation variants.

Prepared final-BEV TF45495 weights are described in the [model card](../models/README.md).
For pretrained inference, complete installation/data setup and configuration, then use those weights without the cache/training stages.

## Installation

Use the [FlowSim repository](https://github.com/hanfeng-wu/FlowSim) at commit
`98f794c9055da0b55055e70e234d7785c0d2b88f`, with its NAVSIM/nuPlan environment.
The integration imports `navsim` and `scripts.domain_gap` from that checkout. Run from this repository's root:

```bash
export REPO_ROOT="$(pwd)"
mkdir -p third_party
git clone https://github.com/hanfeng-wu/FlowSim.git third_party/FlowSim
git -C third_party/FlowSim checkout --detach 98f794c9055da0b55055e70e234d7785c0d2b88f
export FLOWSIM_ROOT="$REPO_ROOT/third_party/FlowSim/flowsim_navsim"

conda create -n jepasim-navsim python=3.10 pip -y
conda activate jepasim-navsim
python -m pip install -r "$FLOWSIM_ROOT/requirements.txt"
python -m pip install -e "$FLOWSIM_ROOT"
python -m pip install einops

export NAVSIM_DEVKIT_ROOT="$FLOWSIM_ROOT"
export PYTHONPATH="$REPO_ROOT/navsim:$REPO_ROOT/navsim/core:$FLOWSIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -c "import torch, navsim, nuplan; from navsim.common.dataloader import SceneLoader; assert torch.cuda.is_available(); print(torch.__version__)"
```

Python 3.10 supports the type annotations in this repository while retaining compatibility with the locked
upstream dependency pins. Follow the [locked FlowSim setup guide](https://github.com/hanfeng-wu/FlowSim/blob/98f794c9055da0b55055e70e234d7785c0d2b88f/README.md)
and [NAVSIM installation notes](https://github.com/hanfeng-wu/FlowSim/blob/98f794c9055da0b55055e70e234d7785c0d2b88f/flowsim_navsim/docs/install.md)
for system dependencies. Keep the upstream PyTorch, torchvision and NumPy pins together.
This environment already supplies OpenCV; `requirements-core.txt` is intended for a separate CPU-test environment.

## Data preparation

Obtain OpenScene trainval metadata, camera images and nuPlan maps, the official camera-only
LTF `ltf_seed_0.ckpt`, and FlowSim VH `flowsim_transfuser_VH.ckpt` from their original providers.
Relevant dataset/model revision identifiers are recorded in `core/protocol.py`.
The code references [official NAVSIM weights](https://huggingface.co/autonomousvision/navsim_baselines)
and [FlowSim weights](https://huggingface.co/Hanfeng0419/FlowSim).

Expected dataset layout:

```text
dataset/
  navsim_logs/trainval/<log>.pkl
  sensor_blobs/trainval/<log>/<camera>/<image>
  maps/
```

Download and extract logs, sensor blobs and maps following the
[locked upstream data instructions](https://github.com/hanfeng-wu/FlowSim/blob/98f794c9055da0b55055e70e234d7785c0d2b88f/flowsim_navsim/docs/install.md).
Use trainval data with camera images from `CAM_L0`, `CAM_F0` and `CAM_R0` for all four history frames.
Set the dataset and checkpoint locations:

```bash
export OPENSCENE_DATA_ROOT=/path/to/openscene
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NAVSIM_EXP_ROOT="$REPO_ROOT/artifacts/navsim_runtime"
export LTF_CKPT=/path/to/ltf_seed_0.ckpt
export FLOWSIM_CKPT=/path/to/flowsim_transfuser_VH.ckpt
export SCENE_FILTERS="$FLOWSIM_ROOT/navsim/planning/script/config/common/train_test_split/scene_filter"
```

Use `navtrain.yaml` and `navtest.yaml` from that scene-filter directory; both use four initial history frames.
`configure.py` freezes the selected tokens and rejects overlapping train/test logs.
Dataset availability and filter choice determine the actual sample coverage; inspect the generated token and sequence audits.

## Training

Run from the repository root. Use the official navtrain and navtest filter YAMLs from the locked FlowSim checkout.
The config generator freezes selections from your actual metadata and rejects overlapping train/test logs.
Use an empty output directory and absolute paths for external inputs:

```bash
python navsim/configure.py \
  --flowsim-root "$FLOWSIM_ROOT" --dataset-root "$OPENSCENE_DATA_ROOT" \
  --train-filter "$SCENE_FILTERS/navtrain.yaml" --test-filter "$SCENE_FILTERS/navtest.yaml" \
  --ltf-checkpoint "$LTF_CKPT" --flowsim-checkpoint "$FLOWSIM_CKPT" \
  --gpus 8 --out artifacts/navsim
python navsim/parallel_pipeline.py prepare --config artifacts/navsim/cache.json \
  --out artifacts/navsim/cache_run --gpus 8
python navsim/sequence_data.py --config artifacts/navsim/camera_sequence.json \
  --out artifacts/navsim/camera_sequence
torchrun --standalone --nproc_per_node=8 navsim/sequence_data.py \
  --config artifacts/navsim/camera_sequence.json --out artifacts/navsim/camera_sequence --phase conditions
python navsim/sequence_data.py --config artifacts/navsim/camera_sequence.json \
  --out artifacts/navsim/camera_sequence --phase verify-conditions
torchrun --standalone --nproc_per_node=8 navsim/prepare_bev_cache.py convert \
  --config artifacts/navsim/sequence.json --out artifacts/navsim/sequence
python navsim/prepare_bev_cache.py merge --config artifacts/navsim/sequence.json \
  --out artifacts/navsim/sequence
torchrun --standalone --nproc_per_node=8 navsim/train_sequence.py \
  --config artifacts/navsim/sequence.json --out artifacts/navsim/sequence
```

Camera-stage caches feed the frozen fusion blocks to produce real-image BEV targets. The BEV cache reuses
the symbolic conditions and computes training-frame normalization. `training_stages: ["tf"]` selects five-frame
windows and TF predictor training.

The paper uses eight GPUs with one clip per GPU, 96,031 five-frame windows, 45,495 TF updates,
peak LR `3e-4`, 1,896 warm-up updates, cosine decay to 1% of the peak, weight decay `.04`,
gradient clipping at 1 and FP32 training. The existing exposure formula determines the update count from
the selected windows and global batch. For a custom run, set the expected checkpoint step in the evaluation
config to the completed update count.

Each run records cache completion markers, training normalization, source hashes and deterministic sampler state.
Use a separate output directory for each run.

## Inference

After training, resolve the generated evaluation config for a one-target inference run on one GPU:

```bash
python - <<'PY'
import json
from pathlib import Path
from eval_sequence import evaluation_config

root = Path('artifacts/navsim')
config = evaluation_config(json.loads((root/'evaluation.json').read_text()), root/'sequence')
config['limit_tokens'] = 1
(root/'inference_smoke.json').write_text(json.dumps(config, indent=2)+'\n')
PY
CUDA_VISIBLE_DEVICES=0 python navsim/core/eval_open_loop.py \
  --config artifacts/navsim/inference_smoke.json --out artifacts/navsim/inference_smoke --preflight
CUDA_VISIBLE_DEVICES=0 python navsim/core/eval_open_loop.py \
  --config artifacts/navsim/inference_smoke.json --out artifacts/navsim/inference_smoke
```

The preflight checks external modules, inputs and the locked upstream revision. The actual run also verifies
the real-image/frozen-interface identity and predictor checkpoint metadata before writing metrics.
The smoke config evaluates the same configured arms, including FlowSim VH, on one target.
It writes `tokens.json`, `records.json`, `identity_checks.json`, `stats.json` and `results.md` below the smoke directory.
Use a new directory for another inference attempt.

### Predictor interface

`core/ltf_features.py::load_predictor(path, device)` loads the final-BEV checkpoint together with its
training-channel mean/std. `train_sequence.py::forecast` performs causal sliding prediction from exactly four real frames.

| Input | Contract for the final-BEV model |
|---|---|
| Feature history | A one-element list containing `[B,4,1,512,8,8]`, standardized with the checkpoint mean/std |
| Actions | `[B,4+F-1,4]`, raw `(dx,dy,dyaw,dt)`; `dt=0.5` seconds |
| Layouts | `[B,4+F-1,1,6,64,256]`, float in `[0,1]`; target geometry plus source boxes projected into target views |

The model pools layout conditioning to its 8×8 grid. Its dense output covers four next-frame TF positions;
the remaining output covers AR steps 2 through `F`. Keep predictions standardized in history, inverse-scale
the selected forecast and pass the native `[B,512,8,8]` tensor to the frozen task heads.
`core/bev_features.py::predict_bev` implements the image-history, conditioning and scale-conversion path.
Only predictor history recurs; this LTF planner has no streaming temporal memory.

## Evaluation

```bash
python navsim/core/eval_sequence.py plan --config artifacts/navsim/evaluation.json \
  --out artifacts/navsim/sequence
torchrun --standalone --nproc_per_node=8 navsim/core/eval_sequence.py worker \
  --config artifacts/navsim/evaluation.json --out artifacts/navsim/sequence
python navsim/core/eval_sequence.py merge --config artifacts/navsim/evaluation.json \
  --out artifacts/navsim/sequence
```

The generated evaluation config selects `sequence/tf/featsim_ltf.pt` and checks the final-BEV interface,
TF stage, update 45,495 and residual output head. Evaluation advances predictor history autoregressively.

The `plan` phase fixes the common target list and writes per-worker requests. Launch the same worker count
as configured in `world_size`; the `merge` phase rejects incomplete or inconsistent shards.
Final outputs are in `artifacts/navsim/sequence/open_loop/`. `stats.json` includes both `per_token_mean`
and `per_log_macro`; the paper's planning comparison uses the target-averaged `per_token_mean`.
`results.md` shows the readable planning summary, and `records.json` retains individual metrics.

Use `timed_eval_sequence.py` with the same phases in a separate output directory for latency instrumentation.
The reference-image, copy-last, predicted-feature and FlowSim arms share a checked token set.
The evaluator distinguishes agreement with frozen LTF outputs from error against logged human trajectories.
This workflow reports open-loop planning consistency.

Optional reconstruction adapters in `core/gs_vendor/` target SplatAD, OmniRe without SMPL, and Street Gaussians.
They require separately installed upstream projects, weights and reconstruction configs.
`run_reconstruction.py` and `preflight_reconstruction.py` accept an optional `nuplan_devkit_root` config key;
otherwise the installed SDK is used. `configs/mini_smoke.yaml` runs a small interface check.

Run OpenScene training, evaluation and renderer identity checks in the Linux/CUDA environment described above.
