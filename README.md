# JEPA-SIM: Render-Free Closed-Loop Driving Simulation

JEPA-SIM predicts the visual features consumed by a frozen driving model, conditioned on ego motion
and camera-space scene layouts. Generated features replace image-encoder observations and become
the context for subsequent predictions.

This repository contains the **UniAD / nuScenes** and **camera-only Latent TransFuser / OpenScene**
implementations. The documented workflows cover data preparation, predictor training, logged-scene
inference and planning-consistency evaluation. The frozen driving model is not retrained.

| Integration | Predicted interface | Training | Inference |
|---|---|---|---|
| UniAD / nuScenes | Six-camera, four-level FPN before BEVFormer | TF initialization, then rollout fine-tuning | Autoregressive features with streaming UniAD state |
| NAVSIM / OpenScene | Final fused LTF BEV, `[B,512,8,8]` | Teacher forcing only | Autoregressive use of the TF checkpoint |

**Guides:** [UniAD](uniad/README.md) · [NAVSIM](navsim/README.md) · [Pretrained models](models/README.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

## Contents

- [Installation](#installation)
- [Pretrained models](#pretrained-models)
- [Data and checkpoints](#data-and-checkpoints)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Tests](#tests)
- [Repository structure](#repository-structure)
- [License and acknowledgements](#license-and-acknowledgements)

## Installation

Use **Linux, Bash, Python 3.10+ and a CUDA-capable GPU** for the benchmark workflows.
The paper training recipes use eight GPUs with one clip per GPU. Logged-scene inference can run on one GPU.
Create separate environments for UniAD and NAVSIM because their upstream dependency stacks differ.

After cloning or unpacking the repository, run commands from its root:

```bash
export REPO_ROOT="$(pwd)"
```

Choose an integration and complete its installation and import checks:

| Integration | Environment instructions |
|---|---|
| UniAD | [Install the MMDetection 3 / MMDetection3D 1 inference stack](uniad/README.md#installation) |
| NAVSIM | [Install the locked FlowSim checkout and its NAVSIM dependencies](navsim/README.md#installation) |

`requirements-core.txt` is for standalone predictor tests. It does not install the complete driving stacks.
In particular, the UniAD runner requires MMCV CUDA operators, and the NAVSIM workflow imports FlowSim's
loader, frozen LTF and evaluation helpers. GPU installation depends on the PyTorch/CUDA combination;
a fully validated CUDA environment lockfile is not included yet.

## Pretrained models

The two main accuracy predictors are prepared in a separate model bundle:

| Predictor | Stage | Parameter count |
|---|---|---:|
| UniAD / nuScenes | Rollout fine-tuning, update 6,000 | 35,938,784 |
| NAVSIM / OpenScene | Final-BEV TF, update 45,495; AR inference | 19,426,560 |

Download the weights from [GitHub Release v1.0.0](https://github.com/Bellakira7/JEPA-SIM/releases/tag/v1.0.0).
See the [model card and usage instructions](models/README.md) for checksums, normalization, reference metrics
and pretrained evaluation commands. Dataset preparation and frozen-host setup are still required,
but predictor training can be skipped when using the prepared weights.

## Data and checkpoints

Datasets and frozen driving-model checkpoints are obtained separately from their providers.
JEPA-SIM predictor weights are distributed separately as described above; data and precomputed features are not bundled.

| Integration | Required inputs | Preparation guide |
|---|---|---|
| UniAD | nuScenes trainval images, sweeps, JSON metadata and maps; temporal validation infos; UniAD 2.0 checkpoint and matching motion anchors | [nuScenes preparation](uniad/README.md#data-preparation) |
| NAVSIM | OpenScene trainval logs and front-camera images; nuPlan maps; LTF seed-0 and FlowSim VH checkpoints; upstream train/test filters | [OpenScene preparation](navsim/README.md#data-preparation) |

Keep external data and generated artifacts in separate directories, for example:

```text
JEPA-SIM/
├── data/
│   ├── nuscenes/                 # samples/, sweeps/, maps/, v1.0-trainval/
│   ├── uniad/                    # temporal infos and downloaded motion-anchor pickle
│   └── openscene/                # maps/, navsim_logs/, sensor_blobs/
├── artifacts/
│   ├── uniad/                    # normalization, training checkpoints, evaluation outputs
│   └── navsim/                   # generated configs, caches, checkpoints, evaluation outputs
└── third_party/
    └── FlowSim/                  # locked external runtime for NAVSIM
```

The guides set the environment variables and generate the caches expected by each script.
Use training-only statistics matching the frozen model and predictor being evaluated.

## Training

### UniAD

Follow the [three-stage training commands](uniad/README.md#training).
The paper configuration uses three-channel map/box layouts, residual outputs at all four FPN levels,
and flow heads at P3/P4. Both fine-tuning branches start independently from the same TF initialization.

| Stage | Updates | Peak LR | Warm-up | Output under `$UNIAD_BENCH` |
|---|---:|---:|---:|---|
| TF initialization | 12,000 | `3e-4` | 500 | `train_tf_init/iter_12000.pt` |
| TF-only control | 6,000 | `1e-4` | 300 | `train_tf_control/iter_6000.pt` |
| Full rollout model | 6,000 | `1e-4` | 300 | `train_ro/iter_6000.pt` |

Fine-tuning uses camera sweeps. Rollout training uses four context frames and the horizon curriculum
`0:2,1000:4,2000:8`, ending with 12-frame clips. The TF-only control is the training ablation;
use the Full rollout checkpoint for the main UniAD workflow.

### NAVSIM

Follow the [configuration, caching and training commands](navsim/README.md#training) in order:

1. Generate configs with `navsim/configure.py`.
2. Extract real-image camera-stage caches with `navsim/parallel_pipeline.py prepare`.
3. Build the TF sequence index and symbolic conditions with `navsim/sequence_data.py`.
4. Replay frozen fusion into final BEV targets with `navsim/prepare_bev_cache.py convert`, then `merge`.
5. Train the BEV predictor with `navsim/train_sequence.py`.

The generated config selects `training_stages: ["tf"]`, no flow heads and eight GPUs.
Camera-stage caching is preparation for the BEV targets; it does not train a camera-feature predictor.
With the paper's 96,031 five-frame windows and global batch 8, training takes 45,495 updates
with LR `3e-4`, 1,896 warm-up updates and FP32 computation.

The resulting checkpoint is:

```text
artifacts/navsim/sequence/tf/featsim_ltf.pt
```

Changing the selected data or global batch changes the exposure-derived update count. The generated
evaluation config checks the paper's interface, TF stage and update 45,495; explicitly adjust that guard
when running an adapted recipe.

## Inference

Set the integration-specific environment variables from its guide and use a trained predictor checkpoint.
Real images initialize the context; subsequent feature observations are generated autoregressively.
Motion and scene-layout conditions follow the log in these public-benchmark workflows.

For a small UniAD run:

```bash
CUDA_VISIBLE_DEVICES=0 python uniad/a_table/a_eval.py \
  --ckpt hf --scenes 0 --max-anchors 1 --ctx 4 \
  --pred "featsim=$UNIAD_BENCH/train_ro/iter_6000.pt" \
  --out "$UNIAD_BENCH/inference_smoke"
```

This exercises predictor inference and frozen UniAD planning on one logged scene and saves an evaluation shard.
It is a small correctness run, not the full validation result.
For NAVSIM, the [one-target inference instructions](navsim/README.md#inference) resolve the generated config
and run `navsim/core/eval_open_loop.py` on one target with the TF checkpoint.

For tensor-level integration, see the [UniAD predictor interface](uniad/README.md#predictor-interface)
and [NAVSIM predictor interface](navsim/README.md#predictor-interface). They document feature shapes,
condition alignment and normalization. The scripts generate features for a driving model; they do not
produce rendered future RGB images.

## Evaluation

### UniAD: full validation

After completing the [UniAD environment and preparation](uniad/README.md):

```bash
CUDA_VISIBLE_DEVICES=0 python uniad/a_table/a_eval.py --ckpt hf --scene-set all \
  --pred "featsim=$UNIAD_BENCH/train_ro/iter_6000.pt" \
  --out "$UNIAD_BENCH/onlog"
python uniad/a_table/a_merge.py "$UNIAD_BENCH/onlog"
```

Outputs are `shard0of1.pkl`, `summary.json` and `summary.md` in the run directory.
Use the **refdet** summary for the main planning protocol: valid common target geometry with non-empty
real-image reference detections. The **geom** summary retains targets without that detection filter.
ADE/FDE compare generated-observation plans with real-image plans from the same frozen UniAD model.
Each observation offset is separate from the planner's waypoint horizon.

### NAVSIM: plan, run workers, merge

After training with the default generated configs:

```bash
python navsim/core/eval_sequence.py plan --config artifacts/navsim/evaluation.json \
  --out artifacts/navsim/sequence
torchrun --standalone --nproc_per_node=8 navsim/core/eval_sequence.py worker \
  --config artifacts/navsim/evaluation.json --out artifacts/navsim/sequence
python navsim/core/eval_sequence.py merge --config artifacts/navsim/evaluation.json \
  --out artifacts/navsim/sequence
```

Outputs are written to `artifacts/navsim/sequence/open_loop/`:

| File | Contents |
|---|---|
| `tokens.json` | Evaluated token list |
| `records.json` | Per-target, per-arm, per-offset metrics |
| `identity_checks.json` | Real-image vs frozen-interface replay checks |
| `stats.json` | Paired summaries, checkpoint hashes and run config |
| `results.md` | Readable planning summary |

Use `per_token_mean` planning agreement for the paper's target-averaged comparison.
`per_log_macro` uses equal log weights and is reported separately.
Human-trajectory errors under `gt/*` are different metrics from real-image planning agreement.
This NAVSIM path is an open-loop evaluation and does not compute PDM/EPDMS.

Use fresh output directories for independent runs. Keep reference checkpoints, normalization, target selection
and inference settings fixed when comparing arms. Optional renderer and public closed-loop adapters are described
in the benchmark guides; their upstream models and assets are installed separately.

## Tests

For dataset-free CPU checks, create a separate Python environment, install a matching
[PyTorch/torchvision pair](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -r requirements-core.txt
python tools/check_release.py
python -m pytest navsim/core/test_protocol.py -q
python -m pytest navsim/test_sequence.py -q
python -m pytest navsim/test_paper_recipes.py -q
python -m pytest navsim/test_parallel_helpers.py -q
python -m pytest tools/test_model_checks.py -q
PYTHONPATH="$PWD/uniad" python -m pytest uniad/feat/test_predictor_cpu.py -q
python -m pytest uniad/a_table/test_aggregation.py uniad/a_table/test_merge.py -q
python uniad/agreement/test_agreement.py
```

Run suites separately because the integrations retain some identical module names.
CPU tests cover synthetic inputs; they do not establish full-dataset CUDA reproduction.
The guides include the real-data interface checks to run after installation and asset preparation.

## Repository structure

```text
uniad/
  feat/                 # predictor, training, statistics and nuScenes data interfaces
  jepa_deps/            # attention and layout-conditioning modules
  uniad_port/           # frozen UniAD inference and streaming state
  a_table/              # on-log autoregressive evaluation and aggregation
  b_table/, offlog/     # off-log geometry, conditions and evaluation
  c_table/              # public closed-loop controller and evaluation
  agreement/            # planning/perception agreement metrics
navsim/
  configure.py          # default final-BEV / TF recipe
  sequence_data.py      # sequence index and symbolic conditions
  prepare_bev_cache.py  # frozen fusion replay and BEV normalization
  train_sequence.py     # TF training and optional rollout fine-tuning
  core/                 # frozen LTF interfaces, predictor and evaluation
models/                 # pretrained predictor model card and integrity manifest
tools/                  # source/model checks and motion-anchor conversion
```

## License and acknowledgements

This implementation builds on UniAD, V-JEPA 2, NAVSIM, FlowSim and their dependencies.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source relationships and separate data/model terms.
A project-wide license has not yet been selected.
