# Pretrained JEPA-SIM predictors

Pretrained predictors for **UniAD / nuScenes** and **NAVSIM / OpenScene**.

| Model | Checkpoint | Parameters | Paper interface and stage |
|---|---|---:|---|
| UniAD / nuScenes | `jepasim_uniad_rollout_6000.pt` | 35,938,784 | Six-camera P3–P6 FPN; rollout fine-tuning update 6,000 |
| NAVSIM / OpenScene | `jepasim_navsim_bev_tf_45495.pt` | 19,426,560 | Final fused 512×8×8 BEV; TF update 45,495, AR inference |

The UniAD bundle also contains `uniad_level_stats_1x_hf.pt`, the matching fixed four-level normalization.
NAVSIM normalization is embedded in its checkpoint. Both predictors store FP32 tensors; the paper's
UniAD accuracy path uses BF16 autocast with dense sliding inference, while the NAVSIM path uses FP32.
Use the frozen host checkpoint matching `teacher_sha256` in the manifest; the host/predictor pairing
and matching normalization are part of the paper protocol.

Download both predictors and their normalization from [GitHub Release v1.0.0](https://github.com/Bellakira7/JEPA-SIM/releases/tag/v1.0.0).
The [manifest](manifest.json) records individual download URLs, file hashes and model-state hashes.
The source repository and model assets are distributed separately.

## Prepare the model bundle

From the source repository root, download the model bundle and unpack it into an ignored artifact directory:

```bash
mkdir -p artifacts/models
curl -fL https://github.com/Bellakira7/JEPA-SIM/releases/download/v1.0.0/jepasim-models-v1.0.0.zip -o artifacts/jepasim-models-v1.0.0.zip
unzip artifacts/jepasim-models-v1.0.0.zip -d artifacts/models
```

The extracted files are:

```text
artifacts/models/
  README.md
  manifest.json
  CHECKSUMS.sha256
  weights/
    jepasim_uniad_rollout_6000.pt
    jepasim_navsim_bev_tf_45495.pt
    uniad_level_stats_1x_hf.pt
```

After installing `requirements-core.txt` in a predictor-check environment, verify each model separately:

```bash
python tools/check_models.py --model uniad --weights-dir artifacts/models/weights
python tools/check_models.py --model navsim --weights-dir artifacts/models/weights
```

The checker verifies file and tensor hashes, metadata, normalization, paper architecture/stage and strict loading
with this source tree. It runs CPU forwards on native NAVSIM BEV grids and reduced UniAD spatial grids
using all released parameters.

## Use the UniAD predictor

Complete `uniad/README.md` installation, nuScenes setup, motion-anchor conversion and frozen UniAD 2.0 setup.
Predictor training and statistics estimation can be skipped when using the prepared model and its bundled statistics.
Use a separate artifact root for this checkpoint's normalization:

```bash
export UNIAD_BENCH="$PWD/artifacts/uniad_pretrained"
mkdir -p "$UNIAD_BENCH/feat"
cp artifacts/models/weights/uniad_level_stats_1x_hf.pt "$UNIAD_BENCH/feat/level_stats_1x_hf.pt"

CUDA_VISIBLE_DEVICES=0 python uniad/a_table/a_eval.py --ckpt hf --scene-set all --ctx 4 \
  --pred "jepasim=$PWD/artifacts/models/weights/jepasim_uniad_rollout_6000.pt" \
  --pred-opt jepasim=dtype:bf16,kv:0 --out "$UNIAD_BENCH/onlog"
python uniad/a_table/a_merge.py "$UNIAD_BENCH/onlog"
```

Architecture: eight blocks, width 384, six heads, GELU MLP ratio 4, AdaLN with three-dimensional relative motion,
three-channel box/map layouts, residual prediction at all levels and flow heads only at P3/P4.
Context is four frames at 0.5-second cadence; the fine-tuning horizon curriculum is `0:2,1000:4,2000:8`.
Apply the bundled inverse normalization before frozen UniAD; recurrent predictor history remains standardized.

## Use the NAVSIM predictor

Complete `navsim/README.md` installation and data/checkpoint setup, and run `configure.py` to generate
`artifacts/navsim/evaluation.json`. Select the evaluation worker count with `--gpus`.
Point the evaluation config to the released file and use its new file checksum:

```bash
python - <<'PY'
import json
from pathlib import Path

entry = json.loads(Path('models/manifest.json').read_text())['models']['navsim']
path = Path('artifacts/navsim/evaluation.json')
config = json.loads(path.read_text())
config.update(
    featsim_checkpoint=str((Path('artifacts/models/weights')/entry['filename']).resolve()),
    expected_checkpoint_sha256=entry['sha256'],
    expected_feature_cut='ltf/final_fused_bev_v1',
    expected_checkpoint_stage='tf', expected_checkpoint_step=45495, expected_warp_levels=[])
path.write_text(json.dumps(config, indent=2)+'\n')
PY
```

Then run the one-target inference or plan/worker/merge evaluation commands in `navsim/README.md`.
Use a worker count matching the generated config. The model has one 512×8×8 BEV stream, eight blocks,
width 384, six heads, six-channel associated camera-layout conditioning, four-dimensional motion including
`dt=0.5`, and a residual output head with no flow. The TF checkpoint is evaluated autoregressively.

## Reference planning results

Planning results reported in the paper:

UniAD uses nuScenes validation, 150 scenes, on-log AR with frozen HF UniAD 2.0, and target-frame means
over 3,878 common refdet targets at each offset. NAVSIM uses OpenScene navtest with four valid history
frames, OpenScene-LTF-AR-v2, 11,774 targets in 131 logs, and `per_token_mean` aggregation.

| Model | Observation offset (s) | Plan ADE (m) | Plan FDE (m) |
|---|---:|---:|---:|
| UniAD | 0.5 | 0.165 | 0.340 |
| UniAD | 1.5 | 0.279 | 0.582 |
| UniAD | 2.5 | 0.362 | 0.759 |
| UniAD | 4.0 | 0.454 | 0.977 |
| NAVSIM | 0.5 | 0.293 | 0.743 |
| NAVSIM | 1.0 | 0.364 | 0.924 |
| NAVSIM | 1.5 | 0.421 | 1.073 |

## Checkpoint format

Each checkpoint contains the predictor tensors, model configuration and normalization metadata for inference
and weight initialization. `manifest.json` records file SHA-256 values, canonical tensor-state hashes,
training stages, feature interfaces and frozen-host identities. Use each release file's `sha256` value
in the evaluation config.
