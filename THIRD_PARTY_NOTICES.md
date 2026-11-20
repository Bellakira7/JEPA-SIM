# Third-party source and dependency notes

No project-wide license grant has been selected. Upstream-derived components retain their original terms;
bundling them does not replace those terms or grant a new license to other components.
The UniAD Apache-2.0 text and V-JEPA 2 MIT notice are included under [licenses/](licenses/).

| Source | Relationship |
|---|---|
| [UniAD](https://github.com/OpenDriveLab/UniAD) | Frozen inference port, heads, configuration and numerical compatibility adaptations; [Apache-2.0 text](licenses/UniAD-Apache-2.0.txt) |
| [V-JEPA 2](https://github.com/facebookresearch/vjepa2) | Attention and patch-embedding implementation adapted by the bundled `ac_modules.py` files; [MIT notice](licenses/V-JEPA2-MIT.txt) |
| [FlowSim](https://github.com/hanfeng-wu/FlowSim) | External locked runtime and reference evaluation interfaces; its full repository is not bundled |
| [NAVSIM](https://github.com/autonomousvision/navsim) | External dataset/model interfaces used through the FlowSim checkout |
| [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit) | External SDK, split definitions, geometry and map API interfaces |
| MMDetection / MMDetection3D / MMCV / MMEngine | External frozen-UniAD runtime dependencies |
| SplatAD / neurad-studio, OmniRe / drivestudio, Street Gaussians | Optional reconstruction backends; only integration adapters are bundled |
| NVIDIA Difix | Optional external image-correction model used through an adapter |
| timm | DropPath and external LTF backbone support |

Dataset, pretrained checkpoint and reconstruction licenses are separate from source-code licensing.
Frozen host weights, data and motion-anchor files must be obtained from their providers under their respective terms.
The JEPA-SIM predictor release is described in [models/README.md](models/README.md).
