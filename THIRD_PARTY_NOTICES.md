# Third-party source and dependency notes

The UniAD Apache-2.0 text and V-JEPA 2 MIT notice are included under [licenses/](licenses/).

| Source | Relationship |
|---|---|
| [UniAD](https://github.com/OpenDriveLab/UniAD) | Frozen inference port, heads, configuration and numerical compatibility adaptations; [Apache-2.0 text](licenses/UniAD-Apache-2.0.txt) |
| [V-JEPA 2](https://github.com/facebookresearch/vjepa2) | Attention and patch-embedding implementation adapted by the bundled `ac_modules.py` files; [MIT notice](licenses/V-JEPA2-MIT.txt) |
| [FlowSim](https://github.com/hanfeng-wu/FlowSim) | Locked runtime and reference evaluation interfaces |
| [NAVSIM](https://github.com/autonomousvision/navsim) | External dataset/model interfaces used through the FlowSim checkout |
| [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit) | External SDK, split definitions, geometry and map API interfaces |
| MMDetection / MMDetection3D / MMCV / MMEngine | External frozen-UniAD runtime dependencies |
| SplatAD / neurad-studio, OmniRe / drivestudio, Street Gaussians | Reconstruction backends used by the integration adapters |
| NVIDIA Difix | Optional external image-correction model used through an adapter |
| timm | DropPath and external LTF backbone support |

Provider links for frozen host weights, datasets and motion anchors appear in the integration guides.
The JEPA-SIM predictor release is described in [models/README.md](models/README.md).
