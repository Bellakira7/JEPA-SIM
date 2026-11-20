"""User-approved OmniRe ablation: deformable pedestrians, no SMPL assets."""
def configure_nosmpl():
    from datasets.nuscenes import nuscenes_sourceloader as ns
    from datasets.base.scene_dataset import ModelType
    for name in ns.SMPLNODE_CLASSES:
        ns.OBJECT_CLASS_NODE_MAPPING[name] = ModelType.DeformableNodes
    from datasets.nuplan import nuplan_sourceloader as np
    for name in np.SMPLNODE_CLASSES:
        np.OBJECT_CLASS_NODE_MAPPING[name] = ModelType.DeformableNodes
    # OpenScene carries per-frame camera-to-LiDAR calibration. Preserve it,
    # including any sensor-time offset, instead of the native constant rig pose.
    from .nuplan_calibration import install
    install(np)
    from .omnire_optional_sky import install as install_optional_sky
    install_optional_sky()


def train():
    configure_nosmpl()
    import runpy
    from pathlib import Path
    import datasets
    repo = Path(datasets.__file__).resolve().parent.parent
    runpy.run_path(str(repo / 'tools/train.py'), run_name='__main__')


if __name__ == '__main__':
    train()
