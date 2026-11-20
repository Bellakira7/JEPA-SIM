"""Use the exported per-frame OpenScene poses in native NuPlan CameraData."""
import json
from pathlib import Path
import numpy as np
import torch


def install(module):
    original_init=module.NuPlanCameraData.__init__
    if not getattr(original_init,'_openscene',False):
        def initialize(self,**kwargs):
            # Missing annotations stay absent; native CameraData.to expects
            # this optional attribute even when load_sky_mask=False.
            self.sky_masks=None
            original_init(self,**kwargs)
        initialize._openscene=True
        module.NuPlanCameraData.__init__=initialize
    original=module.NuPlanCameraData.load_calibrations
    if getattr(original,'_openscene',False):return
    def load(self):
        original(self)
        path=Path(self.data_path)/'manifest.json'
        if not path.exists():return
        manifest=json.loads(path.read_text())
        frame0=manifest['frames'][self.start_timestep]
        world=np.linalg.inv(np.asarray(frame0['ego2global']))
        name=('CAM_F0','CAM_L0','CAM_R0')[int(self.cam_id)]
        poses=[world @ np.asarray(f['cameras'][name]['c2w_global'])
            for f in manifest['frames'][self.start_timestep:self.end_timestep]]
        self.cam_to_worlds=torch.tensor(np.asarray(poses),dtype=torch.float32)
    load._openscene=True
    module.NuPlanCameraData.load_calibrations=load
