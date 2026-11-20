"""Load official models; new poses go through each model's own renderer."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .protocol import DRIVE_CAMS, load_manifest, moved_camera, validate_files


class OmniRe:
    def __init__(self, spec):
        from omegaconf import OmegaConf
        from datasets.driving_dataset import DrivingDataset
        from utils.misc import import_str
        self.manifest, self.by_name = load_manifest(spec['manifest'])
        cfg = OmegaConf.load(spec['upstream_config'])
        if 'SMPLNodes' in cfg.model or cfg.data.pixel_source.load_smpl:
            raise ValueError('omnire_nosmpl requires an explicitly SMPL-free saved config')
        # Keep every pedestrian: map SMPL-category boxes to DeformableNodes.
        from .omnire_variant import configure_nosmpl
        configure_nosmpl()
        self.dataset = DrivingDataset(data_cfg=cfg.data)
        self.trainer = import_str(cfg.trainer.type)(
            **cfg.trainer, num_timesteps=self.dataset.num_img_timesteps,
            model_config=cfg.model, num_train_images=len(self.dataset.train_image_set),
            num_full_images=len(self.dataset.full_image_set),
            test_set_indices=self.dataset.test_timesteps,
            scene_aabb=self.dataset.get_aabb().reshape(2, 3), device=torch.device('cuda'))
        # This is the exact locally trained checkpoint whose digest is recorded
        # by the client. Upstream checkpoints include NumPy scalar metadata;
        # PyTorch 2.6+ changed the default loader to weights_only=True.
        state = torch.load(spec['checkpoint'], map_location='cuda', weights_only=False)
        if int(state['step']) != int(spec['iteration']):
            raise ValueError('OmniRe checkpoint step differs from the renderer specification')
        if 'SMPLNodes' in state['models']:
            raise ValueError('omnire_nosmpl cannot load a checkpoint containing SMPLNodes')
        self.trainer.load_state_dict(state, load_only_model=True, strict=True)
        self.trainer.set_eval()
        self.trainer.render_each_class = False
        front = self.manifest['frames'][int(cfg.data.start_timestep)]['ego2global']
        self.world = np.linalg.inv(np.asarray(front, dtype=np.float64))
        self.start = int(cfg.data.start_timestep)
        self.cam_ids = list(cfg.data.pixel_source.cameras)
        if set(self.cam_ids) != set(range(3)):
            raise ValueError('UniAD experiments require all three cameras')

    @torch.no_grad()
    def render(self, cam, item, D):
        frame, _, _ = self.by_name[Path(item['path']).name]
        cam_id = DRIVE_CAMS.index(cam)
        image_idx = (frame - self.start) * 3 + self.cam_ids.index(cam_id)
        images, camera = self.dataset.full_image_set.get_image(image_idx, self.trainer._get_downscale_factor())
        images = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in images.items()}
        camera = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in camera.items()}
        c2w = camera['camera_to_world']
        if D is not None:
            corrected = self.trainer.process_camera(camera, images['img_idx'].flatten()[0], novel_view=False).camtoworlds
            moved = moved_camera(corrected.cpu().numpy(), D, self.world)
            camera['camera_to_world'] = torch.as_tensor(moved, dtype=c2w.dtype, device=c2w.device)
            # Sky uses rays, separately from the Gaussian camera matrices.
            rotation = torch.as_tensor((self.world @ np.asarray(D) @ np.linalg.inv(self.world))[:3, :3],
                                       dtype=images['viewdirs'].dtype, device=c2w.device)
            images['viewdirs'] = images['viewdirs'] @ rotation.T
            images['origins'] = camera['camera_to_world'][:3, 3].expand_as(images['origins'])
        # Learned camera corrections are log-camera local; compose D around them.
        # novel_view avoids re-applying a log-camera optimization to an off-log pose.
        rgb = self.trainer(images, camera, novel_view=D is not None)['rgb']
        return rgb.clamp(0, 1).cpu().numpy()


class StreetGaussians:
    def __init__(self, spec):
        # Upstream config is parsed at import; set argv before the first lib import.
        sys.argv = ['street_gs', '--config', spec['upstream_config'], 'mode', 'evaluate',
                    'loaded_iter', str(spec['iteration']), 'gpus', '[-1]']
        from lib.config import cfg
        from lib.datasets import dataset as module
        from .street_reader import read_nuscenes
        module.sceneLoadTypeCallbacks['NuScenes'] = read_nuscenes
        from lib.models.street_gaussian_model import StreetGaussianModel
        from lib.models.street_gaussian_renderer import StreetGaussianRenderer
        from lib.models.scene import Scene
        self.manifest, self.by_name = load_manifest(spec['manifest'])
        if Path(cfg.source_path).resolve() != Path(spec['manifest']).resolve().parent:
            raise ValueError('StreetGS config and manifest point to different scenes')
        ds = module.Dataset()
        self.model = StreetGaussianModel(ds.scene_info.metadata)
        # Load the requested path explicitly, so iteration/config cannot choose another checkpoint.
        state = torch.load(spec['checkpoint'], map_location='cuda', weights_only=False)
        if int(state['iter']) != int(spec['iteration']):
            raise ValueError('StreetGS checkpoint iteration differs from the renderer specification')
        self.model.load_state_dict(state)
        self.renderer = StreetGaussianRenderer()
        self.cameras = {c.image_name: c for c in ds.train_cameras[1] + ds.test_cameras[1]}
        self.world = np.asarray(self.manifest['world_from_global'], dtype=np.float64)

    @torch.no_grad()
    def render(self, cam, item, D):
        frame, _, _ = self.by_name[Path(item['path']).name]
        camera = self.cameras[f'{frame:03d}_{DRIVE_CAMS.index(cam)}']
        camera = copy.copy(camera)
        if D is not None:
            camera.set_extrinsic(moved_camera(camera.get_extrinsic(), D, self.world))
        return self.renderer.render(camera, self.model)['rgb'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def load_backend(spec):
    return {'omnire_nosmpl': OmniRe, 'street_gs': StreetGaussians}[spec['method']](spec)
