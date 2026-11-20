"""Check OpenScene export and native SplatAD/OmniRe/StreetGS input geometry."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
from protocol import write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--gs-upstream',type=Path)
    a=ap.parse_args();config=json.loads(Path(a.config).read_text());out=Path(a.out)
    if config.get('nuplan_devkit_root'):
        sys.path.insert(0, config['nuplan_devkit_root'])
    sys.path.insert(0,config['splatad_repo'])
    from prepare_reconstruction import export
    data=out/'processed'/'openscene_pilot'
    if not (data/'manifest.json').exists():
        export(Path(config['reconstruction_metadata']),Path(config['openscene_root'])/'sensor_blobs'/config['data_split'],data,0,12)
    from openscene_splat_parser import OpenSceneParserConfig
    parser=OpenSceneParserConfig(data=data).setup()
    result=parser.get_dataparser_outputs('train')
    if len(result.cameras)!=36 or len(result.metadata['point_clouds'])!=12:
        raise ValueError('Wrong native SplatAD camera/LiDAR coverage')
    # Recover camera pose back into global OpenCV coordinates after ADDataParser
    # coordinate normalization, and compare against metadata.
    import torch
    from nerfstudio.data.dataparsers.ad_dataparser import OPENCV_TO_NERFSTUDIO
    manifest=json.loads((data/'manifest.json').read_text())
    transform=np.eye(4);transform[:3]=result.dataparser_transform.numpy()
    errors=[]
    for ci,name in enumerate(('CAM_F0','CAM_L0','CAM_R0')):
        for j,frame in enumerate(manifest['frames']):
            pose=np.eye(4);pose[:3]=result.cameras.camera_to_worlds[ci*12+j].numpy()
            recovered=np.linalg.inv(transform) @ pose
            recovered[:3,3]+=parser.origin()
            recovered[:3,:3]=recovered[:3,:3] @ OPENCV_TO_NERFSTUDIO
            expected=np.asarray(frame['cameras'][name]['c2w_global'])
            errors.append(float(np.abs(recovered-expected).max()))
    error=max(errors)
    if error>1e-4:raise ValueError(f'Camera world pose mismatch: {error}')
    checks={}
    if a.gs_upstream:
        from run_reconstruction import configure
        sys.path.insert(0,str(a.gs_upstream/'drivestudio'))
        from gs_vendor.omnire_variant import configure_nosmpl
        configure_nosmpl()
        from omegaconf import OmegaConf
        from datasets.driving_dataset import DrivingDataset
        (out/'omnire').mkdir(exist_ok=True)
        _,spec,_=configure('omnire_nosmpl',data,a.gs_upstream,out/'omnire',100)
        cfg=OmegaConf.load(out/'omnire/train.yaml')
        cfg.data.preload_device='cpu'
        ds=DrivingDataset(data_cfg=cfg.data)
        w=np.asarray(manifest['world_from_global'])
        deltas=[]
        for ci,name in enumerate(('CAM_F0','CAM_L0','CAM_R0')):
            cam=ds.pixel_source.camera_data[ci]
            if cam.sky_masks is not None:raise ValueError('Unavailable sky annotations were fabricated')
            expected=np.stack([w @ np.asarray(f['cameras'][name]['c2w_global']) for f in manifest['frames']])
            deltas.append(float(np.abs(cam.cam_to_worlds.numpy()-expected).max()))
        delta=max(deltas)
        if delta>1e-4:raise ValueError(f'OmniRe camera mismatch: {delta}')
        checks['omnire_camera_max_abs']=delta
        # Exercise the real RGB/SSIM loss and backward without any sky label.
        from models.trainers.base import BasicTrainer
        from types import SimpleNamespace
        from pytorch_msssim import SSIM
        context=SimpleNamespace(losses_dict=OmegaConf.create(dict(rgb=dict(w=.8),ssim=dict(w=.2),mask=dict(w=0.))),
            ssim=SSIM(data_range=1.,size_average=True,channel=3),
            sky_opacity_loss_fn=lambda *_: (_ for _ in ()).throw(ValueError('Unexpected sky supervision')),
            depth_loss_fn=None,gaussian_classes={},models={},step=0)
        rgb=torch.rand(32,32,3,requires_grad=True)
        losses=BasicTrainer.compute_losses(context,dict(rgb=rgb,opacity=torch.ones(32,32,1)),dict(pixels=torch.rand(32,32,3)),{})
        total=sum(losses.values());total.backward()
        if not torch.isfinite(total) or not torch.isfinite(rgb.grad).all():raise ValueError('Nonfinite native no-sky RGB loss')
        checks['omnire_no_sky_loss_backward']='passed'
        sys.path.insert(0,str(a.gs_upstream/'street_gaussians'))
        (out/'street').mkdir(exist_ok=True)
        _,spec,_=configure('street_gs',data,a.gs_upstream,out/'street',100)
        sys.argv=['street','--config',str(out/'street/train.yaml')]
        from gs_vendor.street_reader import read_nuscenes
        info=read_nuscenes(data)
        if len(info.train_cameras)!=36:raise ValueError('Wrong StreetGS coverage')
        checks['street_camera_count']=len(info.train_cameras)
        errors=[]
        for camera in info.train_cameras:
            j,ci=camera.metadata['frame'],camera.metadata['cam']
            name=('CAM_F0','CAM_L0','CAM_R0')[ci]
            expected=w @ np.asarray(manifest['frames'][j]['cameras'][name]['c2w_global'])
            errors.append(float(np.abs(camera.metadata['extrinsic']-expected).max()))
        checks['street_camera_max_abs']=max(errors)
        if max(errors)>1e-4:raise ValueError('StreetGS camera geometry mismatch')
    write_json(out/'preflight.json',dict(status='ready',camera_count=36,lidar_count=12,
        trajectories=len(result.metadata['trajectories']),max_abs_global_pose_error=error,
        extra_checks=checks,
        note='Native input/geometry audit; no reconstruction performance result'))
    print('done: SplatAD export/input geometry passed',flush=True)


if __name__=='__main__':main()
