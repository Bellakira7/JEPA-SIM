"""Native SplatAD ADDataParser for calibrated, 2 Hz OpenScene clip exports."""
from collections import defaultdict
from dataclasses import dataclass,field
import json
from pathlib import Path
from typing import Type
import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras,CameraType
from nerfstudio.cameras.lidars import Lidars,LidarType
from nerfstudio.data.dataparsers.ad_dataparser import ADDataParser,ADDataParserConfig,OPENCV_TO_NERFSTUDIO

ACTOR_AXES=np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]],np.float32)


@dataclass
class OpenSceneParserConfig(ADDataParserConfig):
    _target:Type=field(default_factory=lambda:OpenSceneParser)
    data:Path=Path('data/openscene')
    cameras:tuple=('CAM_F0','CAM_L0','CAM_R0')
    lidars:tuple=('MERGED_LIDAR',)
    train_split_fraction:float=1.0
    annotation_interval:float=.5
    include_deformable_actors:bool=True
    allow_per_point_times:bool=False
    add_missing_points:bool=False


@dataclass
class OpenSceneParser(ADDataParser):
    config:OpenSceneParserConfig

    @property
    def actor_transform(self):return torch.from_numpy(ACTOR_AXES)

    def manifest(self):
        return json.loads((self.config.data/'manifest.json').read_text())

    def origin(self):
        return np.asarray(self.manifest()['frames'][0]['ego2global'])[:3,3]

    def _get_cameras(self):
        manifest=self.manifest()
        poses,ks,sizes,times,indices,files=[],[],[],[],[],[]
        for ci,cam in enumerate(self.config.cameras):
            for frame in manifest['frames']:
                item=frame['cameras'][cam]
                pose=np.asarray(item['c2w_global']).copy()
                pose[:3,3]-=self.origin()
                pose[:3,:3]=pose[:3,:3] @ OPENCV_TO_NERFSTUDIO
                poses.append(pose[:3]);ks.append(item['K']);sizes.append(item['size'])
                times.append(item['timestamp_usec']*1e-6);indices.append(ci);files.append(Path(item['path']))
        ks=torch.tensor(np.asarray(ks),dtype=torch.float32)
        cameras=Cameras(camera_to_worlds=torch.tensor(np.asarray(poses),dtype=torch.float32),
            fx=ks[:,0,0],fy=ks[:,1,1],cx=ks[:,0,2],cy=ks[:,1,2],
            width=torch.tensor([s[0] for s in sizes]),height=torch.tensor([s[1] for s in sizes]),
            camera_type=CameraType.PERSPECTIVE,times=torch.tensor(times,dtype=torch.float64),
            metadata={'sensor_idxs':torch.tensor(indices)[:,None],'is_key_frame':torch.ones(len(files),1,dtype=torch.bool)})
        return cameras,files

    def _get_lidars(self):
        frames=self.manifest()['frames']
        # The merged nuPlan cloud supplies measured RGB reconstruction seeds and
        # point-conditioned LiDAR supervision. No fabricated beam/time samples.
        poses=np.asarray([f['lidar2global'][:3] for f in frames])
        poses[:,:,3]-=self.origin()
        lidars=Lidars(lidar_to_worlds=torch.tensor(poses,dtype=torch.float32),
            lidar_type=LidarType.VELODYNE_HDL32E,assume_ego_compensated=True,
            times=torch.tensor([f['timestamp_usec']*1e-6 for f in frames],dtype=torch.float64),
            metadata={'sensor_idxs':torch.zeros(len(frames),1,dtype=torch.int64),'is_key_frame':torch.ones(len(frames),1,dtype=torch.bool)})
        return lidars,[self.config.data/'lidar_raw'/f'{j:03d}.bin' for j in range(len(frames))]

    def _read_lidars(self,lidars,filenames):
        clouds=[]
        for file in filenames:
            pc=np.fromfile(file,np.float32).reshape(-1,6).copy()
            pc[:,3]/=255
            pc[:,4]=0
            clouds.append(torch.from_numpy(pc))
        return clouds

    def _get_actor_trajectories(self):
        manifest=self.manifest()
        instances=json.loads((self.config.data/'instances/instances_info.json').read_text())
        output=[]
        for instance in instances.values():
            ann=instance['frame_annotations']
            if len(ann['frame_idx'])<2:continue
            poses=np.asarray(ann['obj_to_world'])
            poses[:,:3,3]-=self.origin()
            poses=torch.tensor(poses,dtype=torch.float32)
            if not (poses[:,:2,3].std(0)>.5).any():continue
            poses=poses @ torch.from_numpy(ACTOR_AXES)
            human=instance['class_name']=='pedestrian'
            times=torch.tensor([manifest['frames'][j]['timestamp_usec']*1e-6 for j in ann['frame_idx']],dtype=torch.float64)
            dims=torch.tensor(ann['box_size'],dtype=torch.float32).max(0).values[[1,0,2]]
            output.append(dict(uuid=instance['id'],label='human.pedestrian' if human else 'vehicle.car',
                poses=poses,timestamps=times,dims=dims,stationary=False,symmetric=not human,deformable=human))
        return output
