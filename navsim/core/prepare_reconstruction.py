"""Export calibrated OpenScene clips for native SplatAD / OmniRe / StreetGS.


distorted pixel coordinates. No original RGB fills missing rendered pixels.
"""
import argparse
import json
from pathlib import Path
import pickle
import numpy as np
import cv2
from PIL import Image
from protocol import write_json,sha256_file
from scene_layout import box_world

CAMS=('CAM_F0','CAM_L0','CAM_R0')


def export(metadata,sensors,out,start,count):
    from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
    frames=pickle.load(open(metadata,'rb'))[start:start+count]
    if len(frames)!=count or count<12:
        raise ValueError('Clip must contain requested frames and 4 history + 8 future frames')
    timestamps=np.asarray([frame['timestamp'] for frame in frames],dtype=np.int64)
    if not np.all(np.abs(np.diff(timestamps)-500000)<50000):
        raise ValueError('Reconstruction clip must be contiguous at 2 Hz')
    if (out/'manifest.json').exists():
        raise FileExistsError('Use a new reconstruction export directory')
    folders=('images','intrinsics','extrinsics','ego_pose','lidar','lidar_pose','lidar_raw',
        'dynamic_masks/all','dynamic_masks/human','dynamic_masks/vehicle','instances')
    for folder in folders:
        (out/folder).mkdir(parents=True,exist_ok=True)
    instances,tracks,by_frame={}, {},{}
    exported=[]
    for j,frame in enumerate(frames):
        np.savetxt(out/'ego_pose'/f'{j:03d}.txt',frame['ego2global'])
        np.savetxt(out/'lidar_pose'/f'{j:03d}.txt',frame['ego2global'])
        with open(sensors/frame['lidar_path'],'rb') as source:
            pc=LidarPointCloud.from_buffer(source,'pcd').points.T.astype(np.float32)
        if pc.shape[1]<4 or not np.isfinite(pc[:,:4]).all():
            raise ValueError('Invalid measured LiDAR cloud')
        raw=np.zeros((len(pc),6),np.float32)
        # nuPlan PCD bin2 uses x/y/z/intensity/ring/lidar_id in that order.
        raw[:,:min(6,pc.shape[1])]=pc[:,:6]
        raw.tofile(out/'lidar_raw'/f'{j:03d}.bin')
        # DriveStudio NuPlan loader expects points already in ego coordinates.
        ego=raw.copy()
        pose=np.asarray(frame['lidar2ego'])
        ego[:,:3]=raw[:,:3] @ pose[:3,:3].T + pose[:3,3]
        ego.tofile(out/'lidar'/f'{j:03d}.bin')
        cameras={}
        for ci,name in enumerate(CAMS):
            cal=frame['cams'][name]
            c2l=np.eye(4)
            c2l[:3,:3],c2l[:3,3]=cal['sensor2lidar_rotation'],cal['sensor2lidar_translation']
            c2e=np.asarray(frame['lidar2ego']) @ c2l
            c2w=np.asarray(frame['lidar2global']) @ c2l
            k=np.asarray(cal['cam_intrinsic'])
            d=np.asarray(cal['distortion'])
            path=sensors/cal['data_path']
            image=np.asarray(Image.open(path).convert('RGB'))
            h,w=image.shape[:2]
            image=cv2.undistort(image,k,d,None,k)
            target=out/'images'/f'{j:03d}_{ci}.jpg'
            Image.fromarray(image).save(target,quality=95)
            np.savetxt(out/'intrinsics'/f'{ci}.txt',[k[0,0],k[1,1],k[0,2],k[1,2],0,0,0,0,0])
            np.savetxt(out/'extrinsics'/f'{ci}.txt',c2e)
            cameras[name]=dict(path=str(target),original_path=str(path),relative_path=cal['data_path'],
                timestamp_usec=int(frame['timestamp']),c2w_global=c2w.tolist(),K=k.tolist(),distortion=d.tolist(),size=[w,h])
            masks={key:np.zeros((h,w),np.uint8) for key in ('all','human','vehicle')}
            w2c=np.linalg.inv(c2w)
            for box,label in zip(frame['anns']['gt_boxes'],frame['anns']['gt_names']):
                if label not in ('vehicle','pedestrian','bicycle'):
                    continue
                world=box_world(box,frame)
                local=world @ w2c[:3,:3].T + w2c[:3,3]
                if np.any(local[:,2]<.5):continue
                uv=local @ k.T
                uv=uv[:,:2]/uv[:,2:3]
                if not np.isfinite(uv).all():continue
                hull=cv2.convexHull(np.clip(uv,-10000,10000).astype(np.int32))
                cv2.fillPoly(masks['all'],[hull],255)
                cv2.fillPoly(masks['human' if label=='pedestrian' else 'vehicle'],[hull],255)
            for kind,mask in masks.items():
                Image.fromarray(mask).save(out/'dynamic_masks'/kind/f'{j:03d}_{ci}.png')
        by_frame[str(j)]=[]
        for box,label,track in zip(frame['anns']['gt_boxes'],frame['anns']['gt_names'],frame['anns']['track_tokens']):
            if label not in ('vehicle','pedestrian','bicycle'):continue
            if track not in tracks:
                key=str(len(tracks));tracks[track]=key
                instances[key]=dict(id=track,class_name=str(label),frame_annotations=dict(frame_idx=[],obj_to_world=[],box_size=[]))
            key=tracks[track]
            x,y,z,l,w,h,yaw=box
            co,si=np.cos(yaw),np.sin(yaw)
            pose=np.eye(4);pose[:3,:3]=[[co,-si,0],[si,co,0],[0,0,1]];pose[:3,3]=[x,y,z]
            world=np.asarray(frame['lidar2global']) @ pose
            annotation=instances[key]['frame_annotations']
            annotation['frame_idx'].append(j);annotation['obj_to_world'].append(world.tolist());annotation['box_size'].append([float(l),float(w),float(h)])
            by_frame[str(j)].append(int(key))
        exported.append(dict(token=frame['token'],timestamp_usec=int(frame['timestamp']),
            ego2global=np.asarray(frame['ego2global']).tolist(),lidar2global=np.asarray(frame['lidar2global']).tolist(),cameras=cameras))
        print(f'progress: export {j+1}/{count}',flush=True)
    write_json(out/'instances/instances_info.json',instances)
    write_json(out/'instances/frame_instances.json',by_frame)
    manifest=dict(version=1,hz=2,scene=out.name,scene_idx=out.name,log=frames[0]['log_name'],
        protocol='OpenScene-LTF-OL-v1 / reconstruction-front3-pilot',
        world_from_global=np.linalg.inv(np.asarray(frames[0]['ego2global'])).tolist(),frames=exported,
        metadata_sha256=sha256_file(metadata),start_frame=start,frame_count=count,
        distortion='fit cv2.undistort; redistort renders to original grid',sky_masks='unavailable; no fabricated masks')
    manifest['lidar_columns']=['x','y','z','intensity','ring','lidar_id']
    write_json(out/'manifest.json',manifest)
    print('done: export '+str(out),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metadata',required=True,type=Path)
    ap.add_argument('--sensors',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--frames',type=int,default=40)
    a=ap.parse_args()
    print('config: '+json.dumps({k:str(v) for k,v in vars(a).items()}),flush=True)
    export(a.metadata,a.sensors,a.out,a.start,a.frames)


if __name__=='__main__':main()
