"""Logged map/cuboid and associated-source cuboid conditions in LTF panorama."""
import cv2
import numpy as np
import torch

CAMS = ('CAM_L0','CAM_F0','CAM_R0')
COLORS = {'vehicle':(255,0,0),'pedestrian':(0,255,0),'bicycle':(0,0,255)}
FACES = ((0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5))


def project_global(points, frame, camera):
    cal = frame['cams'][camera]
    c2l = np.eye(4)
    c2l[:3,:3], c2l[:3,3] = cal['sensor2lidar_rotation'],cal['sensor2lidar_translation']
    c2w = np.asarray(frame['lidar2global']) @ c2l
    local = (np.asarray(points)-c2w[:3,3]) @ c2w[:3,:3]
    uv,_ = cv2.projectPoints(local,np.zeros(3),np.zeros(3),
        np.asarray(cal['cam_intrinsic']),np.asarray(cal['distortion']))
    return uv.reshape(-1,2),local[:,2]


def box_world(box, frame):
    x,y,z,length,width,height,yaw = box
    corners = np.array([[a,b,c] for c in (-.5,.5) for b in (-.5,.5) for a in (-.5,.5)])
    corners *= [length,width,height]
    co,si = np.cos(yaw),np.sin(yaw)
    r = np.array([[co,-si,0],[si,co,0],[0,0,1]])
    local = corners @ r.T + [x,y,z]
    pose = np.asarray(frame['lidar2global'])
    return local @ pose[:3,:3].T + pose[:3,3]


def stitch(images, size=(256,64)):
    # Exact crops from FlowSimFeatureBuilder._get_camera_feature.
    cropped = [images[0][28:-28,416:-416],images[1][28:-28],images[2][28:-28,416:-416]]
    return cv2.resize(np.concatenate(cropped,axis=1),size)


def condition_layout(scene, raw_frames, gap, target_index=3, image_shapes=None):
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    if not 0 < gap <= target_index < len(raw_frames):
        raise ValueError('Layout requires an earlier source and a valid target')
    target, source = raw_frames[target_index], raw_frames[target_index-gap]
    current_ids = set(target['anns']['track_tokens'])
    # Map polygons use the actual map API and target pose; no target RGB is used.
    position = target['ego2global_translation']
    objects = scene.map_api.get_proximal_map_objects(Point2D(*position[:2]),64,
        [SemanticMapLayer.LANE,SemanticMapLayer.LANE_CONNECTOR,SemanticMapLayer.CROSSWALK])
    main, assoc = [], []
    for camera in CAMS:
        if image_shapes is None:
            cam = getattr(scene.frames[target_index].cameras,camera.lower())
            height,width = cam.image.shape[:2]
        else:
            height,width = image_shapes[camera]
        image = np.zeros((height,width,3),np.uint8)
        other = np.zeros_like(image)
        for layer,entries in objects.items():
            for item in entries:
                xy = np.asarray(item.polygon.exterior.coords)[:,:2]
                points = np.column_stack([xy,np.full(len(xy),position[2])])
                uv,z = project_global(points,target,camera)
                # Avoid near-plane crossings projecting polygons through infinity.
                if len(uv)>=3 and np.all(z>.5) and np.isfinite(uv).all():
                    uv = np.clip(uv,-10000,10000).astype(np.int32)
                    cv2.fillPoly(image,[uv],(32,48,64) if layer!=SemanticMapLayer.CROSSWALK else (80,80,0))
        for frame,canvas,associated in ((target,image,False),(source,other,True)):
            for box,name,track in zip(frame['anns']['gt_boxes'],frame['anns']['gt_names'],frame['anns']['track_tokens']):
                if name not in COLORS or (associated and track not in current_ids):
                    continue
                uv,z = project_global(box_world(box,frame),target,camera)
                if np.all(z>.5) and np.isfinite(uv).all():
                    uv = np.clip(uv,-10000,10000).astype(np.int32)
                    for face in FACES:
                        cv2.fillPoly(canvas,[uv[list(face)]],COLORS[name])
        main.append(image)
        assoc.append(other)
    array = np.concatenate([stitch(main),stitch(assoc)],axis=2)
    return torch.from_numpy(array.copy()).permute(2,0,1).float()/255


def action_between(raw_frames,gap,target_index=3):
    if not 0 < gap <= target_index < len(raw_frames):
        raise ValueError('Action requires an earlier source and a valid target')
    source,target = raw_frames[target_index-gap],raw_frames[target_index]
    pose = np.linalg.inv(source['ego2global']) @ target['ego2global']
    return torch.tensor([pose[0,3],pose[1,3],np.arctan2(pose[1,0],pose[0,0]),gap*.5],dtype=torch.float32)
