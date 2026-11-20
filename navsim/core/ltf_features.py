"""Camera features before each LTF fusion block, and frozen fusion replay.

The later camera stages contain previous fusion context. At inference all four
predicted pre-fusion tensors are injected; no target camera encoder is executed.
"""
from pathlib import Path
from unittest.mock import patch
import torch

GRIDS = ((64,256),(32,128),(16,64),(8,32))
CHANNELS = (64,128,256,512)
POOLS = (8,4,2,1)


@torch.no_grad()
def extract_camera_stages(agent, image):
    stages = []
    backbone = agent._bev_model._backbone
    fuse = backbone.fuse_features
    def capture(camera, latent, index):
        stages.append(camera.clone())
        return fuse(camera,latent,index)
    with patch.object(backbone,'fuse_features',capture):
        bev = agent._extract_bev_features(image)
    assert len(stages)==4
    for tensor, channels, grid in zip(stages,CHANNELS,GRIDS):
        assert tuple(tensor.shape[1:])==(channels,*grid), tuple(tensor.shape)
    return stages, bev


@torch.no_grad()
def replay_camera_stages(agent, stages):
    backbone = agent._bev_model._backbone
    if not backbone.config.latent or not backbone.config.transformer_decoder_join:
        raise ValueError('Requires camera-only LTF with transformer decoder join')
    latent = backbone.lidar_latent.repeat(stages[0].shape[0],1,1,1)
    layers = iter(backbone.lidar_encoder.items())
    if len(backbone.lidar_encoder.return_layers)>4:
        latent = backbone.forward_layer_block(layers,backbone.lidar_encoder.return_layers,latent)
    for index, camera in enumerate(stages):
        latent = backbone.forward_layer_block(layers,backbone.lidar_encoder.return_layers,latent)
        _,latent = backbone.fuse_features(camera,latent,index)
    return latent


def predictor_kwargs():
    return dict(level_grids=GRIDS,level_pools=POOLS,n_cams=1,in_chans=CHANNELS,
        level_strides=(4,8,16,32),
        embed_dim=384,depth=8,num_heads=6,action_dim=4,enable_layout=True,
        residual=True,assoc_layout=True)


def build_predictor(kwargs=None):
    from feat_vendor.pyramid_predictor import PyramidPredictor
    return PyramidPredictor(**(kwargs or predictor_kwargs()))


def load_predictor(checkpoint, device):
    item = torch.load(checkpoint,map_location='cpu',weights_only=False)
    if item['feature_cut'] not in ('ltf/pre_fusion_camera_stages_v1','ltf/final_fused_bev_v1'):
        raise ValueError('Wrong feature cut')
    if item['feature_cut']=='ltf/final_fused_bev_v1':
        from bev_features import build_predictor as build_bev_predictor
        model=build_bev_predictor(item['predictor_kwargs'])
    else:
        model = build_predictor(item['predictor_kwargs'])
    model.load_state_dict(item['model'],strict=True)
    model.requires_grad_(False).to(device).eval()
    mean = [v.to(device).view(1,-1,1,1) for v in item['mean']]
    std = [v.to(device).view(1,-1,1,1) for v in item['std']]
    return model, mean, std, item
