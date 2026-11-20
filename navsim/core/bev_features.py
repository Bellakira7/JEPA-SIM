"""Predict the final fused BEV read by the frozen planning head."""
import torch
import torch.nn.functional as F
from feat_vendor.pyramid_predictor import PyramidPredictor

FEATURE_CUT='ltf/final_fused_bev_v1'
GRIDS=((8,8),)
CHANNELS=(512,)

def predictor_kwargs():
    return dict(level_grids=GRIDS,level_pools=(1,),n_cams=1,in_chans=CHANNELS,
        level_strides=(32,),embed_dim=384,depth=8,num_heads=6,action_dim=4,
        enable_layout=True,residual=True,assoc_layout=True,warp_levels=(),
        use_activation_checkpointing=True)

class BEVPredictor(PyramidPredictor):
    def _assoc_layout_tokens(self,layout):
        # Reuse the six camera-layout channels as a learned condition. Pooling
        # changes condition encoding resolution, not camera-to-BEV geometry.
        b,t,n,c,h,w=layout.shape
        grid=(self.Hp*self.level_pools[0],self.Wp*self.level_pools[0])
        resized=F.adaptive_avg_pool2d(layout.reshape(b*t*n,c,h,w),grid)
        return super()._assoc_layout_tokens(resized.reshape(b,t,n,c,*grid))

def build_predictor(kwargs=None):return BEVPredictor(**(kwargs or predictor_kwargs()))

def predict_bev(agent,predictor,builder,inputs,scene,raw,gap,mean,std,device,feature_cache,condition_cache):
    from scene_layout import action_between,condition_layout
    from train_sequence import forecast
    target=scene.scene_metadata.num_history_frames-1;anchor=target-gap;first=anchor-3
    if first<0:raise ValueError('BEV prediction requires four real anchor frames')
    for i in range(first,anchor+1):
        if i not in feature_cache:
            image=builder._get_camera_feature(inputs,idx=i).unsqueeze(0).to(device)
            feature_cache[i]=agent._extract_bev_features(image)
    for i in range(first,target):
        if i not in condition_cache:
            condition_cache[i]=(action_between(raw,1,target_index=i+1).to(device),condition_layout(scene,raw,1,target_index=i+1).to(device))
    history=[torch.stack([((feature_cache[i]-mean[0])/std[0])[0] for i in range(first,anchor+1)])[None,:,None]]
    actions=torch.stack([condition_cache[i][0] for i in range(first,target)])[None]
    layout=torch.stack([condition_cache[i][1] for i in range(first,target)])[None,:,None]
    dense,ar=forecast(predictor,history,actions,layout,gap)
    output=dense if gap==1 else ar
    return output[0][:,-1,0]*std[0]+mean[0]
