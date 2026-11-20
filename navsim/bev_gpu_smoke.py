"""BEV TF/RO8 backward + causal windows, adapted after copying eval_template.py."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
for key in ('RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT'):os.environ.pop(key,None)
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text());print('config: '+json.dumps(cfg),flush=True)
    import torch
    from train_sequence import initialize_predictor,losses,forecast
    from protocol import write_json
    model,kwargs=initialize_predictor(cfg);model=model.cuda().train();optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['tf_lr'])
    features=[torch.randn(1,12,1,512,8,8,device='cuda')]
    actions=torch.randn(1,11,4,device='cuda');actions[...,3]=.5
    layout=torch.rand(1,11,1,6,64,256,device='cuda');rows=[]
    for stage,horizon in [('tf',1),('ro',8)]:
        torch.cuda.synchronize();started=time.monotonic();optimizer.zero_grad(set_to_none=True)
        loss,_=losses(model,features,actions,layout,stage,horizon)
        assert torch.isfinite(loss);loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        torch.cuda.synchronize();rows.append(dict(stage=stage,horizon=horizon,loss=float(loss),gradient=float(gradient),elapsed_s=time.monotonic()-started))
        print('BEV_GPU_SMOKE_PROGRESS',json.dumps(rows[-1]),flush=True)
    model.eval()
    with torch.inference_mode():
        dense,ar=forecast(model,[features[0][:,:4]],actions,layout,3)
        assert dense[0].shape==(1,4,1,512,8,8) and ar[0].shape==(1,2,1,512,8,8)
        changed=features[0].clone();changed[:,4:]+=1000
        dense2,ar2=forecast(model,[changed[:,:4]],actions,layout,3)
        torch.testing.assert_close(dense[0],dense2[0],atol=0,rtol=0);torch.testing.assert_close(ar[0],ar2[0],atol=0,rtol=0)
    write_json(args.out,dict(status='passed',kind='synthetic interface/optimizer/causality check, not performance',feature_cut=cfg['feature_cut'],predictor_kwargs=kwargs,rows=rows,parameters=sum(p.numel() for p in model.parameters()),max_memory_allocated=torch.cuda.max_memory_allocated()))
    print('BEV_GPU_SMOKE_PASSED',flush=True)

if __name__=='__main__':main()
