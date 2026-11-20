"""Production feature shapes, FP32 TF/RO8 optimizer steps, and optional flow backward."""
import argparse
import json
from pathlib import Path
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--device-index', type=int, default=0)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    print('config: '+json.dumps(config), flush=True)
    import torch
    from train_sequence import initialize_predictor, losses
    from ltf_features import GRIDS, CHANNELS
    from protocol import write_json
    torch.cuda.set_device(args.device_index)
    device = 'cuda'
    model, kwargs = initialize_predictor(config)
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['tf_lr'], weight_decay=.04)
    features = [torch.randn(1,12,1,c,h,w,device=device) for c,(h,w) in zip(CHANNELS,GRIDS)]
    actions = torch.randn(1,11,4,device=device)
    actions[...,3] = .5
    layout = torch.rand(1,11,1,6,*GRIDS[0],device=device)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for stage,horizon in [('tf',1),('ro',8)]:
        torch.cuda.synchronize()
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss,_ = losses(model,features,actions,layout,stage,horizon)
        if not torch.isfinite(loss):
            raise ValueError('Production-shape smoke loss is nonfinite')
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        flow = {}
        for level in config['warp_levels']:
            value = model.flow_proj[str(level)].weight.grad
            if value is None or not torch.isfinite(value).all() or not value.abs().sum() > 0:
                raise ValueError('Missing or invalid flow gradient')
            flow[str(level)] = float(value.abs().sum())
        optimizer.step()
        torch.cuda.synchronize()
        rows.append(dict(stage=stage,horizon=horizon,loss=float(loss),grad_norm=float(gradient),
                         elapsed_s=time.monotonic()-started,flow_gradient_l1=flow))
        print('GPU_SMOKE_PROGRESS',json.dumps(rows[-1]),flush=True)
    value = dict(status='passed',kind='synthetic production-shape interface check; no performance evidence',
                 torch_version=torch.__version__,device=torch.cuda.get_device_name(),device_index=args.device_index,
                 predictor_kwargs=kwargs,history_frames=4,batch_per_gpu=1,precision='FP32',stages=rows,
                 max_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                 max_memory_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json(args.out,value)
    print('PRODUCTION_GPU_TF_RO8_SMOKE_PASSED',str(args.out),flush=True)


if __name__ == '__main__':
    main()
