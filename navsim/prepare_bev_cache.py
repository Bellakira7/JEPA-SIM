"""Cached FP32 camera stages -> final BEV, adapted from protocol eval_template.py."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
RANK=int(os.environ.get('RANK',0));LOCAL_RANK=int(os.environ.get('LOCAL_RANK',0));WORLD=int(os.environ.get('WORLD_SIZE',1))
for key in ('RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT'):os.environ.pop(key,None)
CODE=Path(__file__).resolve().parent;sys.path.insert(0,str(CODE/'core'))
from protocol import write_json,sha256_file

def reference(config):
    from sequence_data import validate_manifest
    path=Path(config['reference_sequences'])
    manifest=json.loads(path.read_text())
    validate_manifest(manifest['config'],path)
    return manifest

def frame_names(manifest):
    pairs=json.loads((Path(manifest['source_cache_root'])/'pairs.json').read_text())
    return sorted({p[field] for p in pairs for field in ('source','target')})

class FrameDataset:
    def __init__(self,names,root):self.names,self.root=names,Path(root)
    def __len__(self):return len(self.names)
    def __getitem__(self,index):
        import torch
        name=self.names[index]
        return name,torch.load(self.root/(name+'.pt'),map_location='cpu',weights_only=False)

def convert(config,out):
    import torch
    from torch.utils.data import DataLoader
    from eval_open_loop import load_agent
    from ltf_features import replay_camera_stages,CHANNELS,GRIDS
    manifest=reference(config);names=frame_names(manifest)[RANK::WORLD]
    cache=out/'bev_features';directory=cache/'cache';directory.mkdir(parents=True,exist_ok=True)
    receipt=cache/f'rank_{RANK:02d}.json';reference_hash=sha256_file(config['reference_sequences'])
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        assert saved['reference_hash']==reference_hash and saved['world']==WORLD and saved['frames']==len(names)
        assert all((directory/(name+'.pt')).is_file() for name in names)
        print('BEV_CACHE_SHARD_ALREADY_COMPLETE',RANK,flush=True);return
    source=json.loads(Path(config['source_config']).read_text());source.update(arms=['identity'],device='cuda')
    sys.path.insert(0,source['flowsim_root']);torch.cuda.set_device(LOCAL_RANK)
    agent=load_agent(source,torch)
    loader=DataLoader(FrameDataset(names,Path(manifest['source_cache_root'])/'cache'),
        batch_size=config.get('conversion_batch_size',4),num_workers=2,pin_memory=True,shuffle=False)
    sums=torch.zeros(512,dtype=torch.float64);squares=sums.clone();count=0;processed=0
    with torch.inference_mode():
        for keys,stages in loader:
            for stage,c,grid in zip(stages,CHANNELS,GRIDS):
                assert stage.dtype==torch.float32 and tuple(stage.shape[1:])==(c,*grid)
            bev=replay_camera_stages(agent,[s.cuda(non_blocking=True) for s in stages]).cpu().float()
            assert tuple(bev.shape[1:])==(512,8,8) and torch.isfinite(bev).all()
            flat=bev.double().permute(1,0,2,3).flatten(1)
            sums+=flat.sum(1);squares+=flat.square().sum(1);count+=flat.shape[1]
            for key,value in zip(keys,bev):
                path=directory/(key+'.pt');tmp=path.with_suffix('.partial')
                torch.save([value.clone()],tmp);tmp.replace(path)
            processed+=len(keys)
            if processed==len(keys) or processed%1000< len(keys) or processed==len(names):
                print('BEV_CACHE_PROGRESS',RANK,processed,len(names),flush=True)
    torch.save(dict(sums=sums,squares=squares,count=count),cache/f'moments_{RANK:02d}.pt')
    write_json(receipt,dict(rank=RANK,world=WORLD,frames=len(names),reference_hash=reference_hash,
        ltf_sha256=manifest['ltf_sha256'],shape=[512,8,8],dtype='float32',
        moments_sha256=sha256_file(cache/f'moments_{RANK:02d}.pt')))
    print('BEV_CACHE_SHARD_COMPLETE',RANK,len(names),flush=True)

def merge(config,out):
    import torch
    manifest=reference(config);names=frame_names(manifest);cache=out/'bev_features';world=config['world_size']
    sums=torch.zeros(512,dtype=torch.float64);squares=sums.clone();count=0;total=0;receipts=[]
    for rank in range(world):
        info=json.loads((cache/f'rank_{rank:02d}.json').read_text())
        assert info['rank']==rank and info['world']==world and info['reference_hash']==sha256_file(config['reference_sequences'])
        assert info['ltf_sha256']==sha256_file(config['ltf_checkpoint'])
        assert info['moments_sha256']==sha256_file(cache/f'moments_{rank:02d}.pt')
        assert info['frames']==len(names[rank::world]);total+=info['frames'];receipts.append(info)
        moments=torch.load(cache/f'moments_{rank:02d}.pt',weights_only=False)
        sums+=moments['sums'];squares+=moments['squares'];count+=moments['count']
    assert total==len(names) and all((cache/'cache'/(name+'.pt')).is_file() for name in names)
    mean=sums/count;std=(squares/count-mean.square()).clamp_min(1e-8).sqrt()
    torch.save(dict(mean=[mean.float()],std=[std.float()]),cache/'normalization.pt')
    write_json(cache/'BEV_CACHE_COMPLETE.json',dict(complete=True,frames=total,reference_hash=sha256_file(config['reference_sequences']),
        feature_cut=config['feature_cut'],dtype='float32',shape=[512,8,8],ltf_sha256=manifest['ltf_sha256'],shards=receipts,
        normalization_sha256=sha256_file(cache/'normalization.pt'),normalization_policy='unique training frames only / channel population moments'))
    new=copy.deepcopy(manifest);new.update(config=config,source_cache_root=str(cache),
        legacy_condition_cache_root=str(Path(manifest['source_cache_root'])/'cache'),
        source_artifacts={name:sha256_file(cache/name) for name in ('normalization.pt','BEV_CACHE_COMPLETE.json')},
        source_reference_manifest_sha256=sha256_file(config['reference_sequences']))
    if (out/'sequences.json').exists():assert json.loads((out/'sequences.json').read_text())==new
    else:write_json(out/'sequences.json',new)
    write_json(out/'CONDITIONS_COMPLETE.json',dict(conditions=len(new['extra_conditions']),manifest_sha256=sha256_file(out/'sequences.json'),
        reused_reference=str(Path(config['reference_sequences']).parent/'CONDITIONS_COMPLETE.json')))
    print('BEV_CACHE_COMPLETE',total,'stage_steps',
          {stage: recipe['steps'] for stage, recipe in new['stages'].items()},flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('convert','merge'));ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();config=json.loads(Path(args.config).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    print('config: '+json.dumps(dict(phase=args.phase,config=config)),flush=True)
    (convert if args.phase=='convert' else merge)(config,out)

if __name__=='__main__':main()
