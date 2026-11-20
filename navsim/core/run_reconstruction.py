"""Train a native 3DGS method, export original-grid RGB, then evaluate LTF."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import cv2
from PIL import Image
from protocol import write_json
from prepare_reconstruction import export,CAMS


def run_logged(command,log,env):
    with open(log,'w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
        while process.poll() is None:
            print(f'progress: native pid={process.pid} log={log}',flush=True)
            time.sleep(30)
    if process.returncode:
        print(Path(log).read_text(errors='replace')[-12000:],flush=True)
        raise subprocess.CalledProcessError(process.returncode,command)


def configure(method,data,upstream,out,steps):
    from omegaconf import OmegaConf
    run=out/'trained'
    if method=='omnire_nosmpl':
        repo=upstream/'drivestudio'
        cfg=OmegaConf.merge(OmegaConf.load(repo/'configs/omnire.yaml'),OmegaConf.load(repo/'configs/datasets/nuplan/3cams.yaml'))
        if 'dataset' in cfg:del cfg['dataset']
        del cfg.model['SMPLNodes']
        cfg.data.pixel_source.load_smpl=False
        cfg.data.pixel_source.load_sky_mask=False
        cfg.trainer.losses.mask.w=0.
        cfg.data.pixel_source.downscale_when_loading=[2]*3
        cfg.data.data_root=str(data.parent);cfg.data.scene_idx=data.name
        cfg.trainer.optim.num_iters=steps
        cfg.logging.saveckpt_freq=steps
        cfg.render.render_novel=None
        path=out/'train.yaml';OmegaConf.save(cfg,path)
        spec=dict(method=method,python=sys.executable,repo=str(repo),manifest=str(data/'manifest.json'),
            upstream_config=str(run/'benchmark/run/config.yaml'),checkpoint=str(run/'benchmark/run/checkpoint_final.pth'),iteration=steps)
        cmd=[sys.executable,'-m','gs_vendor.omnire_variant','--config_file',str(path),'--output_root',str(run),'--project','benchmark','--run_name','run']
    else:
        repo=upstream/'street_gaussians'
        override=dict(task='openscene',exp_name='pilot',source_path=str(data),model_path=str(run),record_dir=str(run/'record'),
            to_cuda=False,resolution=2,gpus=[-1],
            data=dict(type='NuScenes',cameras=[0,1,2],split_train=1,split_test=-1,
                selected_frames=[0,39],white_background=False,use_colmap=False,filter_colmap=False,box_scale=1.5),
            model=dict(gaussian=dict(sh_degree=1,fourier_dim=5,flip_prob=.2),nsg=dict(include_bkgd=True,include_obj=True,include_sky=True,opt_track=True),sky=dict(resolution=1024,white_background=True)),
            train=dict(iterations=steps,test_iterations=[],save_iterations=[steps],checkpoint_iterations=[steps]),
            optim=dict(lambda_dssim=.2,lambda_sky=0.,lambda_sky_scale=[1.]*3,lambda_mask=.1,lambda_reg=.1,lambda_depth_lidar=0.,
                densify_grad_abs_bkgd=True,densify_grad_abs_obj=False,densify_until_iter=25000,densify_grad_threshold_bkgd=.0006,densify_grad_threshold_obj=.0002))
        cfg=OmegaConf.merge(OmegaConf.load(repo/'configs/default.yaml'),OmegaConf.create(override))
        path=out/'train.yaml';OmegaConf.save(cfg,path)
        spec=dict(method=method,python=sys.executable,repo=str(repo),manifest=str(data/'manifest.json'),
            upstream_config=str(run/'configs/config_000000.yaml'),checkpoint=str(run/'trained_model'/f'iteration_{steps}.pth'),iteration=steps)
        cmd=[sys.executable,'-m','gs_vendor.street_train','--config',str(path)]
    return cmd,spec,repo


def redistort(rgb,item):
    w,h=item['size']
    rgb=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_LINEAR)
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    coordinates=np.stack([xx,yy],axis=-1)
    k=np.asarray(item['K']);d=np.asarray(item['distortion'])
    uv=cv2.undistortPoints(coordinates.reshape(-1,1,2),k,d,P=k).reshape(h,w,2)
    return cv2.remap(rgb,uv[:,:,0],uv[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)


def render_all(method,spec,data,out):
    import torch
    manifest=json.loads((data/'manifest.json').read_text())
    if method=='splatad':
        from nerfstudio.utils.eval_utils import eval_setup
        _,pipeline,_,step=eval_setup(Path(spec['upstream_config']),test_mode='inference')
        model=pipeline.model.eval()
        cameras=pipeline.datamanager.train_dataset.cameras
        # The exported order is camera-major in OpenSceneParser.
        def render(ci,j,item):
            camera=cameras[ci*len(manifest['frames'])+j:ci*len(manifest['frames'])+j+1].to(model.device)
            return model.get_outputs_for_camera(camera)['rgb'].clamp(0,1).cpu().numpy()
    else:
        from gs_vendor.backends import load_backend
        backend=load_backend(spec)
        def render(ci,j,item):return backend.render(CAMS[ci],item,None)
    metrics=[]
    with torch.inference_mode():
        for j,frame in enumerate(manifest['frames']):
            for ci,cam in enumerate(CAMS):
                item=frame['cameras'][cam]
                pixels=redistort(render(ci,j,item),item)
                raw=np.asarray(Image.open(item['original_path']).convert('RGB')).astype(np.float32)/255
                mse=float(np.mean((pixels-raw)**2))
                path=out/'renders'/item['relative_path'];path.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(np.rint(np.clip(pixels,0,1)*255).astype(np.uint8)).save(path,quality=95)
                metrics.append(dict(token=frame['token'],camera=cam,mse=mse,psnr=-10*np.log10(max(mse,1e-12))))
            print(f'progress: render {j+1}/{len(manifest["frames"])}',flush=True)
    write_json(out/'reconstruction_metrics.json',dict(dataset=manifest['log'],protocol=manifest['protocol'],
        aggregation='per-image training-view mean',images=len(metrics),mean_psnr=float(np.mean([m['psnr'] for m in metrics])),per_image=metrics))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--method',choices=['splatad','omnire_nosmpl','street_gs'],required=True)
    ap.add_argument('--config',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--upstream',required=True,type=Path)
    a=ap.parse_args();config=json.loads(a.config.read_text());out=a.out
    print('config: '+json.dumps(dict(method=a.method,out=str(out),config=config)),flush=True)
    if config.get('nuplan_devkit_root'):
        sys.path.insert(0, config['nuplan_devkit_root'])
    out.mkdir(parents=True,exist_ok=True)
    data=out/'processed'/'openscene_pilot'
    export(Path(config['reconstruction_metadata']),Path(config['openscene_root'])/'sensor_blobs'/config['data_split'],data,config['reconstruction_start'],40)
    env=dict(os.environ)
    code=Path(__file__).resolve().parent
    if a.method=='splatad':
        repo=a.upstream/'neurad-studio'
        spec=dict(upstream_config=str(out/'trained/openscene/splatad/run/config.yml'))
        command=[sys.executable,str(code/'train_splatad_openscene.py'),'--data',str(data),'--out',str(out/'trained'),'--steps',str(config['splatad_steps'])]
    else:
        command,spec,repo=configure(a.method,data,a.upstream,out,config['gs_steps'])
    env['PYTHONPATH']=str(code)+os.pathsep+str(repo)
    sys.path.insert(0,str(repo))
    write_json(out/'renderer.json',spec)
    run_logged(command,out/'native_train.log',env)
    render_all(a.method,spec,data,out)
    # Evaluate only target frames wholly contained in the reconstructed clip.
    import yaml
    filt=yaml.safe_load(Path(config['scene_filter_yaml']).read_text())
    manifest=json.loads((data/'manifest.json').read_text())
    filt.update(log_names=[manifest['log']],tokens=None,frame_interval=12)
    filter_path=out/'eval_filter.yaml';filter_path.write_text(yaml.safe_dump(filt))
    selected=[f['token'] for f in manifest['frames'][3:-8:12]]
    write_json(out/'eval_tokens.json',selected)
    evaluation=copy.deepcopy(config)
    evaluation.update(device='cuda',scene_filter_yaml=str(filter_path),tokens_file=str(out/'eval_tokens.json'),limit_tokens=None,
        arms=['identity','copy_last_bev',a.method],render_roots={a.method:str(out/'renders')},dataset='OpenScene mini / 20s reconstructed clip / front3 pilot')
    write_json(out/'eval_config.json',evaluation)
    run_logged([config.get('eval_python',sys.executable),str(code/'eval_open_loop.py'),'--config',str(out/'eval_config.json'),'--out',str(out/'open_loop')],out/'open_loop.log',env)
    write_json(out/'DONE.json',dict(method=a.method,training=True,renders=True,open_loop_stats=str(out/'open_loop/stats.json')))


if __name__=='__main__':main()
