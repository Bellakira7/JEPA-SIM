"""OpenScene/LTF open-loop comparison, adapted from eval_template.py.

Reuses FlowSim's official loaders, frozen-head replay, and agreement functions.
All evaluated arms share tokens, gaps, target status and the frozen LTF seed 0.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

# Environment constraints retained from experiment-protocol/eval_template.py.
for _key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
    os.environ.pop(_key, None)
from protocol import FLOWSIM_COMMIT, PROTOCOL, AR_PROTOCOL, paired_summary, sha256_file, write_json


def preflight(config):
    root, data = Path(config['flowsim_root']), Path(config['openscene_root'])
    arms = config['arms']
    allowed = {'identity', 'copy_last_bev', 'flowsim_vh','featsim_ltf'} | set(config.get('render_roots', {}))
    checks = dict(protocol=config.get('protocol') in (PROTOCOL, AR_PROTOCOL),
        gaps=bool(config['gaps']) and len(set(config['gaps'])) == len(config['gaps']) and all(g in (1,2,3) for g in config['gaps']),
        arms=bool(arms) and len(set(arms)) == len(arms) and set(arms) <= allowed,
        identity_gate='identity' in arms,
        ltf_checkpoint=Path(config['ltf_checkpoint']).is_file(),
        flowsim_checkpoint='flowsim_vh' not in arms or bool(config.get('flowsim_checkpoint')) and Path(config['flowsim_checkpoint']).is_file(),
        featsim_checkpoint='featsim_ltf' not in arms or bool(config.get('featsim_checkpoint')) and Path(config['featsim_checkpoint']).is_file(),
        scene_filter=Path(config['scene_filter_yaml']).is_file(),
        logs=(data/'navsim_logs'/config['data_split']).is_dir(),
        sensors=(data/'sensor_blobs'/config['data_split']).is_dir(), maps=(data/'maps').is_dir())
    try:
        commit = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        commit = None
    snapshot_errors = []
    manifest = root / 'UPSTREAM_REVISION.json'
    if manifest.is_file():
        try:
            pinned = json.loads(manifest.read_text())
            for relative, expected_hash in pinned['files'].items():
                path = Path(relative)
                if path.is_absolute() or '..' in path.parts or sha256_file(root/path) != expected_hash:
                    snapshot_errors.append(relative)
            if pinned['commit'] == FLOWSIM_COMMIT and pinned['files'] and not snapshot_errors:
                commit = pinned['commit']
            else:
                commit = None
        except (OSError, ValueError, KeyError) as exc:
            snapshot_errors.append(repr(exc))
            commit = None
    checks['flowsim_version'] = commit == FLOWSIM_COMMIT
    versions, errors = {'python':platform.python_version()}, {}
    for name in ('torch','timm','numpy','nuplan','pytorch_lightning','hydra','omegaconf','cv2',
                 'geopandas','pyogrio','navsim.common.dataloader','navsim.agents.flowsim.flowsim_variational_gap_agent',
                 'scripts.domain_gap.variational_gap_onestep_eval'):
        try:
            module = importlib.import_module(name)
            checks['module/'+name] = True
            versions[name] = getattr(module,'__version__',None)
        except Exception as exc:
            checks['module/'+name] = False
            errors[name] = repr(exc)
    return dict(status='ready' if all(checks.values()) else 'missing_prerequisites',
                checks=checks,flowsim_commit=commit,flowsim_snapshot_errors=snapshot_errors,config=config,versions=versions,import_errors=errors,
                note='Dependency/data audit; no model performance result.')


def load_agent(config, torch):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from unittest.mock import patch
    import timm
    cfg = OmegaConf.load(Path(config['flowsim_root'])/'navsim/planning/script/config/common/agent/flowsim_variational_gap_agent.yaml')
    cfg.pretrained_bev_model_path = config['ltf_checkpoint']
    cfg.checkpoint_path = config.get('flowsim_checkpoint') if 'flowsim_vh' in config['arms'] else None
    cfg.config.latent = True
    create = timm.create_model
    def no_imagenet_download(*args, **kwargs):
        kwargs['pretrained'] = False
        return create(*args, **kwargs)
    with patch.object(timm,'create_model',no_imagenet_download):
        agent = instantiate(cfg)
    # Initialize from strict weights once; upstream initialize() permits partial
    # teacher loading and deserializes the multi-GB checkpoint a second time.
    checkpoint = torch.load(config['ltf_checkpoint'],map_location='cpu',weights_only=False)
    state = checkpoint.get('state_dict',checkpoint)
    prefix = 'agent._transfuser_model.'
    state = {k[len(prefix):] if k.startswith(prefix) else k:v for k,v in state.items()}
    agent._bev_model.load_state_dict(state,strict=True)
    print(f'weights: strict LTF load ({len(state)} tensors)',flush=True)
    del checkpoint, state
    if 'flowsim_vh' in config['arms']:
        checkpoint = torch.load(config['flowsim_checkpoint'],map_location='cpu',weights_only=False)
        prefix = 'agent._flow_sim.'
        state = {k[len(prefix):]:v for k,v in checkpoint['state_dict'].items() if k.startswith(prefix)}
        if not state:
            raise ValueError('Checkpoint contains no FlowSim transition weights')
        agent._flow_sim.load_state_dict(state,strict=True)
        print(f'weights: strict FlowSim VH load ({len(state)} tensors)',flush=True)
        del checkpoint, state
    return agent.requires_grad_(False).to(config['device']).eval()


def rendered_input(inputs, root):
    import numpy as np
    from PIL import Image
    current, updates = inputs.cameras[-1], {}
    for name in ('cam_l0','cam_f0','cam_r0'):
        cam = getattr(current,name)
        relative = Path(cam.camera_path)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Invalid camera path: {relative}')
        with Image.open(Path(root)/relative) as image:
            pixels = np.asarray(image.convert('RGB'))
        if pixels.shape != cam.image.shape:
            raise ValueError(f'Render resolution differs from real image: {relative}')
        updates[name] = replace(cam,image=pixels)
    cameras = list(inputs.cameras)
    cameras[-1] = replace(current,**updates)
    return replace(inputs,cameras=cameras)


def evaluate(config, out, torch):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from scripts.domain_gap.variational_gap_onestep_eval import build_eval_sensor_config
    from scripts.domain_gap.aux_task_agreement_eval import semseg_agreement, detection_agreement, planning_agreement
    agent = load_agent(config,torch)
    scene_filter = instantiate(OmegaConf.load(config['scene_filter_yaml']))
    sequence = config['protocol'] == AR_PROTOCOL
    if sequence:
        if config.get('history_frames') != 4 or config.get('step_seconds') != .5:
            raise ValueError('AR-v2 requires four-frame history and 0.5 s prediction steps')
        scene_filter.num_history_frames = 4+max(config['gaps'])
    if getattr(scene_filter,'include_synthetic_scenes',False):
        raise ValueError('This protocol requires original logged scenes')
    sensor_gaps = list(range(1, scene_filter.num_history_frames)) if sequence else config['gaps']
    sensors = build_eval_sensor_config(scene_filter.num_history_frames,sensor_gaps,latent=True)
    root = Path(config['openscene_root'])
    loader = SceneLoader(data_path=root/'navsim_logs'/config['data_split'],
        original_sensor_path=root/'sensor_blobs'/config['data_split'],scene_filter=scene_filter,sensor_config=sensors)
    tokens = sorted(loader.tokens_stage_one)
    if sequence:
        from sequence_inference import select_tokens
        tokens, audit = select_tokens(loader, config)
        write_json(out/'sequence_token_audit.json', audit)
    if config.get('tokens_file'):
        requested = json.loads(Path(config['tokens_file']).read_text())
        if len(set(requested))!=len(requested) or not set(requested)<=set(tokens):
            raise ValueError('Requested token subset has duplicates or absent scenes')
        tokens = requested
    if config.get('limit_tokens'):
        tokens = tokens[:int(config['limit_tokens'])]
    if not tokens:
        raise ValueError('No available tokens selected')
    write_json(out/'tokens.json',tokens)
    builder, targets_builder = agent.get_feature_builders()[0], agent.get_target_builders()[0]
    records, identity_checks = [], []
    artifacts = dict(ltf_checkpoint_sha256=sha256_file(config['ltf_checkpoint']),
                     scene_filter_sha256=sha256_file(config['scene_filter_yaml']))
    if 'flowsim_vh' in config['arms']:
        artifacts['flowsim_checkpoint_sha256'] = sha256_file(config['flowsim_checkpoint'])
    if 'featsim_ltf' in config['arms']:
        from ltf_features import load_predictor,extract_camera_stages,replay_camera_stages
        from scene_layout import condition_layout,action_between
        predictor,mean,std,feat_checkpoint=load_predictor(config['featsim_checkpoint'],config['device'])
        if config.get('expected_feature_cut') and feat_checkpoint['feature_cut']!=config['expected_feature_cut']:
            raise ValueError('Registered evaluation feature cut differs')
        if sequence != (feat_checkpoint.get('prediction_mode') == 'sliding_ar'):
            raise ValueError('Checkpoint prediction mode differs from evaluation protocol')
        if sequence and (feat_checkpoint.get('history_frames') != 4 or feat_checkpoint.get('step_seconds') != .5):
            raise ValueError('Checkpoint sequence context/cadence mismatch')
        if feat_checkpoint['ltf_sha256']!=artifacts['ltf_checkpoint_sha256']:
            raise ValueError('FeatSim teacher differs from evaluated LTF')
        artifacts['featsim_checkpoint_sha256']=sha256_file(config['featsim_checkpoint'])
        if config.get('expected_checkpoint_stage') and feat_checkpoint.get('stage') != config['expected_checkpoint_stage']:
            raise ValueError('Checkpoint training stage differs from registered evaluation')
        if config.get('expected_checkpoint_step') is not None and feat_checkpoint.get('step') != config['expected_checkpoint_step']:
            raise ValueError('Checkpoint optimizer step differs from registered evaluation')
        if config.get('expected_checkpoint_sha256') and artifacts['featsim_checkpoint_sha256'] != config['expected_checkpoint_sha256']:
            raise ValueError('Registered evaluation checkpoint changed')
        if 'expected_warp_levels' in config and list(feat_checkpoint['predictor_kwargs'].get('warp_levels', [])) != config['expected_warp_levels']:
            raise ValueError('Checkpoint motion compensation differs from registered evaluation')
    batch = lambda f: {k:v.unsqueeze(0).to(config['device']) for k,v in f.items()}
    for i, token in enumerate(tokens,1):
        try:
            scene = loader.get_scene_from_token(token)
            inputs = scene.get_agent_input()
            feature_cache, condition_cache = {}, {}
            base = batch(builder.compute_features_with_scene_gap(inputs,scene,config['gaps'][0]))
            with torch.inference_mode():
                real_bev = agent._extract_bev_features(base['camera_feature'])
                reference = agent._bev_model(dict(camera_feature=base['camera_feature'],status_feature=base['status_feature']))
                ref_sem = reference['bev_semantic_map']
                identity = agent.predict_from_bev_features(real_bev,base['status_feature'])
                identity_sem = agent.infer_bev_semantic_from_fused_features(real_bev)
                for key in ('trajectory','agent_states','agent_labels'):
                    torch.testing.assert_close(identity[key],reference[key],atol=config['identity_tolerance'],rtol=0)
                torch.testing.assert_close(identity_sem,ref_sem,atol=config['identity_tolerance'],rtol=0)
                identity_checks.append(dict(token=token,
                    **{key:float((identity[key]-reference[key]).abs().max())
                       for key in ('trajectory','agent_states','agent_labels')},
                    bev_semantic_map=float((identity_sem-ref_sem).abs().max())))
                gt = torch.as_tensor(scene.get_future_trajectory(num_trajectory_frames=8).poses,device=config['device'])
                for gap in config['gaps']:
                    features = batch(builder.compute_features_with_scene_gap(inputs,scene,gap))
                    previous = agent._extract_bev_features(features['camera_feature_past'])
                    for arm in config['arms']:
                        if arm == 'identity':
                            pred, semantic = identity, identity_sem
                        else:
                            if arm == 'copy_last_bev':
                                z = previous
                            elif arm == 'flowsim_vh':
                                targets = batch(targets_builder.compute_targets_for_gap(scene,gap))
                                salt = int.from_bytes(hashlib.sha256(f'{token}:{gap}'.encode()).digest()[:4],'big')
                                torch.manual_seed(config['seed']+salt)
                                z = agent.simulate_next_bev(prev_bev=previous,
                                    agent_states=targets.get('agent_states_assoc',targets['agent_states']),
                                    agent_labels=targets['agent_labels'],bev_semantic_map=targets['bev_semantic_map'],
                                    status_feature=features['status_feature'])
                            elif arm == 'featsim_ltf':
                                raw=loader.scene_frames_dicts[token]
                                if feat_checkpoint['feature_cut']=='ltf/final_fused_bev_v1':
                                    if not sequence:raise ValueError('BEV predictor requires AR-v2')
                                    from bev_features import predict_bev
                                    z=predict_bev(agent,predictor,builder,inputs,scene,raw,gap,mean,std,
                                        config['device'],feature_cache,condition_cache)
                                elif sequence:
                                    from sequence_inference import predict_stages
                                    decoded = predict_stages(agent, predictor, builder, inputs, scene, raw, gap,
                                        mean, std, config['device'], feature_cache, condition_cache)
                                else:
                                    stages,_=extract_camera_stages(agent,features['camera_feature_past'])
                                    normalized=[((f-m)/s)[:,None,None] for f,m,s in zip(stages,mean,std)]
                                    action=action_between(raw,gap).to(config['device'])[None,None]
                                    layout=condition_layout(scene,raw,gap).to(config['device'])[None,None,None]
                                    generated=predictor(normalized,action,layout)
                                    decoded=[p[:,0,0]*s+m for p,m,s in zip(generated,mean,std)]
                                if feat_checkpoint['feature_cut']!='ltf/final_fused_bev_v1':
                                    z=replay_camera_stages(agent,decoded)
                            else:
                                # Target-view renders are the same control at
                                # each gap; they do not consume an anchor image.
                                changed = rendered_input(inputs,config['render_roots'][arm])
                                rendered = batch(builder.compute_features_with_scene_gap(changed,scene,gap))
                                z = agent._extract_bev_features(rendered['camera_feature'])
                            pred = agent.predict_from_bev_features(z,features['status_feature'])
                            semantic = agent.infer_bev_semantic_from_fused_features(z)
                        for key in ('trajectory','agent_states','agent_labels'):
                            if not torch.isfinite(pred[key]).all():
                                raise ValueError(f'Nonfinite prediction: {arm} {key}')
                        if not torch.isfinite(semantic).all():
                            raise ValueError(f'Nonfinite semantic prediction: {arm}')
                        metrics = {f'planning/{k}':v for k,v in planning_agreement(reference['trajectory'],pred['trajectory']).items()}
                        metrics.update({f'detection/{k}':v for k,v in detection_agreement(reference,pred).items()})
                        metrics.update({f'semseg/{k}':v for k,v in semseg_agreement(ref_sem,semantic).items()})
                        errors = torch.linalg.vector_norm(pred['trajectory'][0,:,:2]-gt[:,:2],dim=-1)
                        metrics['gt/ade_4s_m'] = errors.mean().item()
                        for seconds in (1,2,3,4):
                            metrics[f'gt/fde_{seconds}s_m'] = errors[2*seconds-1].item()
                        metrics = {k:float(v) if math.isfinite(float(v)) else None for k,v in metrics.items()}
                        records.append(dict(token=token,log_name=scene.scene_metadata.log_name,arm=arm,gap_frames=gap,metrics=metrics))
            print(f'progress: {i}/{len(tokens)} token={token}',flush=True)
        except Exception as exc:
            write_json(out/'failure.json',dict(token=token,error=repr(exc),completed_tokens=i-1,config=config))
            raise
    finalize_evaluation(config, out, records, identity_checks, artifacts)


def finalize_evaluation(config, out, records, identity_checks, artifacts):
    """Shared writer for serial evaluation and exact aggregation of GPU shards."""
    tokens = {row['token'] for row in records}
    if not tokens or len(identity_checks) != len(tokens) or {row['token'] for row in identity_checks} != tokens:
        raise ValueError('Identity checks must cover every evaluated token exactly once')
    if any(not math.isfinite(row[key]) or row[key] > config['identity_tolerance'] for row in identity_checks
           for key in ('trajectory','agent_states','agent_labels','bev_semantic_map')):
        raise ValueError('Evaluation identity gate failed')
    # Validate pairing before writing a success result.
    summary = paired_summary(records,config['arms'],config['gaps'],config['dataset'],protocol=config['protocol'])
    write_json(out/'records.json',records)
    write_json(out/'identity_checks.json',dict(atol=config['identity_tolerance'],rtol=0,
        max_absolute_error={key:max(row[key] for row in identity_checks)
            for key in ('trajectory','agent_states','agent_labels','bev_semantic_map')},
        per_token=identity_checks))
    summary.update(status='complete',artifacts=artifacts,config=config,
        note='Agreement: LTF real-image output. gt/*: recorded human trajectory. No PDM/EPDMS; no simulation timing claim.')
    write_json(out/'stats.json',summary)
    lines = [f"{config['dataset']} | {config['protocol']} | per_token_mean | n={len(tokens)} | units: m",'',
        '| Arm | Gap (s) | Plan ADE agreement | Plan FDE agreement | Human ADE (4s) |','|---|---:|---:|---:|---:|']
    for row in summary['table']:
        metrics = row['per_token_mean']
        vals = [metrics[k] for k in ('planning/ade_agreement','planning/fde_agreement','gt/ade_4s_m')]
        lines.append(f"| {row['arm']} | {row['gap_seconds']} | "+' | '.join('—' if v is None else f'{v:.6f}' for v in vals)+' |')
    (out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'done: {out}',flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--preflight',action='store_true')
    args = ap.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    os.environ['NAVSIM_DEVKIT_ROOT'] = config['flowsim_root']
    os.environ['OPENSCENE_DATA_ROOT'] = config['openscene_root']
    os.environ['NUPLAN_MAPS_ROOT'] = str(Path(config['openscene_root'])/'maps')
    os.environ['NUPLAN_MAP_VERSION'] = 'nuplan-maps-v1.0'
    sys.path.insert(0,config['flowsim_root'])
    if not args.preflight and any((args.out/name).exists() for name in ('stats.json','records.json','failure.json')):
        raise FileExistsError('Use a fresh output directory for each evaluation attempt')
    print('config: '+json.dumps(config,ensure_ascii=False),flush=True)
    audit = preflight(config)
    write_json(args.out/'preflight.json',audit)
    if args.preflight:
        print(json.dumps(audit,indent=2),flush=True)
        return 0 if audit['status']=='ready' else 2
    if audit['status']!='ready':
        raise RuntimeError('Missing prerequisites; see preflight.json')
    import torch
    torch.manual_seed(config['seed'])
    try:
        evaluate(config,args.out,torch)
    except Exception as exc:
        if not (args.out/'failure.json').exists():
            write_json(args.out/'failure.json',dict(error=repr(exc),config=config))
        raise
    return 0


if __name__=='__main__':
    raise SystemExit(main())
