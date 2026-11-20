"""Changed critical paths: legal context, causal rollout, epoch coverage and optimizer resume."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'core'))
import torch
from sequence_data import EpochBatchSampler, aligned_steps, make_clips
from train_sequence import forecast, losses, load_batch, initialize_predictor
from ltf_features import build_predictor


def small_model(warp_levels=()):
    return build_predictor(dict(level_grids=((8,8),(4,4),(2,2),(1,1)), level_pools=(8,4,2,1),
        n_cams=1, in_chans=(2,3,4,5), embed_dim=48, depth=2, num_heads=2,
        action_dim=4, enable_layout=True, assoc_layout=True, residual=True, warp_levels=warp_levels,
        level_strides=(4,8,16,32)))


def sample(frames=7):
    features = [torch.randn(1,frames,1,c,h,h) for c,h in [(2,8),(3,4),(4,2),(5,1)]]
    actions = torch.randn(1,frames-1,4)
    actions[...,3] = .5
    return features, actions, torch.rand(1,frames-1,1,6,8,8)


class SequenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(13)

    def test_clip_index_breaks_on_missing_edge_timestamp_and_log(self):
        pairs=[dict(log='a',source=str(i),target=str(i+1),condition=f'{i+1}_1.pt',gap=1) for i in range(8)]
        timestamps={('a',str(i)):i*500000 for i in range(9)}
        clips,_=make_clips(pairs,5,timestamps)
        self.assertEqual(len(clips),5)
        self.assertEqual(clips[0]['frames'],['0','1','2','3','4'])
        missing=[p for p in pairs if p['source']!='3']
        self.assertEqual(len(make_clips(missing,5,timestamps)[0]),1)
        shifted={k:v+(200000 if int(k[1])>=4 else 0) for k,v in timestamps.items()}
        self.assertEqual(len(make_clips(pairs,5,shifted)[0]),1)
        other=copy.deepcopy(pairs)
        other[3]['log']='b'
        self.assertEqual(len(make_clips(other,5,timestamps)[0]),1)

    def test_epoch_coverage_and_exact_sampler_resume(self):
        sampler=EpochBatchSampler(19,8,7)
        batches=[sampler.next() for _ in range(3)]
        self.assertEqual(set(sum(batches,[])[:19]),set(range(19)))
        resumed=EpochBatchSampler(19,8,7)
        resumed.load_state_dict(sampler.state_dict())
        self.assertEqual([sampler.next() for _ in range(5)],[resumed.next() for _ in range(5)])
        self.assertEqual(aligned_steps(25330,8,'tf'),12000)
        self.assertEqual(aligned_steps(20430,8,'ro'),6000)

    def test_real_predictor_uses_earlier_context_but_not_future(self):
        model=small_model().eval()
        with torch.no_grad():
            for projection in model.proj:
                projection.weight.normal_(std=.02)
        features,actions,layout=sample(5)
        history=[x[:,:4].clone().requires_grad_() for x in features]
        prediction=model(history,actions,layout)
        prediction[0][:,-1].square().mean().backward()
        self.assertGreater(history[0].grad[:,:3].abs().sum().item(),0)
        changed=[x.detach().clone() for x in history]
        for x in changed:
            x[:,2:]+=100
        changed_actions=actions.clone();changed_actions[:,2:]+=100
        changed_layout=layout.clone();changed_layout[:,2:]+=100
        later=model(changed,changed_actions,changed_layout)
        for original,new in zip(prediction,later):
            torch.testing.assert_close(original[:,:2],new[:,:2],atol=1e-6,rtol=1e-6)
        self.assertFalse(any('flow_proj' in name for name in model.state_dict()))

    def test_rollout_feeds_predictions_and_has_no_future_feature_leak(self):
        class RecordingPredictor(torch.nn.Module):
            L=4
            def __init__(self):
                super().__init__();self.inputs=[]
            def forward(self,features,actions,layout):
                self.inputs.append(features[0].clone())
                return [x+1 for x in features]
        model=RecordingPredictor()
        features=[torch.arange(7.).view(1,7,1,1,1,1) for _ in range(4)]
        for x in features:
            x[:,4:]=999
        actions=torch.zeros(1,6,4);layout=torch.zeros(1,6,1,6,8,8)
        losses(model,features,actions,layout,'ro',3)
        self.assertEqual([x.flatten().tolist() for x in model.inputs],[[0,1,2,3],[1,2,3,4],[2,3,4,5]])
        self.assertFalse(model.inputs[1].requires_grad)
        with self.assertRaises(ValueError):
            forecast(model,[x[:,:3] for x in features],actions,layout,3)

    def test_clip_loading_dense_and_rollout_backward_resume(self):
        model=small_model()
        batch=sample(7)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            features,actions,layout=batch
            for i in range(7):
                torch.save([x[0,i,0] for x in features],root/f'f{i}.pt')
            for i in range(6):
                torch.save(dict(action=actions[0,i],layout=layout[0,i,0]),root/f'c{i}.pt')
            clip=dict(frames=[f'f{i}' for i in range(7)],conditions=[f'c{i}.pt' for i in range(6)])
            means=[torch.zeros(1,1,1,c,1,1) for c in (2,3,4,5)]
            stds=[torch.ones_like(x) for x in means]
            loaded=load_batch([clip],root,means,stds,'cpu')
            for a,b in zip(loaded[0],features):torch.testing.assert_close(a,b)
            opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
            def step(m,o,stage):
                o.zero_grad();loss,_=losses(m,*loaded,stage,3);loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
                o.step()
            step(model,opt,'tf')
            saved=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()))
            step(model,opt,'ro')
            resumed=small_model();resumed.load_state_dict(saved['model'],strict=True)
            resumed_opt=torch.optim.AdamW(resumed.parameters(),lr=1e-3);resumed_opt.load_state_dict(saved['optimizer'])
            step(resumed,resumed_opt,'ro')
            for a,b in zip(model.parameters(),resumed.parameters()):torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_evaluation_reads_only_four_pre_anchor_images(self):
        from sequence_inference import predict_stages
        class Builder:
            def __init__(self):self.reads=[]
            def _get_camera_feature(self, inputs, idx):
                self.reads.append(idx);return torch.tensor([float(idx)])
        class Predictor(torch.nn.Module):
            L=4
            def forward(self,features,actions,layout):return [x+1 for x in features]
        builder=Builder()
        scene=SimpleNamespace(scene_metadata=SimpleNamespace(num_history_frames=7))
        norm=[torch.zeros(1,1,1,1) for _ in range(4)]
        def extract(agent,image):
            return [image.reshape(1,1,1,1).expand(1,1,1,1) for _ in range(4)],None
        with patch('ltf_features.extract_camera_stages',extract), \
             patch('scene_layout.action_between',return_value=torch.tensor([1.,0.,0.,.5])), \
             patch('scene_layout.condition_layout',return_value=torch.zeros(6,8,8)):
            result=predict_stages(None,Predictor(),builder,None,scene,[None]*7,3,
                                  norm,[x+1 for x in norm],'cpu',{}, {})
        self.assertEqual(builder.reads,[0,1,2,3])
        self.assertEqual(result[0].item(),6.)

    def test_warp_pair_shares_initial_weights_and_has_flow_gradients(self):
        kwargs=dict(level_grids=((8,8),(4,4),(2,2),(1,1)),level_pools=(8,4,2,1),
            n_cams=1,in_chans=(2,3,4,5),embed_dim=48,depth=2,num_heads=2,
            action_dim=4,enable_layout=True,assoc_layout=True,residual=True,level_strides=(4,8,16,32))
        with patch('ltf_features.predictor_kwargs',return_value=kwargs):
            baseline,_=initialize_predictor(dict(seed=13,warp_levels=[]))
            rng=torch.get_rng_state().clone()
            warped,_=initialize_predictor(dict(seed=13,warp_levels=[0,1]))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for name,value in baseline.state_dict().items():
            torch.testing.assert_close(value,warped.state_dict()[name],atol=0,rtol=0)
        loss,_=losses(warped,*sample(5),'tf')
        loss.backward()
        for level in (0,1):
            gradient=warped.flow_proj[str(level)].weight.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(),0)


if __name__ == '__main__':
    if '--ddp-smoke' in sys.argv:
        import torch.distributed as dist
        dist.init_process_group('gloo')
        try:
            torch.set_num_threads(1)
            torch.manual_seed(13)
            model=small_model((0,1) if '--warp-smoke' in sys.argv else ())
            model.use_activation_checkpointing=True
            net=torch.nn.parallel.DistributedDataParallel(model)
            opt=torch.optim.AdamW(net.parameters(),lr=1e-3)
            torch.manual_seed(100+dist.get_rank())
            batch=sample(12)
            for stage,horizon in [('tf',1),('ro',8)]:
                opt.zero_grad()
                loss,_=losses(net,*batch,stage,horizon)
                loss.backward();opt.step()
                flat=torch.cat([p.detach().flatten() for p in net.parameters()])
                gathered=[torch.empty_like(flat) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered,flat)
                for other in gathered:torch.testing.assert_close(flat,other,atol=0,rtol=0)
            if dist.get_rank()==0:print('DDP_2CPU_TF_RO8_GRADIENT_SYNC_PASSED',dict(warp_levels=model.warp_levels),flush=True)
        finally:
            dist.destroy_process_group()
    else:
        unittest.main()
