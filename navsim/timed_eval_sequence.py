"""Instrument the frozen AR-v2 evaluator without extra model forward calls."""
import os
import sys
from pathlib import Path
import inspect
import json

CODE = Path(__file__).resolve().parent
sys.path.insert(0,str(CODE/'core'));sys.path.insert(0,str(CODE))
# Import first: this preserves original torchrun rank before evaluator clears it.
import eval_sequence as seq
import eval_open_loop as ev
import train_sequence as train
from latency_recorder import LatencyRecorder


def once(s, old, new):
    assert s.count(old)==1, (old,s.count(old))
    return s.replace(old,new)


def install():
    source = inspect.getsource(ev.evaluate)
    source = once(source,'    agent = load_agent(config,torch)',
                  '    timing = LatencyRecorder(config,out,torch)\n    agent = load_agent(config,torch)\n    timing.attach(agent)')
    source = once(source,'        try:\n            scene = loader.get_scene_from_token(token)',
                  "        timing.begin_token(token,i)\n        try:\n            scene = timing.call('scene_decode', loader.get_scene_from_token, token, cuda=False)")
    source = once(source,'            inputs = scene.get_agent_input()',
                  "            inputs = timing.call('agent_input_prepare',scene.get_agent_input,cuda=False)")
    source = once(source,"            base = batch(builder.compute_features_with_scene_gap(inputs,scene,config['gaps'][0]))",
                  "            base = timing.call('reference_input_prepare',lambda: batch(builder.compute_features_with_scene_gap(inputs,scene,config['gaps'][0])))")
    source = once(source,"                reference = agent._bev_model(dict(camera_feature=base['camera_feature'],status_feature=base['status_feature']))",
                  "                reference = timing.call('reference_full_ltf',agent._bev_model,dict(camera_feature=base['camera_feature'],status_feature=base['status_feature']))")
    source = once(source,"                    features = batch(builder.compute_features_with_scene_gap(inputs,scene,gap))",
                  "                    timing.set_context('shared',gap)\n                    features = timing.call('gap_input_prepare',lambda: batch(builder.compute_features_with_scene_gap(inputs,scene,gap)))")
    start = source.index("                        if arm == 'identity':")
    end = source.index("                        for key in ('trajectory','agent_states','agent_labels'):",start)
    block=source[start:end]
    replacement="                        timing.set_context(arm,gap)\n                        with timing.span('arm_pipeline'):\n"+''.join('    '+line if line.strip() else line for line in block.splitlines(True))
    source=source[:start]+replacement+source[end:]
    source=once(source,"            print(f'progress: {i}/{len(tokens)} token={token}',flush=True)",
                "            timing.end_token()\n            print(f'progress: {i}/{len(tokens)} token={token}',flush=True)")
    source=once(source,'    finalize_evaluation(config, out, records, identity_checks, artifacts)',
                '    finalize_evaluation(config, out, records, identity_checks, artifacts)\n    timing.finish()')
    ev.LatencyRecorder=LatencyRecorder
    original_forecast=train.forecast
    # Set around each call so the original forecast is restored even on failure.
    source=once(source,"        timing.begin_token(token,i)",
                "        timing.begin_token(token,i)\n        train.forecast = lambda *args, **kwargs: timing.call('bev_forecast', original_forecast, *args, **kwargs)")
    ev.train, ev.original_forecast=train, original_forecast
    exec(compile(source,str(CODE/'timed_eval_sequence.py')+':instrumented_eval','exec'),ev.__dict__)
    worker=inspect.getsource(seq.worker)
    worker=once(worker,"    if (shard/'stats.json').exists():",
                "    if (shard/'stats.json').exists() and (shard/'latency.json').exists():")
    exec(compile(worker,str(CODE/'timed_eval_sequence.py')+':instrumented_worker','exec'),seq.__dict__)


def main():
    install()
    if '--validate' in sys.argv:
        print('[TIMING_ADAPTER_COMPILE_PASS]',flush=True);return
    original_config=seq.evaluation_config
    def config_with_timing(config,out):
        return {**original_config(config,out),'timing_warmup_tokens':config.get('timing_warmup_tokens',20)}
    seq.evaluation_config=config_with_timing
    seq.main()


if __name__=='__main__': main()
