"""Full-dataset timing without additional forward calls or RNG changes."""
import contextlib
import json
from pathlib import Path
import time


class LatencyRecorder:
    def __init__(self, config, out, torch):
        self.config, self.out, self.torch = config, Path(out), torch
        self.rows = []
        self.token = None
        self.index = 0
        self.arm, self.gap = 'shared', 0
        self.warmup = int(config.get('timing_warmup_tokens', 20))
        self.started = time.perf_counter()
        self.token_started = None

    def begin_token(self, token, index):
        self.torch.cuda.synchronize()
        self.token, self.index, self.arm, self.gap = token, index, 'shared', 0
        self.token_started = time.perf_counter()

    def set_context(self, arm, gap):
        self.arm, self.gap = arm, int(gap)

    def row(self, component, wall_ms, event_ms=None):
        self.rows.append(dict(token=self.token, index=self.index, arm=self.arm, gap_frames=self.gap,
                              component=component, wall_ms=wall_ms, cuda_event_ms=event_ms,
                              warmup=self.index <= self.warmup))

    @contextlib.contextmanager
    def span(self, component, cuda=True):
        if cuda:
            self.torch.cuda.synchronize()
            begin = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            begin.record()
        started = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                end.record()
                end.synchronize()
            self.row(component, (time.perf_counter()-started)*1000,
                     begin.elapsed_time(end) if cuda else None)

    def call(self, component, fn, *args, cuda=True, **kwargs):
        with self.span(component, cuda):
            return fn(*args, **kwargs)

    def end_token(self):
        self.torch.cuda.synchronize()
        self.arm, self.gap = 'shared', 0
        self.row('token_total_all_arms_and_metrics', (time.perf_counter()-self.token_started)*1000)

    def attach(self, agent):
        for method, name in [('_extract_bev_features','image_bev_encoder'),
                             ('simulate_next_bev','flowsim_transition'),
                             ('predict_from_bev_features','planning_detection_heads'),
                             ('infer_bev_semantic_from_fused_features','semantic_head')]:
            original = getattr(agent, method)
            def wrapped(*args, _fn=original, _name=name, **kwargs):
                return self.call(_name, _fn, *args, **kwargs)
            setattr(agent, method, wrapped)

    def finish(self):
        value = dict(status='complete', config=self.config, rows=self.rows,
                     device=self.torch.cuda.get_device_name(), torch_version=self.torch.__version__,
                     precision='FP32 original quality-eval path; no AMP/quantization/compilation',
                     warmup_tokens_per_worker=self.warmup, worker_eval_elapsed_s=time.perf_counter()-self.started,
                     timing='synchronized wall time and CUDA-event interval; exactly one original forward per sample',
                     note='CUDA event interval includes host gaps, not kernel busy time. Nested component times are not additive.')
        p = self.out/'latency.json'
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(value)+'\n')
        tmp.replace(p)
        print('[TIMING_COMPLETE]', len(self.rows), 'rows', str(p), flush=True)
