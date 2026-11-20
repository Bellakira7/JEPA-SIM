"""GPU-sharded preparation/evaluation, adapted from eval_template.py.

Keep platform variables out of independent workers, log their configuration and
progress, and publish completion only after strict whole-run aggregation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
    os.environ.pop(key, None)

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / 'core'))
from parallel_helpers import expected_tokens, merge_cache, merge_evaluation, visible_gpu_identifiers
from protocol import write_json


def run_workers(commands, out, devices):
    workers = []
    try:
        for rank, command in enumerate(commands):
            stream = (out / f'worker_{rank:02d}.log').open('a')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[rank])
            try:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
            except BaseException:
                stream.close()
                raise
            workers.append((process, stream))
            print('WORKER_START', rank, 'pid', process.pid, json.dumps(command), flush=True)
        while True:
            states = [process.poll() for process, _ in workers]
            failed = next((i for i, state in enumerate(states) if state not in (None, 0)), None)
            if failed is not None:
                tail = (out / f'worker_{failed:02d}.log').read_text(errors='replace').splitlines()[-30:]
                raise RuntimeError(f'GPU worker {failed} failed: ' + '\n'.join(tail))
            if all(state == 0 for state in states):
                break
            print('PARALLEL_PROGRESS', json.dumps(states), flush=True)
            time.sleep(20)
    finally:
        # Only terminate children created by this invocation on failure/interruption.
        for process, stream in workers:
            if process.poll() is None:
                process.terminate()
        for process, stream in workers:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stream.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=('prepare', 'eval'))
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--gpus', type=int, required=True)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    worker_key = 'preparation_workers' if args.phase == 'prepare' else 'evaluation_workers'
    if args.gpus != config.get(worker_key, config.get('world_size', 1)) or args.gpus < 1:
        raise ValueError('Worker count differs from frozen configuration')
    import torch
    if torch.cuda.device_count() != args.gpus:
        raise ValueError(f'Expected {args.gpus} visible GPUs, found {torch.cuda.device_count()}')
    args.out.mkdir(parents=True, exist_ok=True)
    print('config: ' + json.dumps(dict(phase=args.phase, gpus=args.gpus, config=config)), flush=True)
    commands = []
    if args.phase == 'prepare':
        commands = [[sys.executable, str(CODE / 'train_full.py'), 'prepare', '--config', str(args.config),
            '--out', str(args.out), '--rank', str(rank), '--world-size', str(args.gpus)] for rank in range(args.gpus)]
    else:
        if config.get('tokens_file') or config.get('limit_tokens'):
            raise ValueError('Parallel evaluation must use the complete frozen test subset')
        tokens = expected_tokens(config, 'test')
        if len(tokens) < args.gpus:
            raise ValueError('Not enough evaluation tokens for the requested workers')
        for rank in range(args.gpus):
            root = args.out / 'shards' / f'rank_{rank:02d}'
            root.mkdir(parents=True, exist_ok=True)
            path = root / 'requested_tokens.json'
            write_json(path, tokens[rank::args.gpus])
            write_json(root / 'config.json', {**config, 'tokens_file': str(path)})
            commands.append([sys.executable, str(CODE / 'core/eval_open_loop.py'),
                '--config', str(root / 'config.json'), '--out', str(root)])
    run_workers(commands, args.out, visible_gpu_identifiers(args.gpus))
    if args.phase == 'prepare':
        merge_cache(config, args.out, args.gpus)
    else:
        merge_evaluation(config, args.out, args.gpus)
    print('PARALLEL_PHASE_COMPLETE', args.phase, 'gpus', args.gpus, flush=True)


if __name__ == '__main__':
    main()
