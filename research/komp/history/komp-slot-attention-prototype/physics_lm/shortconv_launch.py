"""Run paired convolution trials on three GPUs, without automatic retries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'shortconv_runs'
DEST.mkdir(parents=True, exist_ok=True)
command = [sys.executable, str(ROOT / 'shortconv_experiment.py')]
subprocess.run(command + ['--preflight'], env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'), check=True)
queues = {i: [(kind + '_' + setting, seed) for seed in (731, 732) for setting in ('plain', 'conv')]
          for i, kind in enumerate(('komp_slot', 'gsa', 'gdn'))}
active = {}
completed = []
logs = []
try:
    while len(completed) < 12:
        for gpu, queue in queues.items():
            if gpu not in active and queue:
                kind, seed = queue.pop(0)
                log = (DEST / f'{kind}-seed{seed}.log').open('w')
                logs.append(log)
                proc = subprocess.Popen(command + ['--kind', kind, '--seed', str(seed)],
                                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                        stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = (proc, kind, seed)
                print('LAUNCHED', gpu, kind, seed, proc.pid, flush=True)
        for gpu, (proc, kind, seed) in list(active.items()):
            code = proc.poll()
            if code is not None:
                if code:
                    raise RuntimeError(f'{kind} seed {seed} failed: {code}; no automatic retry')
                dest = DEST / 'results' / f'{kind}-seed{seed}'
                summary = json.loads((dest / 'summary.json').read_text())
                assert summary['status'] == 'completed' and summary['unique_tokens'] == 10001108
                assert json.loads((dest / 'trained-causality.json').read_text())['status'] == 'passed'
                completed.append({'kind': kind, 'seed': seed})
                del active[gpu]
                print('COMPLETED', kind, seed, summary['generation'], flush=True)
        (DEST / 'status.json').write_text(json.dumps({'completed': completed,
            'running': {g: [k, s] for g, (_, k, s) in active.items()}, 'pending': queues}, indent=2))
        if len(completed) < 12:
            time.sleep(5)
    print('ALL 12 TRIALS COMPLETED', flush=True)
finally:
    for proc, _, _ in active.values():
        if proc.poll() is None:
            proc.terminate()
    for log in logs:
        log.close()
