"""Bounded three-GPU search, locked selection, then 100M-token scale-up."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'lrwd_runs'
KINDS = ('komp_slot', 'gsa', 'gdn')
LRS = (2.5e-4, 5e-4, 1e-3)
WDS = (0., .01, .1)
COMMAND = [sys.executable, str(ROOT / 'lrwd_experiment.py')]
DEST.mkdir(parents=True, exist_ok=True)


def run_stage(stage, queues, data_proc):
    active = {}
    completed = []
    handles = []
    try:
        while any(queues.values()) or active:
            if data_proc.poll() not in (None, 0):
                raise RuntimeError('100M data preparation failed; see data100m.log')
            for gpu, queue in queues.items():
                if gpu not in active and queue:
                    kind, seed, lr, wd, scale = queue.pop(0)
                    name = f'{kind}-{scale}-lr{lr:g}-wd{wd:g}-seed{seed}'
                    handle = (DEST / f'{name}.log').open('x')
                    handles.append(handle)
                    proc = subprocess.Popen(COMMAND + ['--action', 'train', '--kind', kind, '--seed', str(seed),
                                                       '--lr', str(lr), '--wd', str(wd), '--scale', scale],
                                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                            stdout=handle, stderr=subprocess.STDOUT)
                    active[gpu] = (proc, name)
                    print('LAUNCH', stage, gpu, name, flush=True)
            for gpu, (proc, name) in list(active.items()):
                code = proc.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f'{name} failed ({code}); no automatic retry')
                    summary = json.loads((DEST / 'results' / name / 'summary.json').read_text())
                    assert summary['status'] == 'completed'
                    completed.append(name)
                    del active[gpu]
                    print('COMPLETE', stage, name, summary['val']['mean_content_ce'], flush=True)
            (DEST / 'status.json').write_text(json.dumps({'stage': stage, 'completed': completed,
                'running': {g: n for g, (_, n) in active.items()}, 'pending': queues}, indent=2))
            if active or any(queues.values()):
                time.sleep(5)
        (DEST / f'{stage}-completed.json').write_text(json.dumps(completed, indent=2))
    finally:
        for proc, _ in active.values():
            if proc.poll() is None:
                proc.terminate()
        for handle in handles:
            handle.close()


def main():
    with (DEST / 'data100m.log').open('x') as data_log:
        data_proc = subprocess.Popen([sys.executable, str(ROOT / 'prepare_100m.py')],
                                     stdout=data_log, stderr=subprocess.STDOUT)
        try:
            subprocess.run(COMMAND + ['--action', 'preflight'], check=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'))
            run_stage('search', {gpu: [(kind, 731, lr, wd, 'small') for lr in LRS for wd in WDS]
                                for gpu, kind in enumerate(KINDS)}, data_proc)
            subprocess.run(COMMAND + ['--action', 'select'], check=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'))
            selection = json.loads((DEST / 'selection.json').read_text())
            run_stage('replication', {gpu: [(kind, 732, selection['models'][kind]['selected']['lr'],
                                            selection['models'][kind]['selected']['wd'], 'small')]
                                     for gpu, kind in enumerate(KINDS)}, data_proc)
            subprocess.run(COMMAND + ['--action', 'evaluate', '--scale', 'small'], check=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'))
            if data_proc.wait() != 0:
                raise RuntimeError('100M data preparation failed')
            run_stage('large', {gpu: [(kind, 731, selection['models'][kind]['selected']['lr'],
                                      selection['models'][kind]['selected']['wd'], 'large')]
                               for gpu, kind in enumerate(KINDS)}, data_proc)
            subprocess.run(COMMAND + ['--action', 'evaluate', '--scale', 'large'], check=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'))
            (DEST / 'completed.json').write_text(json.dumps({'status': 'completed', 'search_runs': 27,
                'replication_runs': 3, 'large_runs': 3, 'selection': selection}, indent=2))
            print('ALL 33 RUNS AND HELDOUT EVALUATIONS COMPLETED', flush=True)
        finally:
            if data_proc.poll() is None:
                data_proc.terminate()


if __name__ == '__main__':
    main()
