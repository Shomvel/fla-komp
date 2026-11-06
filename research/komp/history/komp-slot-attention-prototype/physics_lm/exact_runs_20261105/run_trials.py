"""Sequential, source-pinned training; stop on the first failing command."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SNAP = ROOT/'snapshot'
EXPECTED = json.loads((ROOT/'snapshot-sha256.json').read_text())
PLAN = json.loads((ROOT/'plan.json').read_text())
os.environ['EXACT_RUN_ROOT'] = str(ROOT)
os.environ['FLA_SOURCE'] = '/gpfs/users/linlizhi/komp-gsa-gdn-20261031-r1/fla-source'
os.environ['FLA_DISABLE_BACKEND_DISPATCH'] = '1'
os.environ['FLA_DISABLE_TENSOR_CACHE'] = '1'


def verify():
    for name, digest in EXPECTED.items():
        assert hashlib.sha256((SNAP/name).read_bytes()).hexdigest() == digest, name


def run(args, name):
    verify()
    log = ROOT/'logs'/f'{name}.log'
    start = time.time()
    print('START', name, flush=True)
    with log.open('w') as output:
        process = subprocess.run([sys.executable, *args], cwd=SNAP/'physics_lm',
                                 stdout=output, stderr=subprocess.STDOUT)
    row = {'name': name, 'exit_code': process.returncode, 'elapsed_seconds': time.time()-start}
    with (ROOT/'progress.jsonl').open('a') as out:
        out.write(json.dumps(row)+'\n')
    print('END', json.dumps(row), flush=True)
    if process.returncode:
        print('\n'.join(log.read_text().splitlines()[-55:]), flush=True)
        raise RuntimeError(f'First failure stops all runs: {name}')


def main():
    import torch
    import triton
    import google.protobuf
    from google.protobuf.internal.api_implementation import Type
    (ROOT/'logs').mkdir(exist_ok=True)
    verify()
    env = {'torch': str(torch.__version__), 'triton': triton.__version__, 'cuda': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(), 'python': sys.version, 'protobuf': google.protobuf.__version__,
           'protobuf_backend': Type()}
    (ROOT/'environment.json').write_text(json.dumps(env, indent=2))
    fla = Path(os.environ['FLA_SOURCE'])
    (ROOT/'fla-sha256.json').write_text(json.dumps({str(p.relative_to(fla)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                for p in sorted((fla/'fla').rglob('*.py'))}, indent=2))
    run(['exact_experiment.py', '--preflight'], 'preflight')
    run([str(ROOT/'control.py')], 'gdn-seed731-control')
    for seed in PLAN['seeds']:
        for kind in PLAN['variants']:
            name = f'{kind}-seed{seed}'
            run(['exact_experiment.py', '--kind', kind, '--seed', str(seed)], name)
            summary = json.loads((ROOT/'results'/name/'summary.json').read_text())
            assert summary['status'] == 'completed' and summary['unique_tokens'] == 10001108
            assert summary['reload_validation_ce_error'] < 1e-4
    verify()
    (ROOT/'completion.json').write_text(json.dumps({'status': 'completed', 'new_runs': 28, 'control_runs': 1,
                                                  'source_verified': True}, indent=2))


if __name__ == '__main__':
    main()
