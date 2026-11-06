"""Two source-pinned GPU workers; fail fast, resume from existing checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
QUEUE = ROOT/'queue'
WORKER = os.environ['WORKER']
EXPECTED = json.loads((ROOT/'evidence/snapshot-sha256.json').read_text())
PLAN = json.loads((ROOT/'plan.json').read_text())


def verify():
    for name, digest in EXPECTED.items():
        assert hashlib.sha256((ROOT/'snapshot'/name).read_bytes()).hexdigest() == digest, name


def main():
    import torch
    import triton
    import google.protobuf
    from google.protobuf.internal.api_implementation import Type
    env = {'worker': WORKER, 'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
           'triton': triton.__version__, 'cuda': torch.version.cuda,
           'protobuf': google.protobuf.__version__, 'protobuf_backend': Type()}
    (ROOT/'evidence'/f'worker-{WORKER}-environment.json').write_text(json.dumps(env, indent=2))
    start = time.monotonic()
    while time.monotonic()-start < int(os.environ.get('WORKER_LIFETIME_SECONDS', '42100')):
        if (QUEUE/'FAILED.json').exists():
            raise RuntimeError('Peer failure stops new work; static diagnosis required')
        heldout = list((ROOT/'runs/results').glob('*/heldout.json'))
        if len(heldout) == PLAN['runs']:
            assert all(json.loads(p.read_text())['status'] == 'completed' for p in heldout)
            (ROOT/'evidence/ALL_COMPLETE.json').write_text(json.dumps({'runs': len(heldout), 'status': 'completed'}))
            break
        if (QUEUE/'STOP').exists():
            break
        claimed = False
        for path in sorted(QUEUE.glob('*.json')):
            if path.name == 'FAILED.json':
                continue
            spec = json.loads(path.read_text())
            if any(not Path(p).exists() for p in spec.get('requires', [])):
                continue
            claim = path.with_suffix(f'.{WORKER}.running')
            try:
                path.rename(claim)
            except FileNotFoundError:
                continue
            verify()
            name = path.stem
            print('START', name, flush=True)
            began = time.time()
            with (ROOT/'logs'/f'{name}.log').open('a') as log:
                code = subprocess.call(spec['command'], cwd=ROOT/'snapshot/suite100m', stdout=log, stderr=subprocess.STDOUT)
            receipt = {'name': name, 'worker': WORKER, 'exit_code': code, 'seconds': time.time()-began,
                       'finished': time.time(), 'spec': spec}
            (QUEUE/f'{name}.done').write_text(json.dumps(receipt, indent=2))
            print('END', json.dumps(receipt), flush=True)
            if code:
                try:
                    with (QUEUE/'FAILED.json').open('x') as out:
                        json.dump(receipt, out)
                except FileExistsError:
                    pass
                raise RuntimeError(f'First failing command: {name}')
            claimed = True
            break
        if not claimed:
            time.sleep(5)
    (ROOT/'evidence'/f'worker-{WORKER}-exit.json').write_text(json.dumps({'worker': WORKER, 'seconds': time.monotonic()-start}))


if __name__ == '__main__':
    main()
