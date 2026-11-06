"""One dedicated eight-GPU group per task, two new learning rates sequentially."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SUITE = ROOT.parent
TASK = os.environ['LR_TASK']


def verify():
    manifest = json.loads((ROOT/'source-manifest.json').read_text())
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    for name, expected in manifest['inherited_files'].items():
        assert hashlib.sha256((SUITE/name).read_bytes()).hexdigest() == expected, name
    for name, expected in json.loads((SUITE/'evidence/snapshot-sha256.json').read_text()).items():
        assert hashlib.sha256((SUITE/'snapshot'/name).read_bytes()).hexdigest() == expected, name


def main():
    import torch
    import triton
    import google.protobuf
    from google.protobuf.internal.api_implementation import Type
    assert TASK in ('depo2', 'brevo2')
    assert torch.cuda.device_count() == 8
    verify()
    (ROOT/'evidence'/f'environment-{TASK}.json').write_text(json.dumps({
        'task': TASK, 'gpus': [torch.cuda.get_device_name(i) for i in range(8)],
        'torch': str(torch.__version__), 'triton': triton.__version__, 'cuda': torch.version.cuda,
        'protobuf': google.protobuf.__version__, 'protobuf_backend': Type(),
    }, indent=2))
    plan = json.loads((ROOT/'lr-screen.json').read_text())
    receipt = json.loads((ROOT/'evidence'/f'reuse-{TASK}.json').read_text())
    assert receipt['status'] == 'verified' and receipt['new_training_steps'] == 0
    for entry in plan['new_runs'][TASK]:
        if (ROOT/'STOP').exists():
            return
        assert not (ROOT/'FAILED.json').exists(), 'Inspect recorded failure before resuming'
        verify()
        done = ROOT/'results'/TASK/entry['name']/'summary.json'
        if done.exists():
            summary = json.loads(done.read_text())
            assert summary['status'] == 'completed' and summary['config']['lr'] == entry['lr']
            continue
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
                   str(ROOT/'lr_train.py'), '--task', TASK, '--candidate', entry['name']]
        print('START', TASK, entry['name'], flush=True)
        started = time.time()
        with (ROOT/'logs'/f'{TASK}-{entry["name"]}.log').open('a') as log:
            code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record = {'task': TASK, 'entry': entry, 'exit_code': code, 'seconds': time.time()-started, 'command': command}
        with (ROOT/'evidence'/f'progress-{TASK}.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print('END', json.dumps(record), flush=True)
        if code:
            try:
                with (ROOT/'FAILED.json').open('x') as stream:
                    json.dump(record, stream)
            except FileExistsError:
                pass
            raise RuntimeError(entry['name'])
        assert done.exists() and json.loads(done.read_text())['status'] == 'completed'
    if not (ROOT/'STOP').exists():
        (ROOT/'evidence'/f'group-{TASK}-complete.json').write_text(json.dumps({
            'status': 'completed', 'task': TASK, 'new_runs': 2, 'reused_runs': 1,
            'source_manifest_sha256': hashlib.sha256((ROOT/'source-manifest.json').read_bytes()).hexdigest(),
        }, indent=2))


if __name__ == '__main__':
    main()
