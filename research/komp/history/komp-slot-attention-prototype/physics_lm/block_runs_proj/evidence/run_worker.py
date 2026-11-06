"""16 isolated H100 workers; E1 -> locked selection -> E2, fail-fast across workers."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path('/gpfs/users/linlizhi/komp-proj-swa-20261101')
INDEX = int(os.environ['PROJ_WORKER'])
WORKER = ROOT/'workers'/f'r{INDEX+1}'
SNAP = ROOT/'snapshot'
RESULT = ROOT/'results'
EXPECTED = json.loads((ROOT/'snapshot-sha256.json').read_text())
EXPECTED.update(json.loads((ROOT/'fla-provenance.json').read_text()))
PLAN = json.loads((ROOT/'plan.json').read_text())


def dump(path, obj):
    temporary = path.with_suffix(f'.tmp-r{INDEX+1}')
    temporary.write_text(json.dumps(obj, indent=2)); temporary.replace(path)


def verify():
    for name, digest in EXPECTED.items():
        assert hashlib.sha256((SNAP/name).read_bytes()).hexdigest() == digest, name
    assert {p.name for p in (SNAP/'komp_slot').glob('*.py')} == {Path(n).name for n in EXPECTED if n.startswith('komp_slot/') and n.endswith('.py')}


def wait_file(path):
    while not path.exists():
        if (ROOT/'failure.json').exists(): raise RuntimeError('A worker failed; stop dependent work')
        time.sleep(5)
    if (ROOT/'failure.json').exists(): raise RuntimeError('A worker failed; stop dependent work')


def run(command, name):
    if (ROOT/'failure.json').exists(): raise RuntimeError('A worker failed; stop new runs')
    verify(); log = WORKER/f'{name}.log'; started = time.time()
    dump(WORKER/'active.json', dict(name=name, started=started, command=command))
    print('START', name, flush=True)
    with log.open('w') as output:
        process = subprocess.run([sys.executable, *command], cwd=SNAP/'physics_lm', stdout=output, stderr=subprocess.STDOUT)
    row = dict(name=name, command=command, exit_code=process.returncode, elapsed_seconds=time.time()-started, log=str(log))
    with (WORKER/'progress.jsonl').open('a') as output: output.write(json.dumps(row)+'\n')
    if process.returncode: raise RuntimeError(f'First failure: {name}; see {log}')
    print('END', json.dumps(row), flush=True)
    return log


def trial(kind, seed):
    name = f'{kind}-seed{seed}'
    run(['block_ablation.py', '--kind', kind, '--seed', str(seed)], name)
    folder = RESULT/'results'/name
    r = json.loads((folder/'summary.json').read_text())
    assert r['status']=='completed' and r['unique_tokens']==10001108 and r['reload_validation_ce_error']<1e-4
    (folder/'model.pt').unlink()


def main():
    import torch, triton, google.protobuf
    from google.protobuf.internal.api_implementation import Type
    WORKER.mkdir(parents=True, exist_ok=True); RESULT.mkdir(exist_ok=True)
    verify()
    dump(WORKER/'environment.json', dict(python=sys.version, torch=str(torch.__version__),triton=triton.__version__,
        gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda, protobuf=google.protobuf.__version__,protobuf_backend=Type()))
    if INDEX == 0:
        log=run([str(SNAP/'tools/check_project.py'), str(SNAP), str(SNAP/'previous-reference/block_flex.py')], 'cpu-fp64-project')
        lines=log.read_text().splitlines(); worst=float(next(s.split()[1] for s in lines if s.startswith('WORST ')))
        assert worst < 1e-9 and not any('default bitwise-equal-old False' in s for s in lines)
        dump(ROOT/'cpu-acceptance.json',dict(status='passed',cases=sum(s.startswith('(') for s in lines),worst=worst))
        run(['block_ablation.py','--preflight'], 'preflight-all10')
        dump(ROOT/'preflight-ready.json', dict(status='passed'))
    wait_file(ROOT/'preflight-ready.json')
    # Interleave seeds then architecture; static sharding gives 3-4 main runs per worker.
    main_runs=[(arm,seed) for seed in PLAN['seeds'] for arm in PLAN['main']]
    for kind,seed in main_runs[INDEX::16]: trial(kind,seed)
    dump(WORKER/'main-done.json',dict(status='completed',runs=main_runs[INDEX::16]))
    for i in range(16): wait_file(ROOT/'workers'/f'r{i+1}'/'main-done.json')
    optional=[(arm,seed) for seed in PLAN['seeds'] for arm in PLAN['optional']]
    for kind,seed in optional[INDEX::16]: trial(kind,seed)
    verify()
    dump(WORKER/'completion.json',dict(status='completed',main=main_runs[INDEX::16],optional=optional[INDEX::16],source_verified=True))


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        try:
            with (ROOT/'failure.json').open('x') as output:
                json.dump(dict(worker=INDEX+1,error=repr(exc),time=time.time()),output)
        except FileExistsError:
            pass
        raise
