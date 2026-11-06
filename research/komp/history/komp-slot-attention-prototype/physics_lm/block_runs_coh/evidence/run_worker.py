"""16 isolated H100 workers; E1 -> locked selection -> E2, fail-fast across workers."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path('/gpfs/users/linlizhi/komp-coh-trials-20261101')
INDEX = int(os.environ['COH_WORKER'])
WORKER = ROOT/'workers'/f'r{INDEX+1}'
SNAP = ROOT/'snapshot'
RESULT = ROOT/'results'
EXPECTED = json.loads((ROOT/'snapshot-sha256.json').read_text())
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
    run(['block_coherence.py', '--kind', kind, '--seed', str(seed)], name)
    folder = RESULT/'results'/name
    r = json.loads((folder/'summary.json').read_text())
    assert r['status']=='completed' and r['unique_tokens']==10001108 and r['reload_validation_ce_error']<1e-4
    assert (RESULT/f'{name}-coherence.json').exists()
    if seed == 731: run(['block_coherence.py','--diagnose','--kind',kind,'--seed',str(seed)], name+'-diagnose')
    (folder/'model.pt').unlink()


def main():
    import torch, triton, google.protobuf
    from google.protobuf.internal.api_implementation import Type
    WORKER.mkdir(parents=True, exist_ok=True); RESULT.mkdir(exist_ok=True)
    verify()
    dump(WORKER/'environment.json', dict(python=sys.version, torch=str(torch.__version__),triton=triton.__version__,
        gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda, protobuf=google.protobuf.__version__,protobuf_backend=Type()))
    if INDEX == 0:
        log=run([str(SNAP/'tools/check_coherence.py'), str(SNAP), str(SNAP/'r3-reference/block_flex.py')], 'cpu-fp64-coherence')
        lines=log.read_text().splitlines(); worst=float(next(s.split()[1] for s in lines if s.startswith('WORST ')))
        assert worst < 1e-9 and not any('default bitwise-equal-old False' in s for s in lines)
        dump(ROOT/'cpu-acceptance.json',dict(status='passed',cases=sum(s.startswith('(') for s in lines),worst=worst))
        run(['block_coherence.py','--preflight'], 'preflight-all12')
        # Atomically publish after the additional statistics/gradient checks, not be.preflight's early ticket.
        dump(ROOT/'preflight-ready.json', dict(status='passed'))
    wait_file(ROOT/'preflight-ready.json')
    e1=[(arm,seed) for seed in PLAN['seeds'] for arm in PLAN['e1']]
    for kind,seed in e1[INDEX::16]: trial(kind, seed)
    dump(WORKER/'e1-done.json',dict(status='completed',runs=e1[INDEX::16]))
    if INDEX == 0:
        for i in range(16): wait_file(ROOT/'workers'/f'r{i+1}'/'e1-done.json')
        ranking=[]
        for kind in PLAN['e1']:
            rows=[json.loads((RESULT/'results'/f'{kind}-seed{s}'/'summary.json').read_text()) for s in PLAN['seeds']]
            ranking.append(dict(kind=kind,mean_content_ce=statistics.mean(r['confirm']['mean_content_ce'] for r in rows),
                mean_val_content_ce=statistics.mean(r['val']['mean_content_ce'] for r in rows)))
        best=min((r for r in ranking if r['kind']!='coh_fifo_off'), key=lambda r:(r['mean_content_ce'], PLAN['e1'].index(r['kind'])))
        dump(ROOT/'selection.json',dict(status='locked',criterion='minimum seven-seed confirm mean_content_ce among enabled thresholds',
            tie_break='threshold order .99, .95, .80',selected_tag=best['kind'].rsplit('_',1)[1],ranking=ranking,
            off_compared_separately=True,selection_bias='same confirm used for threshold selection and reporting; not an independent test'))
    wait_file(ROOT/'selection.json')
    selected=json.loads((ROOT/'selection.json').read_text())['selected_tag']
    arms=['coh_gate_off',f'coh_gate_{selected}','coh_learned_off',f'coh_learned_{selected}']
    e2=[(arm,seed) for seed in PLAN['seeds'] for arm in arms]
    for kind,seed in e2[INDEX::16]: trial(kind,seed)
    verify()
    dump(WORKER/'completion.json',dict(status='completed',e1=e1[INDEX::16],e2=e2[INDEX::16],source_verified=True))


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        try:
            with (ROOT/'failure.json').open('x') as output:
                json.dump(dict(worker=INDEX+1,error=repr(exc),time=time.time()),output)
        except FileExistsError:
            pass
        raise
