"""Fail-fast isolated runner; verifies frozen source before preflight and each run."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/gpfs/users/linlizhi/komp-gsa2-trials-20261101')
SNAP=ROOT/'snapshot';WORK=SNAP/'physics_lm';RESULT=ROOT/'results'
EXPECTED=json.loads((ROOT/'snapshot-sha256.json').read_text())
PLAN=json.loads((ROOT/'plan.json').read_text())

def verify():
    for name,digest in EXPECTED.items():assert hashlib.sha256((SNAP/name).read_bytes()).hexdigest()==digest,name
    # An extra kernel .py would invalidate preflight tickets, so reject it as well.
    assert set(p.name for p in (SNAP/'komp_slot').glob('*.py'))==set(Path(n).name for n in EXPECTED if n.startswith('komp_slot/') and n.endswith('.py'))


def run(command,name):
    verify();log=RESULT/'logs'/f'{name}.log'
    started=time.time();print('START',name,time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),flush=True)
    with log.open('w') as output:
        completed=subprocess.run([sys.executable,*command],cwd=WORK,stdout=output,stderr=subprocess.STDOUT)
    row=dict(name=name,command=command,exit_code=completed.returncode,elapsed_seconds=time.time()-started,log=str(log))
    with (RESULT/'progress.jsonl').open('a') as output:output.write(json.dumps(row)+'\n')
    print('END',json.dumps(row),flush=True)
    if completed.returncode:
        print('\n'.join(log.read_text().splitlines()[-35:]),flush=True)
        raise RuntimeError(f'First failure stops all subsequent runs: {name}')


def main():
    import torch,triton,google.protobuf
    from google.protobuf.internal.api_implementation import Type
    verify()
    env=dict(python=sys.version,torch=str(torch.__version__),triton=triton.__version__,gpu=torch.cuda.get_device_name(),
             protobuf=google.protobuf.__version__,protobuf_backend=Type(),cuda=torch.version.cuda)
    (ROOT/'environment.json').write_text(json.dumps(env,indent=2))
    run(['gsa2_trials.py','--preflight'],'preflight-gsa2')
    for seed in PLAN['seeds']:
        for kind in PLAN['arms']:
            name=f'{kind}-seed{seed}'
            run(['gsa2_trials.py','--kind',kind,'--seed',str(seed)],name)
            folder=RESULT/'results'/name
            summary=json.loads((folder/'summary.json').read_text())
            assert summary['status']=='completed' and summary['unique_tokens']==10001108
            assert summary['reload_validation_ce_error']<1e-4
            (folder/'model.pt').unlink()
    verify()
    (ROOT/'completion.json').write_text(json.dumps(dict(status='completed',runs=len(PLAN['arms'])*len(PLAN['seeds']),source_verified=True),indent=2))

if __name__=='__main__':main()
