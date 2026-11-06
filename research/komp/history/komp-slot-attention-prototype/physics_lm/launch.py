"""Bounded three-GPU launcher; one architecture per GPU, two sequential seeds."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
assert (ROOT/'preflight.json').exists()
kinds=os.environ.get('PHYSICS_KINDS','gdn,gsa,komp').split(',')
active={};queue={k:[731,732] for k in kinds};complete=[]
expected=2*len(kinds)
logs=[]
try:
    while len(complete)<expected:
        for gpu,kind in enumerate(queue):
            if kind not in active and queue[kind]:
                seed=queue[kind].pop(0)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1')
                log=(ROOT/f'{kind}-seed{seed}.log').open('w');logs.append(log)
                proc=subprocess.Popen([sys.executable,str(ROOT/'experiment.py'),'--kind',kind,'--seed',str(seed)],env=env,stdout=log,stderr=subprocess.STDOUT)
                active[kind]=(proc,seed)
                print('LAUNCHED',kind,seed,proc.pid,flush=True)
        for kind,(proc,seed) in list(active.items()):
            code=proc.poll()
            if code is not None:
                if code!=0:raise RuntimeError(f'{kind} seed{seed} failed with exit {code}; no retry')
                path=ROOT/'results'/f'{kind}-seed{seed}'/'summary.json'
                assert json.loads(path.read_text())['status']=='completed'
                complete.append({'kind':kind,'seed':seed});del active[kind]
        (ROOT/'status.json').write_text(json.dumps({'completed':complete,'running':{k:s for k,(_,s) in active.items()},'pending':queue},indent=2))
        if len(complete)<expected:time.sleep(5)
    print('ALL REQUESTED RUNS COMPLETED',flush=True)
except BaseException:
    for proc,_ in active.values():
        if proc.poll() is None:proc.terminate()
    raise
finally:
    for log in logs:log.close()
