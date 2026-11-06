"""Run the eight prespecified trials on three GPUs; stop on first failure."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'decay_runs'
assert json.loads((DEST/'preflight.json').read_text())['status']=='passed'
queues={0:[('sigmoid',731),('sigmoid',732),('gsa_scalar',731),('gsa_scalar',732)],
        1:[('gdn_scalar',731),('gdn_scalar',732)],
        2:[('gsa_slot',731),('gsa_slot',732)]}
active={};complete=[];logs=[]
try:
    while len(complete)<8:
        for gpu,queue in queues.items():
            if gpu not in active and queue:
                kind,seed=queue.pop(0)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1')
                log=(DEST/f'{kind}-seed{seed}.log').open('w');logs.append(log)
                proc=subprocess.Popen([sys.executable,str(ROOT/'decay_experiment.py'),'--kind',kind,'--seed',str(seed)],
                                      env=env,stdout=log,stderr=subprocess.STDOUT)
                active[gpu]=(proc,kind,seed)
                print('LAUNCHED',gpu,kind,seed,proc.pid,flush=True)
        for gpu,(proc,kind,seed) in list(active.items()):
            code=proc.poll()
            if code is not None:
                if code:raise RuntimeError(f'{kind} seed{seed} failed {code}; no automatic retry')
                r=json.loads((DEST/'results'/f'{kind}-seed{seed}'/'summary.json').read_text())
                assert r['status']=='completed' and r['unique_tokens']==10001108
                complete.append({'kind':kind,'seed':seed});del active[gpu]
        (DEST/'status.json').write_text(json.dumps({'completed':complete,
            'running':{g:[k,s] for g,(_,k,s) in active.items()},'pending':queues},indent=2))
        if len(complete)<8:time.sleep(5)
    print('ALL EIGHT TRIALS COMPLETED',flush=True)
finally:
    for proc,_,_ in active.values():
        if proc.poll() is None:proc.terminate()
    for log in logs:log.close()
