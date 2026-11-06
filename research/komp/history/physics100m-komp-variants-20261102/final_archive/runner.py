"""Run isolated variant trials with atomic claims and first-failure stop."""
import json, os, subprocess, time
from pathlib import Path
r=Path(__file__).resolve().parent;q=r/'queue';worker=os.environ['WORKER'];start=time.monotonic()
while time.monotonic()-start<13500:
    if (q/'STOP').exists():break
    if (q/'FAILED.json').exists():time.sleep(10);continue
    claimed=False
    for p in sorted(q.glob('*.json')):
        if p.name=='FAILED.json':continue
        try:spec=json.loads(p.read_text())
        except FileNotFoundError:continue
        if any(not Path(f).exists() for f in spec.get('requires',[])):continue
        if (q/'FAILED.json').exists():break
        claim=p.with_suffix(f'.{worker}.running')
        try:p.rename(claim)
        except FileNotFoundError:continue
        name=p.stem
        with (r/'logs'/f'{name}.log').open('a') as log:
            code=subprocess.call(spec['command'],cwd=r/'suite100m',stdout=log,stderr=subprocess.STDOUT)
        result=dict(name=name,worker=worker,exit_code=code,finished=time.time(),spec=spec)
        (q/f'{name}.done').write_text(json.dumps(result))
        if code:
            try:
                with (q/'FAILED.json').open('x') as f:json.dump(result,f)
            except FileExistsError:pass
        claimed=True;break
    if not claimed:time.sleep(5)
(r/'evidence'/f'worker-{worker}-exit.json').write_text(json.dumps(dict(worker=worker,seconds=time.monotonic()-start,exit=0)))
