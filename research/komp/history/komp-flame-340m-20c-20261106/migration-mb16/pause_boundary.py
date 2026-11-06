import json,os,signal,sys,time
from pathlib import Path
kind=sys.argv[1];target=int(sys.argv[2]);root=Path('/gpfs/users/linlizhi/komp-flame-340m-20c-20261106');run=root/'runs'/kind
marker=run/f'checkpoint-{target:06d}/complete.json'
start=time.monotonic()
while not marker.exists():
 if time.monotonic()-start>3600:raise TimeoutError('checkpoint boundary wait exceeded one hour')
 time.sleep(.02)
pids=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=(p/'cmdline').read_bytes().split(b'\0')
 except (FileNotFoundError,PermissionError):continue
 if b'train.py' in args and b'--kind' in args and kind.encode() in args:pids.append(int(p.name))
assert len(pids)>=8,pids
for pid in pids:os.kill(pid,signal.SIGSTOP)
metrics=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
result=dict(status='paused_at_complete_checkpoint',kind=kind,target_step=target,checkpoint=json.loads(marker.read_text()),stopped_pids=pids,last_logged_step=metrics[-1]['step'],uncheckpointed_complete_steps=[r['step'] for r in metrics if r['step']>target],time=time.time())
(root/'migration-mb16'/f'paused-{kind}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
