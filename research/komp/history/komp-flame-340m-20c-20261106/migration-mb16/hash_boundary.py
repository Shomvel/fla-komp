import hashlib,json,sys,time
from pathlib import Path
base=Path('/gpfs/users/linlizhi/komp-flame-340m-20c-20261106');kind=sys.argv[1]
pause=base/'migration-mb16'/f'paused-{kind}.json'
while not pause.exists():time.sleep(1)
r=json.loads(pause.read_text());assert r['last_logged_step']==r['target_step'] and not r['uncheckpointed_complete_steps']
folder=base/'runs'/kind/f"checkpoint-{r['target_step']:06d}"
files={}
for p in sorted(folder.iterdir()):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 files[p.name]=h.hexdigest()
result=dict(checkpoint=str(folder),step=r['target_step'],files=files)
(base/'migration-mb16/runtime/evidence'/f'checkpoint-{kind}-hashes.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
