"""Read-only alpha-only formal run status; run via SSH stdin."""
import json
from pathlib import Path
ROOT=Path('/gpfs/users/linlizhi/physics100m-exact-20261105/exact_alpha995_formal_20261105')
report={'experiment':'alpha995 single-factor formal','training':[],'completed':[],'evaluation':[],'evaluation_progress':[],'preflight':{},'failures':[]}
for task in ('depo2','brevo2'):
 p=ROOT/f'preflight-{task}/runs-ddp/ddp-preflight/exact.json'
 if p.exists():
  r=json.loads(p.read_text());report['preflight'][task]={k:r[k] for k in ('status','sources','world_size','fp32_gradient_rms')}
 p=ROOT/'logs'/f'group-{task}.log'
 if p.exists() and any(s in p.read_text() for s in ('Traceback (most recent call last)','ChildFailedError')):report['failures'].append({'task':task,'log':str(p)})
for path in sorted((ROOT/'runs-ddp/results').glob('*')):
 p=path/'summary.json'
 if p.exists():
  r=json.loads(p.read_text());report['completed'].append({'name':path.name,'val':r['val'],'positions':r['tokens']})
 elif (path/'metrics.jsonl').exists():
  lines=(path/'metrics.jsonl').read_text().splitlines()
  if lines:report['training'].append({'name':path.name,'latest':json.loads(lines[-1])})
  else:report['training'].append({'name':path.name,'latest':None,'status':'initialized_before_first_step'})
 if (path/'heldout.json').exists():report['evaluation'].append({'name':path.name,**json.loads((path/'heldout.json').read_text())})
 elif (path/'eval-progress.json').exists():
  r=json.loads((path/'eval-progress.json').read_text());report['evaluation_progress'].append({'name':path.name,'test':r['test'],'generation_probes':len(r['records'])})
report['complete']=len(report['completed'])==len(report['evaluation'])==2 and not report['failures']
print(json.dumps(report))
