"""Read-only8GPU group status; run via SSH stdin."""
import hashlib
import json
from pathlib import Path
r=Path('/gpfs/users/linlizhi/physics100m-exact-20261105')
out={'root':str(r),'preflight':{},'groups':{},'training':[],'finished':[],'evaluation':[],'evaluation_progress':[]}
current={'snapshot':hashlib.sha256((r/'evidence/snapshot-sha256.json').read_bytes()).hexdigest(),'ddp_train':hashlib.sha256((r/'ddp_train.py').read_bytes()).hexdigest()}
for kind in ('exact','komp','gdn','full','swa','gsa','gsa2'):
 p=r/'runs-ddp/ddp-preflight'/f'{kind}.json'
 if not p.exists():continue
 row=json.loads(p.read_text());out['preflight'][kind]={k:row[k] for k in ('status','model','world_size','fp32_gradient_rms','parameter_max_abs','sources')}
 out['preflight'][kind]['current']=row['sources']==current
out['passed_current']=sum(x['current'] and x['status']=='passed' for x in out['preflight'].values())
for task in ('depo2','brevo2'):
 p=r/'logs'/f'group-{task}.log'
 out['groups'][task]={'tail':p.read_text().splitlines()[-2:] if p.exists() else []}
 p=r/'evidence'/f'group-{task}-environment.json'
 if p.exists():out['groups'][task]['environment']=json.loads(p.read_text())
for d in sorted([d for folder in ('runs-ddp','runs-gsa2-consistent') for d in (r/folder/'results').glob('*')]):
 implementation='numerically-stabilized' if d.relative_to(r).parts[0]=='runs-gsa2-consistent' else 'original-frozen'
 p=d/'summary.json'
 if p.exists():
  row=json.loads(p.read_text());out['finished'].append({'name':d.name,'implementation':implementation,'val':row['val'],'tokens':row['tokens']})
 elif (d/'metrics.jsonl').exists():
  lines=(d/'metrics.jsonl').read_text().splitlines()
  out['training'].append({'name':d.name,'implementation':implementation,'latest':json.loads(lines[-1]) if lines else None})
 p=d/'eval-progress.json'
 if p.exists() and not (d/'heldout.json').exists():
  row=json.loads(p.read_text());out['evaluation_progress'].append({'name':d.name,'implementation':implementation,'test':row['test'],'generation_probes_completed':len(row.get('records',[]))})
 p=d/'heldout.json'
 if p.exists():out['evaluation'].append({'name':d.name,'implementation':implementation,**json.loads(p.read_text())})
fail=r/'ddp-FAILED.json';out['failure']=json.loads(fail.read_text()) if fail.exists() else None
p=r/'runs-gsa2-consistent/ddp-preflight/gsa2.json'
if p.exists():
 row=json.loads(p.read_text());out['gsa2_stabilized_preflight']={k:row.get(k) for k in ('status','sources','acceptance_scope','fp32_gradient_rms','bf16_real_gradient_rms','bf16_real_parameter_max_abs')}
else:out['gsa2_stabilized_preflight']=None
out['candidate_failures']=[]
for task in ('depo2','brevo2'):
 p=r/'logs'/f'gsa2-consistent-{task}.log'
 if p.exists() and any(marker in p.read_text() for marker in ('Traceback (most recent call last)', 'ChildFailedError')):
  out['candidate_failures'].append({'task':task,'log':str(p)})
out['complete']=(not out['candidate_failures'] and len(out['finished'])==14 and len(out['evaluation'])==14 and out['failure'] is None)
print(json.dumps(out))
