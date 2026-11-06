"""Read-only remote snapshot; execute via SSH stdin."""
import datetime
import json
from pathlib import Path
root=Path('/gpfs/users/linlizhi/physics100m-exact-20261105')
r=root/'exact_tuning_20261105'
report={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tasks':{}}
for task in ('depo2','brevo2'):
 out={'numerical':None,'precision_ddp':None,'candidates':[]}
 for key,path in [('numerical',r/'evidence'/f'numerical-{task}.json'),
                  ('precision_ddp',r/'preflight'/task/'runs-ddp/ddp-preflight/exact.json')]:
  if path.exists():
   v=json.loads(path.read_text());out[key]={k:v[k] for k in ('status','check_count','sources','fp32_gradient_rms','parameter_max_abs') if k in v}
 for c in json.loads((r/'initial-screen.json').read_text())['screen']:
  dest=r/'results'/task/c['name'];row={'name':c['name'],'config':c,'status':'not_started','metrics':[]}
  p=dest/'config.json'
  if p.exists():row['training_config']=json.loads(p.read_text())
  p=dest/'metrics.jsonl'
  if p.exists():
   for line in p.read_text().splitlines():
    try:row['metrics'].append(json.loads(line))
    except json.JSONDecodeError:pass
   if row['metrics']:row['status']='started'
  p=dest/'summary.json'
  if p.exists():row['summary']=json.loads(p.read_text());row['status']=row['summary']['status']
  out['candidates'].append(row)
 p=root/'logs'/f'group-{task}.log'
 out['group_tail']=p.read_text().splitlines()[-2:] if p.exists() else []
 report['tasks'][task]=out
p=root/'ddp-FAILED.json';report['failure']=json.loads(p.read_text()) if p.exists() else None
print(json.dumps(report))
