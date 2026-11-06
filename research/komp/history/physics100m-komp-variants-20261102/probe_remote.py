import json,time
from pathlib import Path
r=Path('/gpfs/users/linlizhi/physics100m-komp-variants-20261102');q=r/'queue'
result=dict(at=time.time(),tickets={},preflight_logs={},failure=None,final=[],tuning_completed=0,active=[],pending=[],selections={})
for p in (r/'runs/preflight').glob('*.json'):result['tickets'][p.stem]=json.loads(p.read_text())
for p in (r/'logs').glob('preflight-*.log'):result['preflight_logs'][p.name]=p.read_text()[-5000:]
if (q/'FAILED.json').exists():result['failure']=json.loads((q/'FAILED.json').read_text())
for p in (r/'runs/results').glob('*/summary.json'):
 s=json.loads(p.read_text())
 if s['config']['phase']!='final':result['tuning_completed']+=1;continue
 row=dict(name=p.parent.name,summary=s)
 for name in ['heldout','generations']:
  f=p.with_name(name+'.json');row[name]=json.loads(f.read_text()) if f.exists() else None
 result['final'].append(row)
for p in q.glob('*.running'):
 ident,worker,_=p.name.rsplit('.',2)
 if (q/(ident+'.done')).exists():continue
 log=r/'logs'/(ident+'.log');result['active'].append(dict(name=ident,worker=worker,log_tail=log.read_text()[-1500:] if log.exists() else ''))
result['pending']=[p.name for p in q.glob('*.json') if p.name!='FAILED.json']
result['selections']={p.stem:json.loads(p.read_text()) for p in (r/'runs/selection').glob('*.json')}
result['training_progress']=[]
for p in (r/'runs/results').glob('*/metrics.jsonl'):
 lines=p.read_text().splitlines()
 if lines:
  try: result['training_progress'].append(dict(run=p.parent.name,last=json.loads(lines[-1])))
  except json.JSONDecodeError: pass
result['controller_logs']={p.name:p.read_text()[-3000:] for p in (r/'logs').glob('*controller*.log')}
result['evaluation_progress']=[]
for p in (r/'runs/results').glob('*/eval-progress.json'):
 d=json.loads(p.read_text())
 result['evaluation_progress'].append(dict(run=p.parent.name,completed_probes=len(d['records']),seconds=d['seconds'],mtime=p.stat().st_mtime,evaluator_sha256=d['identity']['evaluator_sha256']))
print(json.dumps(result))
