from pathlib import Path
import json
root=Path('/gpfs/users/linlizhi/komp-coh-trials-20261101')
rows=[]
for i in range(1,17):
 w=root/'workers'/f'r{i}';active=json.loads((w/'active.json').read_text()) if (w/'active.json').exists() else {}
 progress=[json.loads(s) for s in (w/'progress.jsonl').read_text().splitlines()] if (w/'progress.jsonl').exists() else []
 log=w/(active.get('name','')+'.log')
 lines=log.read_text(errors='replace').splitlines() if log.exists() else []
 recent=[s for s in lines if '"step":' in s][-1:]
 if recent:
  raw=recent[0];start=raw.find('{')
  try:
   record=json.loads(raw[start:]);recent=[{k:record[k] for k in ('step','loss','seconds') if k in record}]
  except (ValueError,KeyError):recent=[raw[:160]]
 rows.append(dict(worker=i,commands=len(progress),active=active.get('name'),progress=recent,done=(w/'completion.json').exists(),exit=(w/'exit-code.txt').read_text().strip() if (w/'exit-code.txt').exists() else None))
groups={}
for r in rows:
 name=r['active']
 if name:
  arm=name.split('-seed')[0]; g=groups.setdefault(arm,dict(workers=0,steps=[]));g['workers']+=1
  if r['progress'] and isinstance(r['progress'][0],dict) and 'step' in r['progress'][0]:g['steps'].append(r['progress'][0]['step'])
for g in groups.values():
 steps=g.pop('steps');g['step_range']=[min(steps),max(steps)] if steps else []
print(json.dumps(dict(completed_runs=len(list((root/'results/results').glob('*/summary.json'))),
 diagnostics=len(list((root/'results').glob('*-diagnostic.json'))),completed_workers=sum(r['done'] for r in rows),groups=groups,
 selection=json.loads((root/'selection.json').read_text()) if (root/'selection.json').exists() else None,
 failure=json.loads((root/'failure.json').read_text()) if (root/'failure.json').exists() else None)))
