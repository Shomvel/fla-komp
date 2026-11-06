from pathlib import Path
import json
r=Path('/gpfs/users/linlizhi/komp-coh-trials-20261101');workers=[]
for w in sorted((r/'fine-workers').glob('*')):
 if not w.is_dir():continue
 a=json.loads((w/'active.json').read_text()) if (w/'active.json').exists() else {}
 log=w/(a.get('name','')+'.log');lines=log.read_text().splitlines() if log.exists() else []
 progress=[s for s in lines if '"step":' in s][-1:]
 if progress:
  try:
   d=json.loads(progress[0][progress[0].index('{'):]);progress=[{k:d[k] for k in ('step','loss','seconds') if k in d}]
  except ValueError:progress=[]
 workers.append(dict(name=w.name,active=a.get('name'),progress=progress,passes=sum(s.startswith('PASS') for s in lines),done=(w/'completion.json').exists(),exit=(w/'exit-code.txt').read_text().strip() if (w/'exit-code.txt').exists() else None))
groups={}
for w in workers:
 if w['done']:continue
 arm=(w['active'] or 'starting').split('-seed')[0];g=groups.setdefault(arm,dict(workers=0,steps=[]));g['workers']+=1
 if w['progress'] and 'step' in w['progress'][0]:g['steps'].append(w['progress'][0]['step'])
for g in groups.values():
 steps=g.pop('steps');g['steps']=[min(steps),max(steps)] if steps else []
print(json.dumps(dict(groups=groups,runs=len(list((r/'results-fine/results').glob('*/summary.json'))),
 completed_workers=sum(w['done'] for w in workers),
 failure=json.loads((r/'fine-failure.json').read_text()) if (r/'fine-failure.json').exists() else None,
 selection=json.loads((r/'fine-selection.json').read_text()) if (r/'fine-selection.json').exists() else None)))
