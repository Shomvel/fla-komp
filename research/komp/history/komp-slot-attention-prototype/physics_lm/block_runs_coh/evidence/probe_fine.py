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
print(json.dumps(dict(workers=workers,runs=len(list((r/'results-fine/results').glob('*/summary.json'))),
 ready=(r/'fine-preflight-ready.json').exists(),failure=json.loads((r/'fine-failure.json').read_text()) if (r/'fine-failure.json').exists() else None,
 selection=json.loads((r/'fine-selection.json').read_text()) if (r/'fine-selection.json').exists() else None)))
