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
print(json.dumps(dict(workers=rows,completed_runs=len(list((root/'results/results').glob('*/summary.json'))),diagnostics=len(list((root/'results').glob('*-diagnostic.json'))),
 selection=json.loads((root/'selection.json').read_text()) if (root/'selection.json').exists() else None,
 failure=json.loads((root/'failure.json').read_text()) if (root/'failure.json').exists() else None)))
