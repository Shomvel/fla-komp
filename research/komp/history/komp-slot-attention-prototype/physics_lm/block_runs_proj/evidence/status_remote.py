from pathlib import Path
import json
R=Path('/gpfs/users/linlizhi/komp-proj-swa-20261101')
result=dict(ready=(R/'preflight-ready.json').exists(),failure=None,completed={},workers=[])
if (R/'failure.json').exists():result['failure']=json.loads((R/'failure.json').read_text())
for p in sorted((R/'results/results').glob('*/summary.json')):
 s=json.loads(p.read_text());kind=s['config']['architecture']
 result['completed'].setdefault(kind,[]).append(dict(seed=s['config']['seed'],ce=s['confirm']['mean_content_ce'],depo=s['generation']['depo']['valid'],brevo=s['generation']['brevo']['valid']))
for w in sorted((R/'workers').glob('r*')):
 if not (w/'active.json').exists():continue
 a=json.loads((w/'active.json').read_text());log=w/(a['name']+'.log')
 lines=log.read_text().splitlines() if log.exists() else []
 result['workers'].append(dict(worker=w.name,done=(w/'completion.json').exists(),main_done=(w/'main-done.json').exists(),active=a['name'],tail=lines[-8:] if result['failure'] else [l[:200] for l in lines[-1:]]))
print(json.dumps(result,indent=2))
