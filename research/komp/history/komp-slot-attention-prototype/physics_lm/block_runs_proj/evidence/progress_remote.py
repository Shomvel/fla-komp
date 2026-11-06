from pathlib import Path
import json,collections,statistics,time
R=Path('/gpfs/users/linlizhi/komp-proj-swa-20261101')
by=collections.defaultdict(list)
for p in (R/'results/results').glob('*/summary.json'):
 s=json.loads(p.read_text());by[s['config']['architecture']].append(s)
print(json.dumps(dict(time=time.time(),ready=(R/'preflight-ready.json').exists(),failure=json.loads((R/'failure.json').read_text()) if (R/'failure.json').exists() else None,
 completed=sum(map(len,by.values())),arms={a:dict(n=len(v),ce=statistics.mean(s['confirm']['mean_content_ce'] for s in v),depo_pct=100*statistics.mean(s['generation']['depo']['valid'] for s in v),brevo_pct=100*statistics.mean(s['generation']['brevo']['valid'] for s in v)) for a,v in sorted(by.items())},
 main_done=sum((R/'workers'/f'r{i}/main-done.json').exists() for i in range(1,17)),done=sum((R/'workers'/f'r{i}/completion.json').exists() for i in range(1,17))),indent=2))
