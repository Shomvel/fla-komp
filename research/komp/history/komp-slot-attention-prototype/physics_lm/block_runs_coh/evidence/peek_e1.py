from pathlib import Path
import json,statistics
r=Path('/gpfs/users/linlizhi/komp-coh-trials-20261101/results')
for arm in ('coh_fifo_off','coh_fifo_99','coh_fifo_95','coh_fifo_80'):
 rows=[json.loads(p.read_text()) for p in (r/'results').glob(arm+'-seed*/summary.json')]
 if rows:print(json.dumps(dict(arm=arm,n=len(rows),ce=statistics.mean(x['confirm']['mean_content_ce'] for x in rows),depo=statistics.mean(x['generation']['depo']['valid'] for x in rows),brevo=statistics.mean(x['generation']['brevo']['valid'] for x in rows))))
 p=r/f'{arm}-seed731-diagnostic.json'
 if p.exists():
  d=json.loads(p.read_text());print(json.dumps(dict(arm=arm,condition_medians=[statistics.median(x['condition'] for x in d['rows'] if x['layer']==i) for i in range(4)])))
