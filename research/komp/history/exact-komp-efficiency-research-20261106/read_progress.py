"""Read-only compact progress for this task's own GPFS output."""
import json,subprocess
source='''
from pathlib import Path
import json
root=Path('/gpfs/users/linlizhi/exact-komp-efficiency-research-20261106')
p=root/'stage1-r2.log'
print('LOG_TAIL',p.read_text()[-2400:] if p.exists() else 'not created')
r=root/'results-stage1'
if r.exists():
 print('RESULT_DIRS',sorted(p.name for p in r.iterdir()))
 op=r/'operator.json'
 if op.exists():
  cases=json.loads(op.read_text());print('OPERATOR',len(cases), 'last',cases[-1]['t'],cases[-1]['dtype'],cases[-1]['passed'])
 for p in sorted(r.glob('*comparison.json')):
  v=json.loads(p.read_text());print('COMPARISON',p.name,v['status'],len(v['failures']))
 for p in sorted(r.glob('bench-*/result-rank0.json')):
  v=json.loads(p.read_text());warm=[x['update_wall_seconds'] for x in v['rows'] if x['phase']=='warm']
  import statistics
  print('BENCH',p.parent.name,statistics.median(warm) if warm else None)
'''
raise SystemExit(subprocess.run(['ssh','linlizhi-bp-dev','python3 -'],input=source,text=True).returncode)
