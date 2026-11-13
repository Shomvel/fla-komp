"""Read-only snapshot and same-token validation comparison with each original baseline."""
import datetime,json,subprocess
from pathlib import Path
remote=r'''
import pathlib,json,statistics,datetime
p=pathlib.Path('/gpfs/users/linlizhi/komp-kda-options-340m-20c-20261106')
b=pathlib.Path('/gpfs/users/linlizhi/komp-flame-340m-20c-20261106')
segments={'exact':[('runs',0,2304),('migration-mb32/runtime/runs',2304,2491),('migration-fla/runtime/runs',2491,2492),('migration-fla-mb16/runtime/runs',2492,10**9)],'block':[('runs',0,4096),('migration-mb32/runtime/runs',4096,4458),('migration-fla-mb16/runtime/runs',4458,10**9)]}
def read(path):
 if not path.exists():return []
 rows=[]
 for line in path.read_text().splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:continue
 return rows
out={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':{}}
for kind in ['exact','block']:
 run=p/'runs'/kind;rows=read(run/'metrics.jsonl');val=read(run/'validation.jsonl')
 report={'preflight':json.loads((p/f'evidence/preflight-{kind}.json').read_text()) if (p/f'evidence/preflight-{kind}.json').exists() else None,'latest':rows[-1] if rows else None,'validation':val,'complete':json.loads((run/'complete.json').read_text()) if (run/'complete.json').exists() else None,'checkpoint':json.loads((run/'latest.json').read_text()) if (run/'latest.json').exists() else None}
 if len(rows)>3:report['warm_last20_median_seconds']=statistics.median(r['seconds'] for r in rows[max(3,len(rows)-20):])
 baseline=[]
 for prefix,lo,hi in segments[kind]:baseline.extend(r for r in read(b/prefix/kind/'validation.jsonl') if lo<r['step']<=hi)
 bv={r['input_tokens']:r for r in baseline};nv={r['input_tokens']:r for r in val};common=sorted(bv.keys()&nv.keys())
 report['same_token_validation']=[{'tokens':t,'baseline_ce':bv[t]['cross_entropy'],'combo_ce':nv[t]['cross_entropy'],'delta_combo_minus_baseline':nv[t]['cross_entropy']-bv[t]['cross_entropy']} for t in common]
 out['models'][kind]=report
print(json.dumps(out))
'''
r=subprocess.run(['ssh','linlizhi-bp-dev','python3 -'],input=remote,text=True,capture_output=True,check=True)
d=json.loads(r.stdout);p=Path(__file__).resolve().parent
(p/'latest.json').write_text(json.dumps(d,indent=2))
for kind,row in d['models'].items():
 print(kind,json.dumps({k:v for k,v in row.items() if k not in ['preflight','validation']}))
receipts=[v['preflight'] for v in d['models'].values()]
if all(receipts):
 assert all(x['status']=='passed' for x in receipts)
 assert len({x['initial_model_sha'] for x in receipts})==1,'Exact/Block initialization differs'
 assert len({x['source_sha'] for x in receipts})==1,'Exact/Block source differs'
 print('Both preflights passed with identical initial model SHA and source SHA')
