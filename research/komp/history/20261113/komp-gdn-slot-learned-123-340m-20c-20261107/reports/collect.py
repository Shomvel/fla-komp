"""Read-only new-run health and same-token comparisons; no training mutations."""
import json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
remote='/gpfs/users/linlizhi/'+ROOT.name
script=r'''
import datetime,json,math,pathlib,statistics
r=pathlib.Path(REMOTE_ROOT)
def read(p):return json.loads(p.read_text()) if p.exists() else None
def lines(p):
 out=[]
 if p.exists():
  for s in p.read_text().splitlines():
   try:out.append(json.loads(s))
   except json.JSONDecodeError:pass
 return out
run=r/'runs/block';metrics=lines(run/'metrics.jsonl');val=lines(run/'validation.jsonl')
recent=metrics[-20:];sec=statistics.median(x['seconds'] for x in recent) if recent else None
checkpoint=read(run/'latest.json')
files={p.name:p.stat().st_size for p in pathlib.Path(checkpoint['path']).iterdir()} if checkpoint else None
references={}
base=r.parent/'komp-flame-340m-20c-20261106'
original=[]
for prefix,lo,hi in [('runs',0,4096),('migration-mb32/runtime/runs',4096,4458),('migration-fla-mb16/runtime/runs',4458,10**9)]:
 original.extend(v for v in lines(base/prefix/'block/validation.jsonl') if lo<v['step']<=hi)
references['original_block']=original
for name,folder,kind in [('learned_block','komp-gdn-slot-learned-340m-20c-20261107','block'),('combo_block','komp-kda-options-ff2272-340m-20c-20261106','block'),('kda','kda-flame-340m-20c-20261106','kda')]:
 references[name]=lines(r.parent/folder/'runs'/kind/'validation.jsonl')
comparisons=[]
for v in val:
 row=dict(tokens=v['input_tokens'],new_ce=v['cross_entropy'],comparisons={})
 for name,values in references.items():
  matches=[x for x in values if x['input_tokens']==v['input_tokens']]
  if matches:row['comparisons'][name]=dict(ce=matches[-1]['cross_entropy'],delta_new_minus_reference=v['cross_entropy']-matches[-1]['cross_entropy'])
 comparisons.append(row)
print(json.dumps(dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),latest=metrics[-1] if metrics else None,budget_percent=100*metrics[-1]['supervised_tokens']/6800000000 if metrics else 0,warm_median_seconds=sec,warm_tokens_per_second=524288/sec if sec else None,validation=val[-1] if val else None,checkpoint=checkpoint,checkpoint_files=files,complete=read(run/'complete.json'),preflight=read(r/'evidence/preflight-block.json'),properties=read(r/'evidence/properties-block.json'),continuous_unique=[x['step'] for x in metrics]==list(range(1,len(metrics)+1)) if metrics else None,recent_finite=all(math.isfinite(x[k]) for x in metrics[-200:] for k in ['loss','grad_norm']),same_token_validation=comparisons)))
'''.replace('REMOTE_ROOT',repr(remote))
compile(script, '<remote-monitor>', 'exec')
p=subprocess.run(['ssh','linlizhi-bp-dev','python3','-'],input=script,text=True,capture_output=True,check=True)
result=json.loads(p.stdout)
helper='/Users/shom/.codex/skills/northjob-gpu-test-bp-dev/scripts/northjob.py'
p=subprocess.run(['python3',helper,'jobs','--alias','komp340m-block-gdn-slot-learned123-20261107'],text=True,capture_output=True,check=True)
result['jobs']=json.loads(p.stdout)
(ROOT/'reports/latest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['preflight','properties']},indent=2))
