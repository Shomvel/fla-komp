"""Read-only progress snapshot for the value-decay Block run."""
import json,subprocess,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
remote='/gpfs/users/linlizhi/'+ROOT.name
script='''import json,pathlib,statistics
r=pathlib.Path(%r)
def read(p):
 return json.loads(p.read_text()) if p.exists() else None
def lines(p):
 out=[]
 if p.exists():
  for s in p.read_text().splitlines():
   try:out.append(json.loads(s))
   except json.JSONDecodeError:pass
 return out
metrics=lines(r/'runs/block/metrics.jsonl');val=lines(r/'runs/block/validation.jsonl')
recent=metrics[-20:];sec=statistics.median(x['seconds'] for x in recent) if recent else None
print(json.dumps(dict(latest=metrics[-1] if metrics else None,budget_percent=100*metrics[-1]['supervised_tokens']/6800000000 if metrics else 0,warm_median_seconds=sec,warm_tokens_per_second=524288/sec if sec else None,validation=val[-1] if val else None,checkpoint=read(r/'runs/block/latest.json'),complete=read(r/'runs/block/complete.json'),preflight=read(r/'evidence/preflight-block.json'))))
'''%remote
p=subprocess.run(['ssh','linlizhi-bp-dev','python3','-'],input=script,text=True,capture_output=True,check=True)
result=json.loads(p.stdout)
helper='/Users/shom/.codex/skills/northjob-gpu-test-bp-dev/scripts/northjob.py'
p=subprocess.run(['python3',helper,'jobs','--alias','komp340m-block-value-decay-20261106'],text=True,capture_output=True,check=True)
result['jobs']=json.loads(p.stdout);result['timestamp_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(ROOT/'reports/latest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
