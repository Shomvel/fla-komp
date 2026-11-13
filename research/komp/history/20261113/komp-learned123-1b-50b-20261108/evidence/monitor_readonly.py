from pathlib import Path
import subprocess,json,datetime
root=Path(__file__).resolve().parent.parent
code=r'''from pathlib import Path
import json,datetime
r=Path('/gpfs/users/linlizhi/komp-learned123-1b-50b-20261108')
def load(p):
 try:return json.loads(p.read_text())
 except Exception as e:return {'error':type(e).__name__}
def tail(p,n=2000):return p.read_text(errors='replace')[-n:] if p.exists() else None
out={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'nodes':{},'evidence':{},'metrics':[]}
for i in range(4):
 n=r/f'nodes/{i}';out['nodes'][str(i)]={'runtime':load(n/'runtime.json') if (n/'runtime.json').exists() else None,'log_tail':tail(n/'worker.log'),'exit':tail(n/'exit.txt',100)}
for p in sorted((r/'evidence').glob('*.json')):
 if p.name.startswith(('preflight','failure','restore','model','environment','operator','properties','nccl')):out['evidence'][p.name]=load(p)
p=r/'runs/block/metrics.jsonl'
if p.exists():
 for line in p.read_text().splitlines()[-3:]:
  try:out['metrics'].append(json.loads(line))
  except json.JSONDecodeError:pass
for name in ['config','latest','complete']:
 p=r/f'runs/block/{name}.json'
 if p.exists():out[name]=load(p)
print(json.dumps(out))
'''
p=subprocess.run(['ssh','linlizhi-bp-dev','python3','-'],input=code,text=True,capture_output=True,check=True,timeout=120);d=json.loads(p.stdout);stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');dest=root/'evidence'/f'monitor-{stamp}.json';dest.write_text(json.dumps(d,indent=2)+'\n');print('snapshot',dest);print('time',d['captured_utc']);print('evidence',list(d['evidence']));print('metrics',d['metrics'])
for i,v in d['nodes'].items():print('node',i,'exit',v['exit'],'tail',v['log_tail'])
