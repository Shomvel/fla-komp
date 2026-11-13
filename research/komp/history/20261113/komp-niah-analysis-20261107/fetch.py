import subprocess,gzip,json
from pathlib import Path
out=Path(__file__).resolve().parent
code=r'''
from pathlib import Path
import json,gzip,sys,datetime
r=Path('/gpfs/users/linlizhi/komp-flame-downstream-eval-20261107')
requests=[];results=[]
for p in sorted((r/'requests').glob('niah_single_*.jsonl')):
 for line in p.read_text().splitlines():
  x=json.loads(line);x.pop('doc');requests.append(x)
for m in ['kda','value','exact123','exact_partial']:
 for p in sorted((r/'sharded/batches'/m).glob('s*/niah_single_*.json')):
  b=json.loads(p.read_text())
  for x in b['results']:results.append(dict(model=m,**x))
files={str(p.relative_to(r/'source/lm_eval/tasks/ruler')):p.read_text() for p in (r/'source/lm_eval/tasks/ruler').glob('*.py')}
for p in (r/'source/lm_eval/tasks/ruler').glob('niah_single*.yaml'):files[p.name]=p.read_text()
obj=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),requests=requests,results=results,source=files)
sys.stdout.buffer.write(gzip.compress(json.dumps(obj).encode()))
'''
p=subprocess.run(['ssh','linlizhi-bp-dev','python3','-'],input=code.encode(),capture_output=True,check=True)
(out/'raw.json.gz').write_bytes(p.stdout)
d=json.loads(gzip.decompress(p.stdout));print('requests',len(d['requests']),'results',len(d['results']),'compressed bytes',len(p.stdout))
(out/'source').mkdir(exist_ok=True)
for k,v in d['source'].items():(out/'source'/k).write_text(v)
subprocess.run(['scp','-q','linlizhi-bp-dev:/gpfs/shared_data/qiji/huggingface.co/fla-hub/transformer-1.3B-100B/tokenizer.json',str(out/'tokenizer.json')],check=True)
