"""Run only after old r3 job is confirmed absent; preserve all old bytes."""
import json,hashlib,shutil,importlib.util,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_eval',ROOT/'evaluate.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
from sharding import write_atomic,batch_path,check_batch,digest
out=ROOT/'sharded';out.mkdir(exist_ok=False)
archive=out/'r3-archive';archive.mkdir()
for name in ['evaluate.py','job.yaml','README.md']:
 shutil.copy2(ROOT/name,archive/name)
shutil.copytree(ROOT/'results',archive/'results')
shutil.copytree(ROOT/'evidence',archive/'evidence')
manifest=json.loads((ROOT/'evidence/requests-manifest.json').read_text())
order={'phonebook':0,'based_swde':1,'based_fda':2,'niah':3,'based_nq_1024':4,'based_drop':5}
cases=sorted(manifest['cases'],key=lambda c:(next((v for k,v in order.items() if c['case'].startswith(k)),6),c['case']))
plan=dict(version=4,shards=3,original_eval_sha=digest(ROOT/'evaluate.py'),request_manifest_sha=digest(ROOT/'evidence/requests-manifest.json'),protocol_audit_sha=digest(ROOT/'evidence/generation-contract-audit.json'),models={},cases=[])
for name,*_ in mod.MODELS:
 receipt=json.loads((ROOT/'evidence'/f'{name}-load.json').read_text())
 assert receipt['eval_script_sha']==plan['original_eval_sha']
 plan['models'][name]=receipt
for case in cases:
 p=ROOT/'requests'/f"{case['case']}.jsonl";assert digest(p)==case['sha256']
 pending=sorted([json.loads(x) for x in p.read_text().splitlines()],key=lambda r:len(mod.context_ids(r,2048)))
 batches=[]
 while pending:
  batch=[pending.pop(0)]
  while pending and len(batch)<8 and len(mod.context_ids(pending[0],2048))==len(mod.context_ids(batch[0],2048)) and pending[0]['generation']==batch[0]['generation'] and (len(batch)+1)*(len(mod.context_ids(pending[0],2048))+128)<=32768:batch.append(pending.pop(0))
  n=len(batches);batches.append(dict(key=f"{case['case']}-{n:05d}",shard=n%3,ids=[r['id'] for r in batch],tokens=len(mod.context_ids(batch[0],2048))))
 assert len({x for b in batches for x in b['ids']})==case['count']
 plan['cases'].append(dict(**case,batches=batches))
write_atomic(out/'plan.json',plan);plan_sha=digest(out/'plan.json')
audit=dict(plan_sha=plan_sha,models={},incomplete_tails=[])
for name in plan['models']:
 inherited=0;partial_rows=0
 for case in plan['cases']:
  p=archive/'results'/name/f"{case['case']}.jsonl"
  old={}
  if p.exists():
   lines=p.read_bytes().splitlines(keepends=True)
   for i,line in enumerate(lines):
    if not line.endswith(b'\n'):
     assert i==len(lines)-1;audit['incomplete_tails'].append(dict(path=str(p),bytes=len(line)));continue
    row=json.loads(line);assert row['id'] not in old;old[row['id']]=row
  expected={x for b in case['batches'] for x in b['ids']};assert set(old)<=expected
  for batch in case['batches']:
   present=[old[x] for x in batch['ids'] if x in old]
   if len(present)==len(batch['ids']):
    value=dict(model=name,shard=batch['shard'],batch_key=batch['key'],plan_sha=plan_sha,checkpoint_sha=plan['models'][name]['checkpoint_sha'],results=present,inherited=True,source_file=str(p),source_sha=digest(p))
    check_batch(value,batch,plan_sha);write_atomic(batch_path(name,batch),value);inherited+=len(present)
   elif present:
    write_atomic(out/'partial-reference'/name/f"{batch['key']}.json",present);partial_rows+=len(present)
 audit['models'][name]=dict(inherited_samples=inherited,partial_reference_samples=partial_rows)
write_atomic(out/'migration-audit.json',audit)
print(json.dumps(audit,indent=2))
