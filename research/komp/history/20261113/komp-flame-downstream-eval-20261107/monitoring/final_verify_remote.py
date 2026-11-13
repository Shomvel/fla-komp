"""Read-only final coverage and pinned artifact verification on BP Dev."""
import datetime, hashlib, json, sys
from pathlib import Path
r=Path('/gpfs/users/linlizhi/komp-flame-downstream-eval-20261107')
sys.path.insert(0,str(r))
from sharding import check_batch, batch_path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
plan=json.loads((r/'sharded/plan.json').read_text());plan_sha=sha(r/'sharded/plan.json')
before=json.loads((r/'recovery-r4g/before.json').read_text())
old={}
for shard in before['shards'].values():
 for group in ['files','batch_hashes']:
  for p,h in shard[group].items():
   assert p not in old or old[p]==h
   old[p]=h
for p,h in old.items():assert sha(r/p)==h,('old_changed',p)
print('Verified old artifacts',len(old),file=sys.stderr,flush=True)
out={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha':plan_sha,'old_artifacts_unchanged':len(old),'old_batches_unchanged':sum(len(v['batch_hashes']) for v in before['shards'].values()),'models':{},'batch_manifest':{}}
assert len(plan['models'])==8 and len(plan['cases'])==23
for model,receipt in plan['models'].items():
 assert sha(Path(receipt['checkpoint']))==receipt['checkpoint_sha'],('checkpoint_changed',model)
 cases={};model_files=set()
 for case in plan['cases']:
  ids=[];scores=[]
  for b in case['batches']:
   p=batch_path(model,b);v=json.loads(p.read_text());check_batch(v,b,plan_sha)
   assert v['model']==model and v['checkpoint_sha']==receipt['checkpoint_sha']
   ids.extend(z['id'] for z in v['results']);scores.extend(z['score'] for z in v['results'])
   key=str(p.relative_to(r));out['batch_manifest'][key]=sha(p);model_files.add(p)
  expected=[i for b in case['batches'] for i in b['ids']]
  assert len(ids)==len(set(ids))==len(expected) and set(ids)==set(expected),(model,case['case'])
  summary=json.loads((r/'sharded/results'/model/f"{case['case']}.summary.json").read_text())
  assert summary['complete'] and summary['count']==len(ids) and abs(summary['score']-sum(scores)/len(scores))<1e-12
  assert summary['plan_sha']==plan_sha and summary['checkpoint_sha']==receipt['checkpoint_sha']
  cases[case['case']]={'count':len(ids),'score':summary['score']}
 assert set((r/'sharded/batches'/model).glob('s*/*.json'))==model_files
 assert sum(z['count'] for z in cases.values())==20457
 for s in range(3):
  for suffix,status in [('acceptance','passed'),('complete','completed')]:
   assert json.loads((r/f'sharded/evidence-v2/{model}-s{s}-{suffix}.json').read_text())['status']==status
 out['models'][model]={'count':20457,'items':23,'batch_files':len(model_files),'checkpoint_sha':receipt['checkpoint_sha'],'checkpoint_hash_verified':True,'tokens':receipt['tokens'],'partial':receipt['partial'],'cases':cases}
 print('Verified model',model,file=sys.stderr,flush=True)
assert len(out['batch_manifest'])==34000
out['total_results']=sum(v['count'] for v in out['models'].values());out['total_items']=sum(v['items'] for v in out['models'].values());out['status']='passed'
print(json.dumps(out))
