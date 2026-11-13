"""Immutable ownership, atomic batches and exact-ID aggregation for r4."""
import json,os,hashlib,time,socket
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SHARED=ROOT/'sharded'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_atomic(p,value):
 p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with tmp.open('w') as f:
  f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def lock_worker(name,shard):
 # GPFS flock was empirically local to the mount/node. Atomic mkdir is namespace-wide.
 p=SHARED/'claims-v2'/f'{name}-s{shard}'
 p.parent.mkdir(parents=True,exist_ok=True)
 p.mkdir(exist_ok=False)
 identity=json.loads(Path(os.environ['EVAL_JOB_RECEIPT']).read_text())
 assert identity['alias']==os.environ['EVAL_JOB_ALIAS']
 write_atomic(p/'owner.json',dict(host=os.uname().nodename,pid=os.getpid(),model=name,shard=shard,created=time.time(),job_identity=identity['identity'],job_alias=identity['alias']))
 return p
def batch_path(name,batch):return SHARED/'batches'/name/f"s{batch['shard']}"/f"{batch['key']}.json"
def check_batch(value,batch,plan_sha):
 assert value['plan_sha']==plan_sha and value['batch_key']==batch['key'] and value['shard']==batch['shard']
 assert [r['id'] for r in value['results']]==batch['ids']
 assert len(set(batch['ids']))==len(batch['ids'])
def collect_model(name,plan,plan_sha):
 summaries=[];progress=[]
 for case in plan['cases']:
  rows=[];sources=[]
  for batch in case['batches']:
   p=batch_path(name,batch)
   if not p.exists():continue
   value=json.loads(p.read_text());check_batch(value,batch,plan_sha)
   assert value['model']==name and value['checkpoint_sha']==plan['models'][name]['checkpoint_sha']
   rows.extend(value['results']);sources.append(digest(p))
  ids=[x['id'] for x in rows];expected=[x for b in case['batches'] for x in b['ids']]
  assert len(set(ids))==len(ids) and set(ids)<=set(expected)
  rec=dict(model=name,case=case['case'],count=len(rows),total=len(expected),score=sum(x['score'] for x in rows)/len(rows) if rows else None,complete=len(rows)==len(expected),plan_sha=plan_sha,checkpoint_sha=plan['models'][name]['checkpoint_sha'],tokens=plan['models'][name]['tokens'],partial=plan['models'][name]['partial'])
  progress.append(rec)
  if rec['complete']:
   assert set(ids)==set(expected)
   rec['batch_hashes']=sources;summaries.append(rec)
 return summaries,progress

def publish_batch(p,value):
 # Link a complete fsynced inode into the namespace; existing destination is never replaced.
 p.parent.mkdir(parents=True,exist_ok=True)
 temp=p.with_name(p.name+f'.{socket.gethostname()}.{os.getpid()}.tmp')
 with temp.open('x') as f:
  f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
 try:
  try:os.link(temp,p)
  except FileExistsError:
   old=json.loads(p.read_text())
   for key in ['model','shard','batch_key','plan_sha','checkpoint_sha','results']:assert old[key]==value[key],('existing_batch_mismatch',str(p),key)
 finally:temp.unlink()
