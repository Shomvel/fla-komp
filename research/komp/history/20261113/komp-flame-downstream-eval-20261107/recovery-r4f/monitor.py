from pathlib import Path
import json,hashlib,datetime
r=Path('/gpfs/users/linlizhi/komp-flame-downstream-eval-20261107');rec=r/'recovery-r4f'
before=json.loads((rec/'before.json').read_text());out={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'shards':{}}
for s in [1]:
 receipt=json.loads((r/f'sharded/job-r4f-s{s}.json').read_text());migration=rec/f'migration-s{s}.json';v={'migration':json.loads(migration.read_text()) if migration.exists() else None,'workers':{},'launcher':(r/f'logs/launcher-r4f-s{s}.log').read_text()[-3000:] if (r/f'logs/launcher-r4f-s{s}.log').exists() else None}
 old=before['shards'][str(s)]['batch_hashes']
 for name,i in [('block123',1),('learned',3),('learned123',4),('block',5)]:
  owner=r/f'sharded/claims-v2/{name}-s{s}/owner.json';o=json.loads(owner.read_text()) if owner.exists() else None
  worker={'owner':o,'new_owner':o and o['job_identity']==receipt['identity']}
  for suffix in ['acceptance','failure','progress','complete']:
   p=r/f'sharded/evidence-v2/{name}-s{s}-{suffix}.json'
   if p.exists() and worker['new_owner'] and p.stat().st_mtime>=o['created']:
    x=json.loads(p.read_text());worker[suffix]={k:y for k,y in x.items() if k not in ['receipt','exact_length_repeatability','smoke','traceback']}
  additions=[p for p in (r/f'sharded/batches/{name}/s{s}').glob('*.json') if str(p.relative_to(r)) not in old]
  worker['new_batches']=len(additions);worker['new_samples']=sum(len(json.loads(p.read_text())['results']) for p in additions)
  log=r/f'logs/model-r4f-s{s}-{i}.log';worker['log_tail']=log.read_text(errors='replace')[-700:] if log.exists() else None
  v['workers'][name]=worker
 out['shards'][str(s)]=v
print(json.dumps(out))
