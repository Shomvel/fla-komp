from pathlib import Path
import json,hashlib,datetime,os,errno
r=Path('/gpfs/users/linlizhi/komp-flame-downstream-eval-20261107');rec=r/'recovery-r4e'
before=json.loads((rec/'before.json').read_text());out={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'shards':{}}
for s in [0,1,2]:
 prior=before['shards'][str(s)];receipt=json.loads((r/f'sharded/job-r4e-s{s}.json').read_text())
 for p,h in prior['files'].items():assert hashlib.sha256((r/p).read_bytes()).hexdigest()==h,p
 for p,h in prior['batch_hashes'].items():assert hashlib.sha256((r/p).read_bytes()).hexdigest()==h,p
 probes=[];workers={}
 for name in ['block123','learned','learned123','block']:
  claim=r/f'sharded/claims-v2/{name}-s{s}';owner=json.loads((claim/'owner.json').read_text());assert owner['job_identity']==receipt['identity']
  try:claim.mkdir()
  except FileExistsError:pass
  else:raise AssertionError('duplicate claim accepted')
  probe=claim/'publish-probe.json';assert probe.exists();h=hashlib.sha256(probe.read_bytes()).hexdigest()
  try:os.link(probe,probe)
  except FileExistsError:pass
  else:raise AssertionError('duplicate publish accepted')
  assert hashlib.sha256(probe.read_bytes()).hexdigest()==h
  probes.append(name)
  acc=r/f'sharded/evidence-v2/{name}-s{s}-acceptance.json';progress=r/f'sharded/evidence-v2/{name}-s{s}-progress.json'
  a=json.loads(acc.read_text()) if acc.exists() and acc.stat().st_mtime>=owner['created'] else None
  p=json.loads(progress.read_text()) if progress.exists() and progress.stat().st_mtime>=owner['created'] else None
  additions=[q for q in (r/f'sharded/batches/{name}/s{s}').glob('*.json') if str(q.relative_to(r)) not in prior['batch_hashes']]
  workers[name]={'acceptance':a['status'] if a else None,'new_batches':len(additions),'new_samples':sum(len(json.loads(q.read_text())['results']) for q in additions),'progress':p}
 for name in ['kda','value','exact123','exact_partial']:
  assert json.loads((r/f'sharded/evidence-v2/{name}-s{s}-complete.json').read_text())['status']=='completed'
  assert json.loads((r/f'sharded/claims-v2/{name}-s{s}/owner.json').read_text())['job_identity']!=receipt['identity']
 out['shards'][str(s)]=dict(new_identity=receipt['identity'],old_batches_unchanged=len(prior['batch_hashes']),cross_client_claim_and_publish_probes=probes,workers=workers,completed_models_not_restarted=['kda','value','exact123','exact_partial'])
out['status']='passed' if all(w['acceptance']=='passed' and w['new_batches']>0 for s in out['shards'].values() for w in s['workers'].values()) else 'pending_runtime_progress'
(rec/'recovery-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
