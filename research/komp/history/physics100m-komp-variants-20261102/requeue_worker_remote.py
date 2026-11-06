"""Reconcile one deleted worker's claims; invoke through SSH stdin."""
import json, os, shutil, sys, time
from pathlib import Path
r=Path('/gpfs/users/linlizhi/physics100m-komp-variants-20261102'); q=r/'queue'
worker=int(sys.argv[1]); generation=sys.argv[2]
assert worker in range(1,13)
assert generation=='recovery-20261102T0257Z'
assert not (q/'FAILED.json').exists(), 'diagnose failure before recovery'
archive=r/generation/f'r{worker}'; archive.mkdir(parents=True,exist_ok=False)
plans=[]
for claim in sorted(q.glob(f'*.{worker}.running')):
 ident, owner, _=claim.name.rsplit('.',2)
 if (q/(ident+'.done')).exists():continue
 assert ident.startswith(('200-','300-'))
 assert len(list(q.glob(ident+'.*.running')))==1
 assert not (q/(ident+'.json')).exists()
 dest=r/'runs/results'/ident[4:]
 if ident.startswith('200-'):
  complete=(dest/'summary.json').exists()
  if complete:
   summary=json.loads((dest/'summary.json').read_text())
   assert summary['status']=='completed' and summary['config']['phase']=='final'
  else:assert (dest/'checkpoint.pt').stat().st_size>0
 else:
  assert (dest/'summary.json').exists()
  complete=(dest/'heldout.json').exists() and (dest/'generations.json').exists()
  if complete:
   heldout=json.loads((dest/'heldout.json').read_text()); records=json.loads((dest/'generations.json').read_text())
   assert heldout['status']=='completed' and heldout['n_generation']==len(records)==256
 plans.append((claim,ident,complete))
for claim,ident,complete in plans:
 shutil.copyfile(claim,archive/claim.name)
 log=r/'logs'/(ident+'.log')
 if log.exists():shutil.copyfile(log,archive/log.name)
exit_file=r/'evidence'/f'worker-{worker}-exit.json'
if exit_file.exists():shutil.copyfile(exit_file,archive/exit_file.name)
result=dict(worker=worker,at=time.time(),requeued=[],reconciled=[])
for claim,ident,complete in plans:
 if complete:
  with (q/(ident+'.done')).open('x') as f:json.dump(dict(name=ident,worker=str(worker),exit_code=0,finished=time.time(),reconciled_from='complete result JSON after exact worker deletion'),f)
  result['reconciled'].append(ident)
 else:
  # Atomic rename keeps the spec continuously visible to the controller.
  claim.rename(q/(ident+'.json'))
  result['requeued'].append(ident)
(archive/'requeue.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
