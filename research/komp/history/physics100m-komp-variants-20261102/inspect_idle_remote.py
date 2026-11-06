import json,sys,time
from pathlib import Path
r=Path('/gpfs/users/linlizhi/physics100m-20261101');q=r/'queue';worker=int(sys.argv[1])
active=[]
for claim in q.glob(f'*.{worker}.running'):
 ident=claim.name.rsplit('.',2)[0]
 if not (q/(ident+'.done')).exists():active.append(ident)
pending=[p.name for p in q.glob('*.json') if p.name!='FAILED.json']
result=dict(at=time.time(),worker=worker,active=active,pending=pending,failure=(q/'FAILED.json').exists())
print(json.dumps(result));assert not active and not pending and not result['failure'],'worker may be busy'
