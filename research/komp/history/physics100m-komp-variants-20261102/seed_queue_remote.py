"""Seed 135 validation-only LR/WD trials, gated on all three acceptance tickets."""
import itertools,json,os,time
from pathlib import Path
r=Path('/gpfs/users/linlizhi/physics100m-komp-variants-20261102');q=r/'queue'
kinds=['komp_learned','komp_gdn_slot','komp_learned_gdn_slot']
tickets=[str(r/'runs/preflight'/f'{kind}.json') for kind in kinds]
assert not list(q.glob('100-*'))
for task,kind,lr,wd in itertools.product(['mano','lano','depo2','brevo2','capo'],kinds,[.00025,.0005,.001],[0.,.01,.1]):
 name=f'100-{task}-{kind}-lr{lr:g}-wd{wd:g}'
 spec=dict(command=['python','train.py','--kind',kind,'--task',task,'--phase','tune','--seed','731','--lr',str(lr),'--wd',str(wd)],requires=tickets+[str(r/'data'/task/'manifest.json')])
 with (q/(name+'.json')).open('x') as f:json.dump(spec,f)
print(json.dumps(dict(at=time.time(),tuning_jobs=135,final_jobs_planned=30,preflight_requires=tickets)))
