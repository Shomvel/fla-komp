"""Queue final seeds only after all nine validation-only LR/WD trials complete."""
import json,time,os
from pathlib import Path
r=Path('/gpfs/users/linlizhi/physics100m-komp-variants-20261102');q=r/'queue';runs=r/'runs';selections=runs/'selection';selections.mkdir(parents=True,exist_ok=True)
for_loop=True
while for_loop:
 if (q/'STOP').exists():break
 if (q/'FAILED.json').exists():time.sleep(15);continue
 complete=0
 for task in ['mano','lano','depo2','brevo2','capo']:
  for kind in ['komp_learned','komp_gdn_slot','komp_learned_gdn_slot']:
   target=selections/f'{task}-{kind}.json';candidates=[]
   for lr in [.00025,.0005,.001]:
    for wd in [0.,.01,.1]:
     name=f'{task}-{kind}-tune-lr{lr:g}-wd{wd:g}-seed731';f=runs/'results'/name/'summary.json'
     if f.exists():
      j=json.loads(f.read_text());assert j['status']=='completed';candidates.append(dict(lr=lr,wd=wd,val_ce=j['val']['ce'],run=name))
   if len(candidates)!=9:continue
   best=min(candidates,key=lambda x:(x['val_ce'],x['lr'],x['wd']))
   if not target.exists():target.write_text(json.dumps(dict(**best,criterion='minimum final validation CE at 10M tokens',candidates=candidates),indent=2))
   for seed in [731,732]:
    name=f"{task}-{kind}-final-lr{best['lr']:g}-wd{best['wd']:g}-seed{seed}"
    summary=runs/'results'/name/'summary.json';heldout=summary.with_name('heldout.json')
    ident='200-'+name
    if not any(q.glob(ident+'.*')):
     spec=dict(command=['python','train.py','--kind',kind,'--task',task,'--phase','final','--seed',str(seed),'--lr',str(best['lr']),'--wd',str(best['wd'])],requires=[str(r/'data'/task/'manifest.json')])
     temp=q/(ident+'.tmp');temp.write_text(json.dumps(spec));temp.rename(q/(ident+'.json'))
    ident='300-'+name
    if summary.exists() and not any(q.glob(ident+'.*')):
     spec=dict(command=['python','../evaluate.py','--run',name],requires=[str(summary),str(target)])
     temp=q/(ident+'.tmp');temp.write_text(json.dumps(spec));temp.rename(q/(ident+'.json'))
    if heldout.exists() and heldout.with_name('generations.json').exists():complete+=1
 if complete==30:
  (r/'evidence/ALL_COMPLETE.json').write_text(json.dumps(dict(final_runs=30)));(q/'STOP').touch();break
 time.sleep(15)
