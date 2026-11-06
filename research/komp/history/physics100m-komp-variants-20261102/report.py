"""Report actual final JSON from one or more monitor snapshots (never tuning)."""
import argparse,csv,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('snapshots',nargs='+',type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
a.out.mkdir(exist_ok=True,parents=True)
rows={}
for snap in a.snapshots:
 for r in json.loads(snap.read_text())['final']:
  s=r['summary'];c=s['config'];assert c['phase']=='final'
  h=r.get('heldout');g=r.get('generations')
  if h is not None and g is not None:
   assert h['status']=='completed' and h['n_generation']==len(g)==256
   assert h['generation_accuracy']==sum(v['valid'] for v in g)/len(g)
  else:h=None
  row=dict(task=c['task'],kind=c['model']['kind'],seed=c['seed'],tokens=s['tokens'],lr=c['lr'],wd=c['wd'],parameters=c['model']['parameters'],val_ce=s['val']['ce'],test_ce=h['test']['ce'] if h else None,generation_percent=100*h['generation_accuracy'] if h else None,train_seconds=s['train_seconds'],eval_seconds=h['seconds'] if h else None)
  rows[c['task'],c['model']['kind'],c['seed']]=row
rows=sorted(rows.values(),key=lambda r:(r['task'],r['kind'],r['seed']))
(a.out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
lines=['All rows are final 100M-token runs. Generation accuracy uses 256 probes per seed. Missing evaluation is not zero.','', '| Task | Model | Seed | Validation CE | Test CE | Generation % | LR | WD | Train seconds |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
fmt=lambda x:'pending' if x is None else f'{x:.4f}'
for r in rows:
 lines.append(f"| {r['task']} | {r['kind']} | {r['seed']} | {fmt(r['val_ce'])} | {fmt(r['test_ce'])} | {fmt(r['generation_percent'])} | {r['lr']:g} | {r['wd']:g} | {r['train_seconds']:.1f} |")
lines+=['','Paired differences A minus original KOMP. Two seeds are descriptive; no significance claims.','', '| Task | Variant | Paired n | Δ test CE | Δ generation pp |','|---|---|---:|---:|---:|']
lookup={(r['task'],r['kind'],r['seed']):r for r in rows}
for task in ['mano','lano','depo2','brevo2','capo']:
 for kind in ['komp_learned','komp_gdn_slot','komp_learned_gdn_slot']:
  pairs=[]
  for seed in [731,732]:
   x=lookup.get((task,kind,seed));b=lookup.get((task,'komp',seed))
   if x and b and x['test_ce'] is not None and b['test_ce'] is not None:pairs.append((x,b))
  if pairs:lines.append(f"| {task} | {kind} | {len(pairs)} | {statistics.mean(x['test_ce']-b['test_ce'] for x,b in pairs):+.4f} | {statistics.mean(x['generation_percent']-b['generation_percent'] for x,b in pairs):+.4f} |")
(a.out/'TABLE.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(final_rows=len(rows),complete_eval=sum(r['test_ce'] is not None for r in rows),out=str(a.out))))
