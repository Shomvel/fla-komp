"""CPU tensor-by-tensor numerical gate. Original strict thresholds, no waiver."""
import argparse,json,math
from pathlib import Path
import torch

def row(name,actual,reference):
 a=actual.double();b=reference.double();delta=a-b
 finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
 ref2=float(b.square().sum());err2=float(delta.square().sum())
 rms=float(delta.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-7))
 mx=float(delta.abs().max()/b.abs().max().clamp_min(1));absolute=float(delta.abs().max())
 return dict(name=name,finite=finite,rms=rms,normalized_max=mx,absolute=absolute,reference_squared_norm=ref2,error_squared_norm=err2,passed=finite and ((rms<.035 and mx<.04) or absolute<=1e-7))

def main():
 p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();checks=[];reports=[]
 left=sorted(a.baseline.glob('check-rank*.pt'));right=sorted(a.candidate.glob('check-rank*.pt'));assert left and [x.name for x in left]==[x.name for x in right]
 for lp,rp in zip(left,right):
  rank=lp.stem.split('rank')[1]
  lr=json.loads((a.baseline/f'result-rank{rank}.json').read_text());rr=json.loads((a.candidate/f'result-rank{rank}.json').read_text())
  for key in ('fixture_sha','model_checkpoint_sha','world'):assert lr[key]==rr[key],key
  assert lr['args']['case']==rr['args']['case']
  l=torch.load(lp,map_location='cpu',weights_only=True);r=torch.load(rp,map_location='cpu',weights_only=True)
  loss_delta=abs(l['loss']-r['loss']);assert math.isfinite(loss_delta)
  reports.append(dict(rank=rank,loss_absolute=loss_delta,passed=loss_delta<.005))
  for category in ('grad','delta'):
   assert l[category].keys()==r[category].keys()
   for name,ref in l[category].items():checks.append(row(f'rank{rank}/{category}/{name}',r[category][name],ref))
  del l,r
 result=dict(status='passed' if all(x['passed'] for x in checks+reports) else 'failed',policy=dict(rms=.035,normalized_max=.04,nearzero_absolute=1e-7,loss_absolute=.005),losses=reports,checks=checks,failures=[x for x in checks if not x['passed']],scope='Same frozen native FLA model, B16x2, same checkpoint/inputs; not a comparison to legacy/B32 and not long-horizon quality proof')
 a.out.write_text(json.dumps(result,indent=2));print(result['status'],'failed tensors',len(result['failures']));raise SystemExit(0 if result['status']=='passed' else 1)
if __name__=='__main__':main()
