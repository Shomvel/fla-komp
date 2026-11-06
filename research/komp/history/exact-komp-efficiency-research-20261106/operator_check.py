"""GPU operator gate including checkpoint replay and state-output gradients.
Stops on first failing case. Run once, diagnose statically before rerunning.
"""
import argparse,importlib.util,json,sys
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'candidate'))
from geometry_cache import geometry_contexts

def load(name,variant):
 folder=ROOT/variant/'snapshot/komp_slot/exact_komp'
 spec=importlib.util.spec_from_file_location(name,folder/'__init__.py',submodule_search_locations=[str(folder)])
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod.exact_komp

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--prune-gc',action='store_true');a=p.parse_args()
 import os
 os.environ['EXACT_KOMP_PRUNE_GC']='1' if a.prune_gc else '0'
 base=load('base_exact','baseline');candidate=load('candidate_exact','candidate');rows=[]
 for dtype in (torch.float32,torch.bfloat16):
  for t in (1,15,16,17,496,497,511,512,513,2048):
   torch.manual_seed(42+t);b,h,m,d=1,2,32,64
   raw=[torch.randn(b,t,h,d,device='cuda',dtype=dtype) for _ in range(3)]
   raw += [torch.full((b,t,h,m),-.005,device='cuda'),torch.full((b,t,h),.2,device='cuda'),torch.randn(b,h,m,d,device='cuda'),torch.randn(b,h,m,d,device='cuda')*.1]
   # Same cotangents exercise output plus both differentiable final states.
   cot=[torch.randn(b,t,h,d,device='cuda',dtype=dtype),torch.randn(b,h,m,d,device='cuda'),torch.randn(b,h,m,d,device='cuda')]
   def run(fn,cache):
    xs=[x.detach().clone().requires_grad_() for x in raw]
    def f(*x):
     o,states,_=fn(*x[:5],(x[5],x[6]),eviction='fifo',start_position=7,compute_chunk=16)
     return o,states[0],states[1]
    options=dict(context_fn=geometry_contexts) if cache else {}
    ys=checkpoint(f,*xs,use_reentrant=False,**options)
    grads=torch.autograd.grad(ys,xs,cot)
    return [x.detach() for x in ys],grads
   gold,gg=run(base,False);actual,ag=run(candidate,True)
   limits=(.005,.005) if dtype==torch.float32 else (.035,.04)
   checks=[]
   for name,x,y in zip(['output','keys','state','dq','dk','dv','dg','dbeta','dk0','ds0'],[*actual,*ag],[*gold,*gg]):
    x=x.double();y=y.double();delta=x-y
    rms=float(delta.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-7));mx=float(delta.abs().max()/y.abs().max().clamp_min(1));absolute=float(delta.abs().max())
    checks.append(dict(name=name,rms=rms,normalized_max=mx,absolute=absolute,passed=bool(torch.isfinite(x).all()) and ((rms<limits[0] and mx<limits[1]) or absolute<=1e-7)))
   row=dict(t=t,dtype=str(dtype),checks=checks,passed=all(c['passed'] for c in checks));rows.append(row);a.out.write_text(json.dumps(rows,indent=2))
   assert row['passed'],f'First failed class: dtype={dtype}, T={t}; inspect before rerunning'
 print('passed',len(rows),'operator cases; not a full model acceptance')
if __name__=='__main__':main()
