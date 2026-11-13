"""Fused activation comparison including BF16 intermediate-rounding witnesses."""
import json,os,sys
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'snapshot-ff'))
from fla.modules.rounded_swiglu import rounded_swiglu
torch.manual_seed(761);results=[]
for dtype in [torch.bfloat16,torch.float16]:
 for shape in [(65536,),(16,2048,3072)]:
  a=torch.randn(shape,device='cuda',dtype=dtype)*2;b=torch.randn_like(a);dy=torch.randn_like(a)*.01
  if len(shape)==1:a=torch.linspace(-100,100,65536,device='cuda').to(dtype)
  a.requires_grad_();b.requires_grad_()
  ref=F.silu(a)*b;ga,gb=torch.autograd.grad(ref,(a,b),dy)
  got=rounded_swiglu(a,b);ca,cb=torch.autograd.grad(got,(a,b),dy)
  rows=[]
  for name,x,y in [('out',got,ref),('dgate',ca,ga),('dup',cb,gb)]:
   assert torch.isfinite(x).all(),(dtype,shape,name)
   err=(x.float()-y.float()).square().mean().sqrt()/y.float().square().mean().sqrt().clamp_min(1e-30)
   row=dict(name=name,relative_rms=float(err),max_abs=float((x.float()-y.float()).abs().max()),different=int(torch.count_nonzero(x!=y)))
   assert err<=5e-4,row
   rows.append(row)
  results.append(dict(dtype=str(dtype),shape=shape,metrics=rows,passed=True));print(json.dumps(results[-1]),flush=True)
  del a,b,dy,ref,got,ga,gb,ca,cb;torch.cuda.empty_cache()
(ROOT/'evidence/ff-differential.json').write_text(json.dumps(dict(passed=True,results=results),indent=2)+'\n')
