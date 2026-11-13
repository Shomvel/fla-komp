"""Existing FLA RMSNorm versus production PyTorch RMSNorm for the two real row widths."""
import json,sys,time,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'snapshot-norm'))
from fla.modules.komp_rmsnorm import OptimizedRMSNorm
results=[]
torch.manual_seed(762)
for shape in [(1048576,64),(32768,2048)]:
 x=torch.randn(shape,device='cuda',requires_grad=True);dy=torch.randn_like(x)*.01
 ref=torch.nn.RMSNorm(shape[-1],eps=1e-6,device='cuda')
 got=OptimizedRMSNorm(shape[-1],eps=1e-6,device='cuda');got.load_state_dict(ref.state_dict())
 a=ref(x);ag=torch.autograd.grad(a,(x,ref.weight),dy)
 b=got(x);bg=torch.autograd.grad(b,(x,got.weight),dy)
 rows=[]
 for name,u,v in zip(['out','dx','dw'],(b,*bg),(a,*ag)):
  err=float((u-v).square().mean().sqrt()/v.square().mean().sqrt().clamp_min(1e-30))
  assert torch.isfinite(u).all() and err<=1e-4,(shape,name,err)
  rows.append(dict(name=name,relative_rms=err,max_abs=float((u-v).abs().max())))
 timing={}
 for name,mod in [('pytorch',ref),('fla',got)]:
  def step():
   x.grad=None;mod.zero_grad(set_to_none=True);y=mod(x);y.backward(dy)
  for _ in range(3):step()
  torch.cuda.synchronize();elapsed=[]
  for _ in range(10):
   start=time.perf_counter();step();torch.cuda.synchronize();elapsed.append(time.perf_counter()-start)
  timing[name]=statistics.median(elapsed)
 results.append(dict(shape=shape,metrics=rows,seconds=timing,passed=True));print(json.dumps(results[-1]),flush=True)
 del x,dy,ref,got,a,ag,b,bg;torch.cuda.empty_cache()
(ROOT/'evidence/norm-differential.json').write_text(json.dumps(dict(passed=True,results=results),indent=2)+'\n')
