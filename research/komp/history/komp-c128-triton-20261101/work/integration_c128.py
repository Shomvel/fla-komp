"""C128 score-gate chain-rule and serial scan acceptance at training-shaped batch-head count."""
import json
from pathlib import Path
import torch
from komp_slot.block_fast import block_komp_fast
from komp_slot.block_flex import block_komp_flex
from validate_block_komp import inputs,rms
rows=[]
torch.set_num_threads(4)
for dtype in (torch.float32,torch.bfloat16):
 x=inputs(8,257,8,64,64,32,dtype,732,slot=True)
 a=[y.detach().clone().requires_grad_() for y in x];z=[y.detach().double().requires_grad_() for y in x]
 scores=torch.randn(8,257,8,device='cuda')*.2
 sa=scores.clone().requires_grad_();sz=scores.double().requires_grad_()
 ini=torch.zeros(8,8,32,device='cuda')
 def run(fn,y,s):return fn(*y[:4],y[4]*2*s.sigmoid(),(y[5],y[6]),chunk_size=128,evict='learned',scores=s,initial_scores=ini)[0]
 o=run(block_komp_fast,a,sa);ref=run(block_komp_flex,z,sz)
 weight=torch.randn_like(o)
 ga=torch.autograd.grad(o,(*a,sa),weight);gz=torch.autograd.grad(ref,(*z,sz),weight.double())
 for name,u,v in zip(('out','dq','dk','dv','dg','dbeta','dk0','dv0','dscore'),(o,*ga),(ref,*gz)):
  err=rms(u,v);assert torch.isfinite(u).all() and err<(.005 if dtype==torch.float32 else .01),(dtype,name,err)
  rows.append(dict(dtype=str(dtype),name=name,rms_relative=err))
 assert sa is not None and ga[-1].abs().max()>0
 print('PASS serial scan + learned gate',dtype,flush=True)
Path('../evidence/integration.json').write_text(json.dumps(dict(status='passed',checks=rows),indent=2))
