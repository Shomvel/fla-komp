import json,math
from pathlib import Path
import torch
from reference import block_komp_reference,block_komp_token_reference
ROOT=Path(__file__).resolve().parent
torch.manual_seed(42);torch.set_num_threads(2)
checks=[]
for n in (1,31,32,33,128,129):
 m=8;k=8
 q,kx,v=[torch.randn(1,n,1,k,dtype=torch.float64,requires_grad=True) for _ in range(3)]
 g=(-.001-.01*torch.rand(1,n,1,m,dtype=torch.float64)).requires_grad_();beta=torch.full((1,n,1),.2,dtype=torch.float64,requires_grad=True)
 keys=torch.randn(1,1,m,k,dtype=torch.float64,requires_grad=True);state=torch.randn(1,1,m,k,dtype=torch.float64,requires_grad=True)
 inputs=[q,kx,v,g,beta,keys,state]
 a,aa=block_komp_reference(*inputs[:5],tuple(inputs[5:]),chunk_size=128)
 b,bb=block_komp_token_reference(*inputs[:5],tuple(inputs[5:]),chunk_size=128)
 # Age order starts at slot T % M; differentiation through gather preserves state loss semantics.
 order=(torch.arange(m)+n)%m;bb=tuple(x[:,:,order] for x in bb)
 weights=[torch.randn_like(z) for z in (a,*aa)]
 ga=torch.autograd.grad((a,*aa),inputs,weights,retain_graph=True);gb=torch.autograd.grad((b,*bb),inputs,weights)
 maximum=0.
 for x,y in zip((a,*aa,*ga),(b,*bb,*gb)):
  torch.testing.assert_close(x,y,atol=1e-9,rtol=1e-9);maximum=max(maximum,float((x-y).detach().abs().max()))
 checks.append(dict(tokens=n,max_abs=maximum))
(ROOT/'evidence/reference-state-contract.json').write_text(json.dumps(dict(status='passed',fifo_block_return='age order',exact_return='physical slot order',permutation='age_to_physical = (arange(M)+T)%M',checks=checks),indent=2))
