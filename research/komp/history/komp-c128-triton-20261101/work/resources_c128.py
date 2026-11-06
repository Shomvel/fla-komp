import json
from pathlib import Path
import torch,triton
from triton.runtime.jit import JITFunction
from validate_block_komp import inputs
from komp_slot.block_fast import block_komp_fast
rows=[]
original=JITFunction.run
def compile_only(self,*args,**kwargs):
 kwargs['warmup']=True
 k=original(self,*args,**kwargs)
 row=dict(name=self.__name__,shared=k.metadata.shared,warps=k.metadata.num_warps,stages=k.metadata.num_stages)
 rows.append(row);print(json.dumps(row),flush=True)
 return k
JITFunction.run=compile_only
for slot in (False,True):
 for dtype in (torch.float32,torch.bfloat16):
  for learned in (False,True):
   x=[v.requires_grad_() for v in inputs(1,257,2,64,64,32,dtype,43,slot=slot)]
   kw=dict(evict='learned',scores=torch.zeros(1,257,2,device='cuda'),initial_scores=torch.zeros(1,2,32,device='cuda')) if learned else {}
   o,(k,v)=block_komp_fast(*x[:5],(x[5],x[6]),chunk_size=128,**kw)
   torch.autograd.grad((o,k,v),x,(torch.ones_like(o),torch.ones_like(k),torch.ones_like(v)))
Path('../evidence/resources-fixed.json').write_text(json.dumps(rows,indent=2))
assert all(r['shared'] <= 232448 for r in rows), [r for r in rows if r['shared'] > 232448]
