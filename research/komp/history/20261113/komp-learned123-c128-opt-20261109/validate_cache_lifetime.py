"""Check checkpoint-cache lifetime while the completed loss is still referenced."""
import gc,json,sys
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'snapshot-final'))
from fla.ops.komp.geometry_cache import cached_geometry,geometry_contexts
x=torch.ones(16,device='cuda',requires_grad=True)
def fn(z):
 a=cached_geometry(('lifetime',),lambda:torch.ones(4*1024*1024,device='cuda'))
 return z*a.sum()
gc.collect();torch.cuda.synchronize();base=torch.cuda.memory_allocated()
loss=checkpoint(fn,x,use_reentrant=False,context_fn=geometry_contexts).sum()
forward=torch.cuda.memory_allocated()
loss.backward();torch.cuda.synchronize();held=torch.cuda.memory_allocated()
del loss;gc.collect();torch.cuda.synchronize();released=torch.cuda.memory_allocated()
assert released-base<1024*1024,(base,released)
out=dict(passed=True,base=base,forward=forward,after_backward_loss_held=held,after_loss_released=released,
 production_rule='delete raw/loss immediately after backward to bound graph-local cache lifetime')
(ROOT/'evidence/cache-loss-lifetime.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
