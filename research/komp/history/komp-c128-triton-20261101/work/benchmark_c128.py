"""Matched C128 flex/Triton timings; C64 is a different algorithmic configuration."""
import json,time,statistics
from pathlib import Path
import torch,triton
from komp_slot.block_fast import block_komp_fast
from komp_slot.block_flex import block_komp_flex
from validate_block_komp import inputs
R=Path(__file__).parent.parent
results=[]
torch.set_num_threads(4)
for shape in [(1,512,8,64,64,32),(8,512,8,64,64,32),(1,8192,8,64,64,32)]:
 for evict in ('fifo','learned'):
  for name,fn,c in [('triton_c64',block_komp_fast,64),('triton_c128',block_komp_fast,128),('flex_c128',block_komp_flex,128)]:
   x=[a.requires_grad_() for a in inputs(*shape,torch.bfloat16,193,slot=True)]
   # Flex quality experiments use FP32 internal arithmetic.
   if name.startswith('flex'):x=[a.detach().float().requires_grad_() for a in x]
   kw={} if evict=='fifo' else dict(evict='learned',scores=torch.randn(*shape[:3],device='cuda'),initial_scores=torch.randn(shape[0],shape[2],shape[-1],device='cuda'))
   def fwd():return fn(*x[:5],(x[5],x[6]),chunk_size=c,**kw)
   def train():
    o,(k,v)=fwd()
    torch.autograd.grad((o,k,v),x,(torch.ones_like(o),torch.ones_like(k),torch.ones_like(v)))
   train();torch.cuda.synchronize()
   base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();train();torch.cuda.synchronize()
   peak=torch.cuda.max_memory_allocated()-base
   forward_samples=[triton.testing.do_bench(fwd,warmup=50,rep=300) for _ in range(3)]
   train_samples=[triton.testing.do_bench(train,warmup=50,rep=300) for _ in range(3)]
   fms=statistics.median(forward_samples);tms=statistics.median(train_samples)
   row=dict(shape=shape,evict=evict,implementation=name,dtype=str(x[0].dtype),forward_ms=fms,forward_backward_ms=tms,forward_samples_ms=forward_samples,forward_backward_samples_ms=train_samples,incremental_peak_bytes=peak)
   results.append(row);print(json.dumps(row),flush=True)
   (R/'evidence/benchmark-stable.json').write_text(json.dumps(results,indent=2))
   del x
