"""Same-local-shape DDP vs single-rank accumulated gradient scaling witness."""
import contextlib,json,os
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models import ROOT,Mixer,D

class Witness(nn.Module):
 def __init__(self,kind):super().__init__();torch.manual_seed(99);self.mix=Mixer(kind)
 def forward(self,x,valid):
  y=self.mix(x);return y[:,:valid].float().square().sum()/D

def inputs(rank,micro):
 generator=torch.Generator(device='cuda').manual_seed(8200+rank*7+micro)
 return torch.randn(1,2048,D,generator=generator,device='cuda')*.1

def main():
 rank=int(os.environ['RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2);dist.init_process_group('nccl');assert dist.get_world_size()==8
 rows=[]
 for kind in ('exact','block'):
  model=Witness(kind).cuda();wrapped=DDP(model,device_ids=[rank],broadcast_buffers=False)
  for label,counts in [('full',[2048,2048]),('tail',[2048,128])]:
   model.zero_grad(set_to_none=True);den=sum(counts)
   for micro,valid in enumerate(counts):
    with (wrapped.no_sync() if micro==0 else contextlib.nullcontext()),torch.autocast('cuda',dtype=torch.bfloat16):
     (wrapped(inputs(rank,micro),valid)/den).backward()
   if rank==0:
    actual={n:p.grad.detach().clone() for n,p in model.named_parameters()}
    reference=Witness(kind).cuda();reference.load_state_dict(model.state_dict())
    # Separate module has no reducer hooks; identical local shapes and parameters.
    for rr in range(8):
     for micro,valid in enumerate(counts):
      with torch.autocast('cuda',dtype=torch.bfloat16):(reference(inputs(rr,micro),valid)/(8*den)).backward()
    worst_rms=worst_max=0.
    for name,p in reference.named_parameters():
     a,b=actual[name].double(),p.grad.double();delta=a-b
     rms=float(delta.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-8));mx=float(delta.abs().max()/b.abs().max().clamp_min(1.))
     assert torch.isfinite(a).all() and rms<.01 and mx<.01,(kind,label,name,rms,mx)
     worst_rms=max(worst_rms,rms);worst_max=max(worst_max,mx)
    del reference,actual
    rows.append(dict(kind=kind,case=label,local_valid_tokens=den,global_valid_tokens=8*den,worst_gradient_relative_rms=worst_rms,worst_gradient_normalized_max=worst_max,tolerance=.01))
   dist.barrier()
  del wrapped,model;torch.cuda.empty_cache()
 if rank==0:(ROOT/'evidence/ddp-scaling.json').write_text(json.dumps(dict(status='passed',scope='KOMP mixer, BF16, same local shape B1/T2048; full/tail masked-token normalization vs single-GPU accumulated replay; not full-model optimizer equivalence',checks=rows),indent=2))
 dist.destroy_process_group()
if __name__=='__main__':main()
