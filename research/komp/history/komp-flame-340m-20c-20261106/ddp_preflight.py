import os,json,time,hashlib,random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models import ROOT,LM,optimizer,metadata

def digest(model):
 h=hashlib.sha256()
 for name,p in model.state_dict().items():h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 rank=int(os.environ['RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2)
 dist.init_process_group('nccl');assert dist.get_world_size()==8
 initial=[]
 for kind in ('exact','block'):
  model=LM(kind).cuda();initial.append(digest(model));opt=optimizer(model)
  wrapped=DDP(model,device_ids=[rank],broadcast_buffers=False,gradient_as_bucket_view=True)
  records=[]
  for step in range(2):
   torch.manual_seed(900+rank+step*8)
   ids=torch.randint(0,32000,(1,2049),device='cuda')
   start=time.perf_counter();opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=wrapped(ids[:,:-1],ids[:,1:])/2048
   loss.backward()
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
   assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
   before=model.layers[0].mix.q.weight.detach().clone();opt.step();assert not torch.equal(before,model.layers[0].mix.q.weight)
   torch.cuda.synchronize();records.append(dict(step=step,loss=float(loss),grad_norm=float(norm),seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated()))
   if step==0:
    # Disk roundtrip restores actual full model/Adam/RNG; byte hash checks parameters.
    path=ROOT/f'evidence/preflight-{kind}-rank{rank}.pt';expected=digest(model)
    saved_rng=torch.get_rng_state();saved_cuda=torch.cuda.get_rng_state()
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=saved_rng,cuda_rng=saved_cuda),path)
    expected_rand=torch.rand(4,device='cuda')
    with torch.no_grad():model.layers[0].mix.q.weight.add_(1.)
    state=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['rng']);torch.cuda.set_rng_state(state['cuda_rng']);assert torch.equal(torch.rand(4,device='cuda'),expected_rand)
    assert digest(model)==expected
    # Adam moments and step values must survive the serialized restore.
    for key,s in opt.state_dict()['state'].items():
     for n,v in s.items():
      if torch.is_tensor(v):assert torch.equal(v.cpu(),state['optimizer']['state'][key][n])
    del state;path.unlink()
  hashes=[None]*8;dist.all_gather_object(hashes,digest(model));assert len(set(hashes))==1,hashes
  (ROOT/f'evidence/ddp-{kind}-rank{rank}.json').write_text(json.dumps(dict(status='passed',metadata=metadata(model),initial_hash=initial[-1],final_hash=hashes[0],records=records,checkpoint_rng_restore=True),indent=2))
  del wrapped,opt,model;torch.cuda.empty_cache();dist.barrier()
 assert initial[0]==initial[1],initial
 if rank==0:(ROOT/'evidence/ddp.json').write_text(json.dumps(dict(status='passed',world_size=8,shared_initial_hash=initial[0],kinds=['exact','block']),indent=2))
 dist.destroy_process_group()
if __name__=='__main__':main()
