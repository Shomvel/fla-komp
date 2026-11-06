"""Bounded real-data microbatch selection at unchanged effective global batch."""
import json,os,time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models import ROOT,LM,optimizer
from data import Stream

def main():
 rank=int(os.environ['RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2);dist.init_process_group('nccl')
 manifest=json.loads((ROOT/'configs/data-manifest.json').read_text());results={};choice={}
 for kind in ('exact','block'):
  stream=Stream(manifest,rank);model=LM(kind).cuda();opt=optimizer(model);wrapped=DDP(model,device_ids=[rank],broadcast_buffers=False,gradient_as_bucket_view=True);rows=[]
  for microbatch in (1,2,4):
   torch.cuda.reset_peak_memory_stats();timings=[]
   for iteration in range(2):
    ids=torch.stack([stream.next() for _ in range(microbatch)]).cuda();opt.zero_grad(set_to_none=True);dist.barrier();torch.cuda.synchronize();start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=wrapped(ids[:,:-1],ids[:,1:])/(microbatch*2048)
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
    elapsed=torch.tensor(time.perf_counter()-start,device='cuda');dist.all_reduce(elapsed,op=dist.ReduceOp.MAX);timings.append(float(elapsed))
   peak=torch.tensor(torch.cuda.max_memory_allocated(),device='cuda',dtype=torch.int64);dist.all_reduce(peak,op=dist.ReduceOp.MAX)
   row=dict(microbatch=microbatch,seconds=timings,peak_bytes=int(peak),warm_tokens_per_second=8*microbatch*2048/timings[-1],last_loss=float(loss.detach()),grad_norm=float(norm));rows.append(row)
   if rank==0:print(kind,json.dumps(row),flush=True)
  candidates=[r for r in rows if r['peak_bytes']<60*1024**3];assert candidates
  best=max(candidates,key=lambda r:r['warm_tokens_per_second']);choice[kind]=best['microbatch'];results[kind]=rows
  del wrapped,opt,model;stream.close();torch.cuda.empty_cache();dist.barrier()
 if rank==0:
  (ROOT/'configs/batching.json').write_text(json.dumps(dict(microbatch=choice,global_sequences=256,world_size=8),indent=2))
  (ROOT/'evidence/batch-benchmark.json').write_text(json.dumps(dict(status='passed',selected=choice,results=results,scope='8rank real-data full model warm fwd/bwd/optimizer; global256 training includes accumulation/data/checkpoints additionally'),indent=2))
 dist.destroy_process_group()
if __name__=='__main__':main()
