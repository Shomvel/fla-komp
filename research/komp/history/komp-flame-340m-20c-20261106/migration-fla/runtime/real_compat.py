"""Decisive old/new comparison at actual B32/T2048 eight-rank global256 shape."""
import gc,hashlib,json,os,sys,time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models import ROOT,LM,legacy,map_weights,map_optimizer,mappings,optimizer
from data import Stream

def row(name,a,b):
 a,b=a.float(),b.float();delta=a-b;rms=float(delta.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-7));mx=float(delta.abs().max()/b.abs().max().clamp_min(1));absolute=float(delta.abs().max())
 return dict(name=name,rms=rms,normalized_max=mx,absolute=absolute,passed=(rms<.035 and mx<.04) or absolute<=1e-7)
def digest(model):
 h=hashlib.sha256()
 for n,p in model.state_dict().items():h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def main():
 kind=sys.argv[1];folder=Path(sys.argv[2]);rank=int(os.environ['RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2);dist.init_process_group('nccl');assert dist.get_world_size()==8
 state=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);stream=Stream(json.loads((ROOT/'configs/data-manifest.json').read_text()),rank)
 loader=torch.load(folder/f'rank{rank}.pt',map_location='cpu',weights_only=False);stream.load_state_dict(loader['stream']);ids=torch.stack([stream.next() for _ in range(32)]).cuda();results=[]
 for label,valid in [('full',65536),('final_tail',63616)]:
  old=legacy.LM(kind).cuda().train();old.load_state_dict(state['model']);oo=legacy.optimizer(old);oo.load_state_dict(state['optimizer']);wrapped=DDP(old,device_ids=[rank],broadcast_buffers=False,gradient_as_bucket_view=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):ol=wrapped(ids[:,:-1],ids[:,1:],valid)/valid
  ol.backward();old_norm=torch.nn.utils.clip_grad_norm_(old.parameters(),1.,error_if_nonfinite=True)
  gold={n:p.grad.detach().cpu().clone() for n,p in old.named_parameters()} if rank==0 else None
  oo.step();old_delta={n:p.detach().cpu()-state['model'][n] for n,p in old.named_parameters()} if rank==0 else None
  old_loss=ol.detach().double();dist.all_reduce(old_loss);old_loss/=8
  del wrapped,oo,ol;old.zero_grad(set_to_none=True);old.cpu();gc.collect();torch.cuda.empty_cache()
  new=LM(kind).cuda().train();new.load_state_dict(map_weights(state['model']));no=optimizer(new);no.load_state_dict(map_optimizer(old,state['optimizer'],new,no))
  # Independent reverse mapping of both Adam moments and step against old parameter identities.
  check_old=legacy.optimizer(old);check_old.load_state_dict(state['optimizer']);named=dict(new.named_parameters())
  for name,p in old.named_parameters():
   for key,value in check_old.state[p].items():
    pieces=[no.state[named[n]][key].detach().cpu() for n,_ in mappings(name)]
    if value.ndim==0:assert all(torch.equal(z,value.cpu()) for z in pieces)
    else:assert torch.equal(pieces[0] if len(pieces)==1 else torch.cat(pieces),value.cpu()),(name,key)
  del check_old,old;gc.collect()
  wrapped=DDP(new,device_ids=[rank],broadcast_buffers=False,gradient_as_bucket_view=True);torch.cuda.reset_peak_memory_stats()
  with torch.autocast('cuda',dtype=torch.bfloat16):nl=wrapped(ids[:,:-1],ids[:,1:],valid)/valid
  nl.backward();new_norm=torch.nn.utils.clip_grad_norm_(new.parameters(),1.,error_if_nonfinite=True);assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters())
  checks=[]
  if rank==0:
   for name,reference in gold.items():
    pieces=[named[n].grad.detach().cpu() for n,_ in mappings(name)];actual=pieces[0] if len(pieces)==1 else torch.cat(pieces);checks.append(row('grad/'+name,actual,reference))
  no.step()
  if rank==0:
   for name,reference in old_delta.items():
    pieces=[named[n].detach().cpu() for n,_ in mappings(name)];actual=(pieces[0] if len(pieces)==1 else torch.cat(pieces))-state['model'][name];checks.append(row('delta/'+name,actual,reference))
  hashes=[None]*8;dist.all_gather_object(hashes,digest(new));assert len(set(hashes))==1
  new_loss=nl.detach().double();dist.all_reduce(new_loss);new_loss/=8
  loss_difference=float(abs(new_loss-old_loss));peak=torch.tensor(torch.cuda.max_memory_allocated(),device='cuda');dist.all_reduce(peak,op=dist.ReduceOp.MAX)
  if rank==0:
   failed=[r for r in checks if not r['passed']];result=dict(case=label,local_valid_tokens=valid,global_valid_tokens=8*valid,old_loss=float(old_loss),new_loss=float(new_loss),loss_absolute_difference=loss_difference,old_grad_norm=float(old_norm),new_grad_norm=float(new_norm),peak_bytes=int(peak),failures=len(failed),worst=sorted(checks,key=lambda r:r['rms'],reverse=True)[:10],checks=checks,replicas_identical=True);results.append(result)
   (ROOT/f'evidence/real-compat-{kind}.json').write_text(json.dumps(results,indent=2));print(kind,label,'failures',len(failed),'lossdiff',loss_difference,'peak',int(peak),flush=True)
  del wrapped,new,no,nl,named,gold,old_delta;gc.collect();torch.cuda.empty_cache();dist.barrier()
 if rank==0:
  accepted=all(r['failures']==0 and r['loss_absolute_difference']<.005 for r in results)
  receipt=dict(status='passed' if accepted else 'failed',kind=kind,shape='B32/T2048/8ranks/global256',cases=results,all_adam_slots_independently_verified=True,memory_receipt_sha=hashlib.sha256((ROOT/f'evidence/memory-{kind}.json').read_bytes()).hexdigest(),scope='Actual full and final-tail training shapes; T257 diagnostic failures retained, not all-shape equivalence')
  (ROOT/f'evidence/compat-{kind}.json').write_text(json.dumps(receipt,indent=2));flag=torch.tensor(int(accepted),device='cuda')
 else:flag=torch.tensor(0,device='cuda')
 dist.broadcast(flag,0);assert bool(flag),'real training shape compatibility failed; full distributions saved'
 dist.destroy_process_group()
if __name__=='__main__':main()
