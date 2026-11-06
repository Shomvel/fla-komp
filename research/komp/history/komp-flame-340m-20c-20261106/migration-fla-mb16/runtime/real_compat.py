"""Actual mb16 x 2, full and tail; explicit trajectory-compatibility evidence."""
import contextlib,gc,hashlib,json,os,sys
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models import ROOT,LM,legacy,map_weights,map_optimizer,mappings,optimizer
from data import Stream

def row(name,a,b):
 a,b=a.float(),b.float();delta=a-b;an=float(b.double().square().sum());dn=float(delta.double().square().sum());rms=float(delta.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-7));mx=float(delta.abs().max()/b.abs().max().clamp_min(1));absolute=float(delta.abs().max())
 return dict(name=name,rms=rms,normalized_max=mx,absolute=absolute,reference_norm=an**.5,error_norm=dn**.5,reference_squared_norm=an,error_squared_norm=dn,passed=(rms<.035 and mx<.04) or absolute<=1e-7)
def digest(model):
 h=hashlib.sha256()
 for n,p in model.state_dict().items():h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def accumulated(wrapped,ids,valid,microbatch):
 loss_sum=torch.zeros((),device='cuda',dtype=torch.float64)
 for start in range(0,32,microbatch):
  part=ids[start:start+microbatch];count=min(microbatch*2048,valid-start*2048)
  with wrapped.no_sync() if start+microbatch<32 else contextlib.nullcontext():
   with torch.autocast('cuda',dtype=torch.bfloat16):raw=wrapped(part[:,:-1],part[:,1:],count);loss=raw/valid
   loss.backward();loss_sum+=raw.detach().double()/valid
 return loss_sum

def main():
 kind=sys.argv[1];folder=Path(sys.argv[2]);native=kind=='exact';rank=int(os.environ['RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2);dist.init_process_group('nccl');assert dist.get_world_size()==8
 state=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);stream=Stream(json.loads((ROOT/'configs/data-manifest.json').read_text()),rank)
 policy=json.loads((ROOT/'configs/native-acceptance.json').read_text())
 loader=torch.load(folder/f'rank{rank}.pt',map_location='cpu',weights_only=False);stream.load_state_dict(loader['stream']);ids=torch.stack([stream.next() for _ in range(32)]).cuda();results=[]
 causal_model=LM(kind).cuda().eval();causal_model.load_state_dict(state['model'] if native else map_weights(state['model']))
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  probe=ids[:1,:-1];changed=probe.clone();changed[:,1024:]=(changed[:,1024:]+37)%32000
  original=causal_model(probe)[:,:1024];altered=causal_model(changed)[:,:1024];causal_delta=(original-altered).float();causal_rms=float(causal_delta.square().mean().sqrt()/original.float().square().mean().sqrt().clamp_min(1e-7));assert torch.isfinite(altered).all() and causal_rms<1e-5
 del causal_model,original,altered,causal_delta;gc.collect();torch.cuda.empty_cache()
 for label,valid in [('full',65536),('final_tail',63616)]:
  old=(LM(kind) if native else legacy.LM(kind)).cuda().train();old.load_state_dict(state['model']);oo=optimizer(old) if native else legacy.optimizer(old);oo.load_state_dict(state['optimizer']);wrapped=DDP(old,device_ids=[rank],broadcast_buffers=False,gradient_as_bucket_view=True)
  ol=accumulated(wrapped,ids,valid,32 if native else 16);old_norm=torch.nn.utils.clip_grad_norm_(old.parameters(),1.,error_if_nonfinite=True)
  gold={n:p.grad.detach().cpu().clone() for n,p in old.named_parameters()} if rank==0 else None
  oo.step();old_delta={n:p.detach().cpu()-state['model'][n] for n,p in old.named_parameters()} if rank==0 else None
  old_loss=ol.detach().double();dist.all_reduce(old_loss);old_loss/=8
  del wrapped,oo,ol;old.zero_grad(set_to_none=True);old.cpu();gc.collect();torch.cuda.empty_cache()
  new=LM(kind).cuda().train();new.load_state_dict(state['model'] if native else map_weights(state['model']));no=optimizer(new);no.load_state_dict(state['optimizer'] if native else map_optimizer(old,state['optimizer'],new,no))
  check_old=optimizer(old) if native else legacy.optimizer(old);check_old.load_state_dict(state['optimizer']);named=dict(new.named_parameters())
  def targets(name):return [(name,None)] if native else mappings(name)
  for name,p in old.named_parameters():
   for key,value in check_old.state[p].items():
    pieces=[no.state[named[n]][key].detach().cpu() for n,_ in targets(name)]
    if value.ndim==0:assert all(torch.equal(z,value.cpu()) for z in pieces)
    else:assert torch.equal(pieces[0] if len(pieces)==1 else torch.cat(pieces),value.cpu()),(name,key)
  del check_old,old;gc.collect()
  wrapped=DDP(new,device_ids=[rank],broadcast_buffers=False,gradient_as_bucket_view=True);torch.cuda.reset_peak_memory_stats()
  nl=accumulated(wrapped,ids,valid,16);new_norm=torch.nn.utils.clip_grad_norm_(new.parameters(),1.,error_if_nonfinite=True);assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters())
  checks=[]
  if rank==0:
   for name,reference in gold.items():
    pieces=[named[n].grad.detach().cpu() for n,_ in targets(name)];actual=pieces[0] if len(pieces)==1 else torch.cat(pieces);checks.append(row('grad/'+name,actual,reference))
  no.step();assert all(torch.isfinite(p).all() for p in new.parameters())
  if rank==0:
   for name,reference in old_delta.items():
    pieces=[named[n].detach().cpu() for n,_ in targets(name)];actual=(pieces[0] if len(pieces)==1 else torch.cat(pieces))-state['model'][name];checks.append(row('delta/'+name,actual,reference))
  hashes=[None]*8;dist.all_gather_object(hashes,digest(new));assert len(set(hashes))==1
  new_loss=nl.detach().double();dist.all_reduce(new_loss);new_loss/=8
  loss_difference=float(abs(new_loss-old_loss));peak=torch.tensor(torch.cuda.max_memory_allocated(),device='cuda');dist.all_reduce(peak,op=dist.ReduceOp.MAX)
  if rank==0:
   failed=[r for r in checks if not r['passed']];overall={}
   for prefix in ['grad/','delta/']:
    rows=[r for r in checks if r['name'].startswith(prefix)];overall[prefix.rstrip('/')]=(sum(r['error_squared_norm'] for r in rows)/max(sum(r['reference_squared_norm'] for r in rows),1e-30))**.5
   result=dict(case=label,microbatch=16,accumulation=2,local_micro_valid=[32768,valid-32768],local_valid_tokens=valid,global_valid_tokens=8*valid,old_loss=float(old_loss),new_loss=float(new_loss),loss_absolute_difference=loss_difference,old_grad_norm=float(old_norm),new_grad_norm=float(new_norm),peak_bytes=int(peak),failures=len(failed),whole_model_relative_l2=overall,worst=sorted(checks,key=lambda r:r['rms'],reverse=True)[:10],checks=checks,replicas_identical=True,all_parameters_and_gradients_finite=True,causal_prefix_relative_rms=causal_rms);results.append(result)
   (ROOT/f'evidence/real-compat-{kind}.json').write_text(json.dumps(results,indent=2));print(kind,label,'failures',len(failed),'lossdiff',loss_difference,'global',overall,'peak',int(peak),flush=True)
  del wrapped,new,no,nl,named,gold,old_delta;gc.collect();torch.cuda.empty_cache();dist.barrier()
 if rank==0:
  accepted=all(r['failures']==0 and r['loss_absolute_difference']<.005 for r in results)
  native_accepted=all(r['loss_absolute_difference']<.005 and r['whole_model_relative_l2']['grad']<.035 and r['whole_model_relative_l2']['delta']<.035 and r['all_parameters_and_gradients_finite'] and r['replicas_identical'] and r['causal_prefix_relative_rms']<1e-5 for r in results)
  receipt=dict(status='passed' if accepted else 'failed',strict_trajectory_compatibility='passed' if accepted else 'failed',native_forward_backward_acceptance='passed' if native_accepted else 'failed',native_training_acceptance='pending_full_state_restore',acceptance_policy=policy,acceptance_policy_sha=hashlib.sha256((ROOT/'configs/native-acceptance.json').read_bytes()).hexdigest(),kind=kind,source_sha=hashlib.sha256((ROOT/'source-manifest.json').read_bytes()).hexdigest(),batching_sha=hashlib.sha256((ROOT/'configs/batching.json').read_bytes()).hexdigest(),shape='B16x2/T2048/8ranks/global256',baseline='native FLA B32x1' if native else 'legacy B16x2',cases=results,all_adam_slots_independently_verified=True,memory_receipt_sha=hashlib.sha256((ROOT/f'evidence/memory-{kind}.json').read_bytes()).hexdigest(),scope='Actual full and final-tail training shapes; prior T257 and B32 failures preserved')
  (ROOT/f'evidence/compat-{kind}.json').write_text(json.dumps(receipt,indent=2));flag=torch.tensor(int(native_accepted),device='cuda')
 else:flag=torch.tensor(0,device='cuda')
 dist.broadcast(flag,0);assert bool(flag),'native training acceptance failed; strict compatibility independently preserved'
 dist.destroy_process_group()
if __name__=='__main__':main()
