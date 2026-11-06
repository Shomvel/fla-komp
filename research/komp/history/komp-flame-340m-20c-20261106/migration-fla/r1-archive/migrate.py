"""Explicit checkpoint migration: validate new shape, discard trial updates, restore every state."""
import argparse,contextlib,hashlib,json,os,time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models import ROOT,LM,optimizer,legacy,map_weights,map_optimizer
from data import Stream
from train import fingerprint,save,rng_state,restore_rng,atomic_json
BASE=ROOT.parents[1]
PREVIOUS=BASE/'migration-mb32/runtime'
def digest(model):
 h=hashlib.sha256()
 for n,p in model.state_dict().items():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--kind',required=True,choices=['exact','block']);a=ap.parse_args()
 rank=int(os.environ['RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2);dist.init_process_group('nccl');assert dist.get_world_size()==8
 source_sha=fingerprint();data_sha=hashlib.sha256((ROOT/'configs/data-manifest.json').read_bytes()).hexdigest();batching_sha=hashlib.sha256((ROOT/'configs/batching.json').read_bytes()).hexdigest()
 stop=json.loads((BASE/'migration-fla'/f'stop-{a.kind}.json').read_text())
 folder=Path(stop['latest']['path'])
 complete=json.loads((folder/'complete.json').read_text());assert complete['source_sha']==hashlib.sha256((PREVIOUS/'source-manifest.json').read_bytes()).hexdigest()
 assert complete['data_sha']==data_sha
 provenance=json.loads((ROOT/f'evidence/checkpoint-{a.kind}-hashes.json').read_text())
 for name,sha in provenance['files'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha,name
 manifest=json.loads((ROOT/'configs/data-manifest.json').read_text());stream=Stream(manifest,rank);model=LM(a.kind).cuda();opt=optimizer(model)
 def restore():
  s=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);model.load_state_dict(map_weights(s['model']))
  old=legacy.LM(a.kind);opt.load_state_dict(map_optimizer(old,s['optimizer'],model,opt));del old,s
  r=torch.load(folder/f'rank{rank}.pt',map_location='cpu',weights_only=False);stream.load_state_dict(r['stream']);restore_rng(r['rng'])
 restore();original_hash=digest(model);original_stream=stream.state_dict();expected=stream.next();stream.load_state_dict(original_stream)
 rng=rng_state();random_probe=torch.rand(16,device='cuda');restore_rng(rng)
 wrapped=DDP(model,device_ids=[rank],broadcast_buffers=False,gradient_as_bucket_view=True)
 begin=time.perf_counter();opt.zero_grad(set_to_none=True);loss_sum=torch.zeros((),device='cuda',dtype=torch.float64);torch.cuda.reset_peak_memory_stats()
 # Exactly global256 sequences, one local B32 microbatch. Trial never counts toward formal tokens.
 for micro in range(1):
  ids=torch.stack([stream.next() for _ in range(32)]).cuda()
  with contextlib.nullcontext(),torch.autocast('cuda',dtype=torch.bfloat16):
   raw=wrapped(ids[:,:-1],ids[:,1:]);loss_sum+=raw.detach().double();(raw/(32*2048)).backward()
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
 opt.step();assert digest(model)!=original_hash
 dist.all_reduce(loss_sum);torch.cuda.synchronize();seconds=time.perf_counter()-begin
 peak=torch.tensor(torch.cuda.max_memory_allocated(),device='cuda',dtype=torch.int64);dist.all_reduce(peak,op=dist.ReduceOp.MAX);assert int(peak)<65*1024**3
 # Drop trial gradients and reload original full checkpoint; verify model, Adam, RNG, data cursor.
 opt.zero_grad(set_to_none=True);restore();assert digest(model)==original_hash
 assert torch.equal(stream.next(),expected);stream.load_state_dict(original_stream)
 assert torch.equal(torch.rand(16,device='cuda'),random_probe);restore_rng(rng)
 old=legacy.LM(a.kind);old_opt=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)['optimizer'];expected_opt=map_optimizer(old,old_opt,model,opt);del old,old_opt
 for key,state in opt.state_dict()['state'].items():
  for name,value in state.items():
   if torch.is_tensor(value):assert torch.equal(value.cpu(),expected_opt['state'][key][name]),(key,name)
 del expected_opt
 out=ROOT/'runs'/a.kind;out.mkdir(parents=True,exist_ok=True)
 save(out,model,opt,stream,complete['step'],complete['tokens'],rank,source_sha,data_sha)
 if rank==0:
  receipt=dict(status='passed',source_sha=source_sha,data_sha=data_sha,batching_sha=batching_sha,previous_source_sha=complete['source_sha'],previous_batching_sha=hashlib.sha256((PREVIOUS/'configs/batching.json').read_bytes()).hexdigest(),source_checkpoint=str(folder),source_checkpoint_hashes=provenance,previous_final_logged_step=stop["final_logged"]["step"],rollback_steps=stop["rollback_steps"],rollback_tokens=stop["rollback_tokens"],rollback_authorization="No complete-step rollback: graceful pause checkpoint",model_mapping="native FLA parameter renaming plus split up weight and Adam moments",compatibility_receipt_sha=hashlib.sha256((ROOT/f"evidence/compat-{a.kind}.json").read_bytes()).hexdigest(),cutover_step=complete['step'],cutover_tokens=complete['tokens'],initial_model_hash=original_hash,microbatch=32,accumulation=1,global_batch=256,trial_loss=float(loss_sum/(256*2048)),trial_seconds=seconds,peak_bytes=int(peak),grad_norm=float(norm),trial_tokens_counted=0,model_adam_rng_loader_restored=True,shape_change_bitwise_equivalence_claimed=False)
  atomic_json(ROOT/f'evidence/migration-{a.kind}.json',receipt);atomic_json(ROOT/f'evidence/release-{a.kind}.json',receipt);print(json.dumps(receipt),flush=True)
 dist.barrier();stream.close();dist.destroy_process_group()
if __name__=='__main__':main()
