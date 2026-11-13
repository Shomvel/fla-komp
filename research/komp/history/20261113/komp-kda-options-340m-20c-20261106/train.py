"""Flame-recipe paired DDP training with token-exact budget and recoverable input state."""
import argparse,contextlib,hashlib,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models import ROOT,LM,optimizer,metadata
from data import Stream,evaluate
TARGET=6_800_000_000;GLOBAL=256*2048;SEQ=2048;WORLD=8

def atomic_json(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def rng_state():return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate(),numpy=np.random.get_state())
def restore_rng(s):torch.set_rng_state(s['torch']);torch.cuda.set_rng_state(s['cuda']);random.setstate(s['python']);np.random.set_state(s['numpy'])
def fingerprint():
 manifest=json.loads((ROOT/'source-manifest.json').read_text())
 for name,sha in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
 return hashlib.sha256((ROOT/'source-manifest.json').read_bytes()).hexdigest()
def save(out,model,opt,stream,step,tokens,rank,source_sha,data_sha):
 folder=out/f'checkpoint-{step:06d}';folder.mkdir(exist_ok=True)
 tmp=folder/f'rank{rank}.pt.tmp';torch.save(dict(stream=stream.state_dict(),rng=rng_state()),tmp);tmp.replace(folder/f'rank{rank}.pt')
 if rank==0:
  tmp=folder/'model.pt.tmp';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,tokens=tokens,source_sha=source_sha,data_sha=data_sha),tmp);tmp.replace(folder/'model.pt')
 dist.barrier()
 if rank==0:
  atomic_json(folder/'complete.json',dict(step=step,tokens=tokens,world_size=WORLD,source_sha=source_sha,data_sha=data_sha))
  atomic_json(out/'latest.json',dict(path=str(folder),step=step,tokens=tokens))
  # Keep two completed resumable checkpoints plus final metrics; only this run's files.
  old=sorted(p for p in out.glob('checkpoint-*') if (p/'complete.json').exists())[:-2]
  import shutil
  for p in old:shutil.rmtree(p)
 dist.barrier()
def resume(out,model,opt,stream,rank,source_sha,data_sha):
 if not (out/'latest.json').exists():return 0,0
 folder=Path(json.loads((out/'latest.json').read_text())['path']);assert (folder/'complete.json').exists()
 state=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
 assert state['source_sha']==source_sha and state['data_sha']==data_sha
 model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);step,tokens=state['step'],state['tokens'];del state
 state=torch.load(folder/f'rank{rank}.pt',map_location='cpu',weights_only=False);stream.load_state_dict(state['stream']);restore_rng(state['rng'])
 return step,tokens

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['exact','block'],required=True);ap.add_argument('--release-check',action='store_true');args=ap.parse_args();random.seed(42);np.random.seed(42);torch.manual_seed(42)
 rank=int(os.environ['RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2);dist.init_process_group('nccl');assert dist.get_world_size()==WORLD
 batching_path=ROOT/'configs/batching.json';batching_sha=hashlib.sha256(batching_path.read_bytes()).hexdigest();microbatch=json.loads(batching_path.read_text())['microbatch'];assert microbatch in (1,2,4,16,32) and 32%microbatch==0
 source_sha=fingerprint();data_path=ROOT/'configs/data-manifest.json';data_sha=hashlib.sha256(data_path.read_bytes()).hexdigest();manifest=json.loads(data_path.read_text())
 audit_path=ROOT/'evidence/data-audit.json';assert hashlib.sha256(audit_path.read_bytes()).hexdigest()==manifest['data_audit_sha256']
 audit=json.loads(audit_path.read_text())
 for name,sha in audit['tokenizer_files'].items():assert hashlib.sha256((Path(manifest['tokenizer_path'])/name).read_bytes()).hexdigest()==sha,name
 assert hashlib.sha256(Path(manifest['excluded_text_sha256_path']).read_bytes()).hexdigest()==audit['exclusion_sha256']
 for split in ('validation','test'):
  assert hashlib.sha256(Path(manifest[split+'_tokens_path']).read_bytes()).hexdigest()==audit['token_files'][split]['sha256']
 acceptance=json.loads((ROOT/f'evidence/preflight-{args.kind}.json').read_text())
 assert acceptance['status']=='passed' and acceptance['source_sha']==source_sha and acceptance['data_sha']==data_sha and acceptance['batching_sha']==batching_sha and acceptance['kind']==args.kind
 stream=Stream(manifest,rank);model=LM(args.kind).cuda();opt=optimizer(model)
 out=ROOT/('release-check' if args.release_check else 'runs')/args.kind;out.mkdir(parents=True,exist_ok=True)
 step,tokens=resume(out,model,opt,stream,rank,source_sha,data_sha)
 if step==0:
  h=hashlib.sha256()
  for name,param in model.state_dict().items():h.update(name.encode());h.update(param.detach().cpu().contiguous().numpy().tobytes())
  assert h.hexdigest()==acceptance['initial_model_sha'],'Formal initialization differs from preflight'
 wrapped=DDP(model,device_ids=[rank],broadcast_buffers=False,gradient_as_bucket_view=True)
 if rank==0:atomic_json(out/'config.json',dict(model=metadata(model),source_sha=source_sha,data_sha=data_sha,target_input_tokens=TARGET,target_supervised_tokens=TARGET,global_sequences=256,sequence_length=SEQ,microbatch=microbatch,accumulation=32//microbatch,batching_sha=batching_sha,world_size=WORLD,optimizer=dict(lr=3e-4,betas=[.8,.95],eps=1e-15,weight_decay=.1),warmup_steps=1024,cosine_min_ratio=.1,packing='2049 IDs, 2048 inputs and next-token targets; shared boundary token',final_padding='unbudgeted trailing compute positions masked from loss',checkpoint_retention='latest two complete checkpoints in this new run only'))
 if args.release_check:
  # Loader restore must reproduce exact real text IDs; no advancement lost.
  s=stream.state_dict();expected=stream.next();stream.load_state_dict(s);assert torch.equal(stream.next(),expected);stream.load_state_dict(s)
 limit=2*GLOBAL if args.release_check else TARGET
 while tokens<limit:
  begin=time.perf_counter();local_budget=min(GLOBAL,limit-tokens)//WORLD
  microsteps=math.ceil(local_budget/(SEQ*microbatch));opt.zero_grad(set_to_none=True);loss_sum=torch.zeros((),device='cuda',dtype=torch.float64)
  end_tokens=tokens+local_budget*WORLD
  if end_tokens<=1024*GLOBAL:lr=3e-4*end_tokens/(1024*GLOBAL)
  else:lr=3e-4*(.1+.9*.5*(1+math.cos(math.pi*min(1.,(end_tokens-1024*GLOBAL)/(TARGET-1024*GLOBAL)))))
  for group in opt.param_groups:group['lr']=lr
  for micro in range(microsteps):
   valid=min(SEQ*microbatch,local_budget-micro*SEQ*microbatch);ids=torch.stack([stream.next() for _ in range(microbatch)]).cuda(non_blocking=False)
   sync=wrapped.no_sync() if micro+1<microsteps else contextlib.nullcontext()
   with sync,torch.autocast('cuda',dtype=torch.bfloat16):
    raw=wrapped(ids[:,:-1],ids[:,1:],valid);loss=raw/local_budget
    loss_sum+=raw.detach().double();loss.backward()
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
  if args.release_check:assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
  before=model.model.layers[0].attn.q.weight.detach().clone() if args.release_check else None
  opt.step()
  if args.release_check:assert not torch.equal(before,model.model.layers[0].attn.q.weight)
  step+=1;tokens=end_tokens;dist.all_reduce(loss_sum);torch.cuda.synchronize()
  compute_tokens=(step-1)*GLOBAL+microsteps*SEQ*microbatch*WORLD
  row=dict(kind=args.kind,step=step,effective_input_tokens=tokens,input_tokens=tokens,supervised_tokens=tokens,compute_tokens=compute_tokens,loss=float(loss_sum/(local_budget*WORLD)),lr=lr,grad_norm=float(norm),seconds=time.perf_counter()-begin,peak_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),documents=stream.state['documents'],excluded_documents=stream.state['excluded_documents'])
  if rank==0:
   with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps(row),flush=True)
  if step%512==0 and not args.release_check:
   val=evaluate(model,manifest['validation_tokens_path'],rank);val.update(step=step,input_tokens=tokens)
   if rank==0:
    with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(val)+'\n')
  pause_flag=torch.tensor(int((out/'PAUSE').exists()),device='cuda')
  dist.all_reduce(pause_flag,op=dist.ReduceOp.MAX);pause_requested=bool(pause_flag.item())
  if step%256==0 or tokens==limit or args.release_check or pause_requested:save(out,model,opt,stream,step,tokens,rank,source_sha,data_sha)
  if pause_requested:
   if rank==0:atomic_json(out/'paused.json',dict(status='paused',step=step,tokens=tokens))
   stream.close();dist.destroy_process_group();return
  if args.release_check and step==1:
   expected=stream.next();step,tokens=resume(out,model,opt,stream,rank,source_sha,data_sha);assert torch.equal(stream.next(),expected)
   # Rewind after witness so resumed training consumes exactly the saved next sequence.
   step,tokens=resume(out,model,opt,stream,rank,source_sha,data_sha)
 if args.release_check:
  hashes=[None]*WORLD
  h=hashlib.sha256()
  for name,p in model.state_dict().items():h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
  dist.all_gather_object(hashes,h.hexdigest());assert len(set(hashes))==1
  if rank==0:atomic_json(ROOT/f'evidence/release-{args.kind}.json',dict(status='passed',source_sha=source_sha,data_sha=data_sha,batching_sha=batching_sha,microbatch=microbatch,real_data=True,global_batch=256,steps=step,ddp_final_hash=hashes[0],loader_checkpoint_restore=True))
 else:
  val=evaluate(model,manifest['validation_tokens_path'],rank);test=evaluate(model,manifest['test_tokens_path'],rank)
  if rank==0:atomic_json(out/'complete.json',dict(status='completed',step=step,effective_input_tokens=tokens,input_tokens=tokens,supervised_tokens=tokens,compute_tokens=compute_tokens,validation=val,test=test,source_sha=source_sha,data_sha=data_sha))
 stream.close();dist.destroy_process_group()
if __name__=='__main__':main()
