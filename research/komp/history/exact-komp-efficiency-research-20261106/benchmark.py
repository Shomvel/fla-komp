"""Isolated actual-shape Exact A/B. Run each variant in a fresh GPU process.
One rank is a performance/numerical screening only; eight ranks reproduce global256.
No formal trainer entry points, migration, checkpoint deletion or dataset writes.
"""
import argparse,contextlib,gc,hashlib,importlib.metadata,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
ROOT=Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def digest(model):
 h=hashlib.sha256()
 for name,p in model.state_dict().items():h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def restore_rng(rng):
 torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda']);random.setstate(rng['python']);np.random.set_state(rng['numpy'])

def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=['baseline','candidate'],required=True);p.add_argument('--cache',action='store_true');p.add_argument('--prune-gc',action='store_true');p.add_argument('--fixture',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case',choices=['full','tail'],default='full');p.add_argument('--mode',choices=['check','bench','profile'],default='check');p.add_argument('--warmup',type=int,default=3);p.add_argument('--steps',type=int,default=10);a=p.parse_args()
 import os
 os.environ['EXACT_KOMP_PRUNE_GC']='1' if a.prune_gc else '0'
 assert not a.prune_gc or a.variant=='candidate'
 assert not a.cache or a.variant=='candidate'
 rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1));assert world in (1,8)
 torch.cuda.set_device(local);torch.set_num_threads(2)
 if world>1:dist.init_process_group('nccl')
 if rank==0:a.out.mkdir(parents=True,exist_ok=False)
 if world>1:dist.barrier()
 receipt=json.loads((a.fixture/'fixture.json').read_text())
 for name in ['model.pt',f'input-rank{rank}.pt']:
  assert sha(a.fixture/name)==receipt['files'][name],name
 sys.path.insert(0,str(ROOT/a.variant));from models import LM,optimizer
 state=torch.load(a.fixture/'model.pt',map_location='cpu',weights_only=False)
 fixed=torch.load(a.fixture/f'input-rank{rank}.pt',map_location='cpu',weights_only=False)
 assert fixed['ids'].shape==(32,2049)
 def build():
  model=LM('exact').cuda().train();model.load_state_dict(state['model']);opt=optimizer(model);opt.load_state_dict(state['optimizer'])
  for layer in model.model.layers:
   layer.attn.checkpoint_mixer=True
   if a.variant=='candidate':layer.attn.cache_geometry=a.cache
  restore_rng(fixed['rng'])
  wrapped=DDP(model,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True) if world>1 else model
  return model,opt,wrapped
 model,opt,wrapped=build()
 valid=65536 if a.case=='full' else 63616
 def update(capture=False):
  opt.zero_grad(set_to_none=True);events=[];losses=[]
  torch.cuda.synchronize()
  start=time.perf_counter();io=0.
  for micro in range(2):
   t=time.perf_counter()
   with torch.profiler.record_function('input_h2d'):
    ids=fixed['ids'][micro*16:(micro+1)*16].cuda(non_blocking=False)
   io+=time.perf_counter()-t
   count=min(32768,valid-micro*32768)
   sync=wrapped.no_sync() if world>1 and micro==0 else contextlib.nullcontext()
   with sync:
    e=[torch.cuda.Event(enable_timing=True) for _ in range(3)];e[0].record()
    with torch.profiler.record_function('model_forward'),torch.autocast('cuda',dtype=torch.bfloat16):
     raw=wrapped(ids[:,:-1],ids[:,1:],count);loss=raw/valid
    e[1].record()
    with torch.profiler.record_function('model_backward'):loss.backward()
    e[2].record();events.append(e);losses.append(raw.detach().double()/valid)
   del raw,loss,ids
  torch.cuda.synchronize();fb_end=time.perf_counter()
  gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()} if capture else None
  assert all(p.grad is not None for p in model.parameters())
  if capture:assert all(torch.isfinite(g).all() for g in gradients.values())
  e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True);e0.record()
  with torch.profiler.record_function('clip_and_adam'):
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
  e1.record();torch.cuda.synchronize();end=time.perf_counter()
  loss=sum(losses)
  if world>1:
   with torch.profiler.record_function('loss_all_reduce'):
    dist.all_reduce(loss);loss/=world
  torch.cuda.synchronize();end=time.perf_counter()
  row=dict(loss=float(loss),grad_norm=float(norm),forward_ms=sum(e[0].elapsed_time(e[1]) for e in events),backward_ms_including_ddp=sum(e[1].elapsed_time(e[2]) for e in events),clip_adam_ms=e0.elapsed_time(e1),fixed_input_h2d_host_seconds=io,forward_backward_wall_seconds=fb_end-start,update_wall_seconds=end-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),effective_tokens=world*valid,capture_overhead_included=capture)
  if world>1:
   elapsed=torch.tensor(row['update_wall_seconds'],device='cuda');dist.all_reduce(elapsed,op=dist.ReduceOp.MAX);row['slowest_rank_update_seconds']=float(elapsed)
  if capture:
   assert all(torch.isfinite(q).all() for q in model.parameters())
   delta={n:p.detach().cpu()-state['model'][n] for n,p in model.named_parameters()}
   torch.save(dict(grad=gradients,delta=delta,loss=row['loss']),a.out/f'check-rank{rank}.pt')
  return row
 torch.cuda.reset_peak_memory_stats()
 rows=[]
 if a.mode=='check':
  rows.append(dict(phase='numerical',**update(capture=True)))
  hashes=[digest(model)]
  if world>1:
   dist.all_gather_object(hashes:=[None]*world,digest(model));assert len(set(hashes))==1
  # Restore from the fixture and compare model, every Adam state slot, all RNGs.
  model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore_rng(fixed['rng'])
  for name,value in model.state_dict().items():assert torch.equal(value.cpu(),state['model'][name]),name
  restored=opt.state_dict()
  for index,slots in state['optimizer']['state'].items():
   for name,value in slots.items():
    got=restored['state'][index][name]
    assert torch.equal(got.cpu(),value.cpu()) if torch.is_tensor(value) else got==value
  assert torch.equal(torch.get_rng_state(),fixed['rng']['torch'])
  assert torch.equal(torch.cuda.get_rng_state(),fixed['rng']['cuda'])
  assert random.getstate()==fixed['rng']['python']
  npnow=np.random.get_state();npold=fixed['rng']['numpy'];assert npnow[0]==npold[0] and np.array_equal(npnow[1],npold[1]) and npnow[2:]==npold[2:]
  # Full-model prefix witness, same T/shape; no changes to real fixture IDs.
  model.eval()
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   ids=fixed['ids'][:1,:-1].cuda();changed=ids.clone();changed[:,1024:]=(changed[:,1024:]+37)%32000
   original=model(ids)[:,:1024];altered=model(changed)[:,:1024]
   causal=float((original-altered).float().square().mean().sqrt()/original.float().square().mean().sqrt().clamp_min(1e-7))
   assert torch.isfinite(altered).all() and causal<1e-5
  rows[-1].update(replica_hashes=hashes,model_adam_rng_restored=True,causal_prefix_rms=causal)
 else:
  for i in range(a.warmup+1):rows.append(dict(phase='cold_first_update' if i==0 else 'warmup',**update()))
  torch.cuda.reset_peak_memory_stats()
  if a.mode=='profile':
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True,with_stack=False) as prof:
    rows.append(dict(phase='profile_not_benchmark',**update()))
   prof.export_chrome_trace(str(a.out/f'trace-rank{rank}.json'))
   (a.out/f'kernels-rank{rank}.txt').write_text(prof.key_averages().table(sort_by='self_device_time_total',row_limit=100))
  else:
   for i in range(a.steps):rows.append(dict(phase='warm',**update()))
 result=dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},rank=rank,world=world,fixture_sha=sha(a.fixture/'fixture.json'),model_checkpoint_sha=receipt['files']['model.pt'],source_files={str(q.relative_to(ROOT/a.variant)):sha(q) for q in (ROOT/a.variant).rglob('*.py')},torch=torch.__version__,triton=importlib.metadata.version('triton'),gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,rows=rows,scope='Fixed real inputs; repeated batch; no loader/tokenization or checkpoint IO in timing. Cold first update includes compilation and lazy setup; backward includes checkpoint replay and DDP. Not a formal training run.')
 (a.out/f'result-rank{rank}.json').write_text(json.dumps(result,indent=2))
 if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
