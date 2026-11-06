"""Real checkpoint first-layer B16/T2048 diagnostic, not full-model throughput."""
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=['baseline','candidate'],required=True);p.add_argument('--cache',action='store_true');p.add_argument('--prune-gc',action='store_true');p.add_argument('--fixture',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 import os
 os.environ['EXACT_KOMP_PRUNE_GC']='1' if a.prune_gc else '0'
 assert not a.prune_gc or a.variant=='candidate'
 assert not a.cache or a.variant=='candidate';a.out.mkdir(parents=True,exist_ok=False)
 sys.path.insert(0,str(ROOT/a.variant));from models import KOMPConfig,KOMPBlock
 from benchmark import sha
 receipt=json.loads((a.fixture/'fixture.json').read_text())
 for name in ('model.pt','input-rank0.pt'):assert sha(a.fixture/name)==receipt['files'][name]
 state=torch.load(a.fixture/'model.pt',map_location='cpu',weights_only=False)['model']
 fixed=torch.load(a.fixture/'input-rank0.pt',map_location='cpu',weights_only=False)
 block=KOMPBlock(KOMPConfig(kind='exact'),0).cuda().train()
 prefix='model.layers.0.';block.load_state_dict({n[len(prefix):]:p for n,p in state.items() if n.startswith(prefix)})
 block.attn.checkpoint_mixer=True
 if a.variant=='candidate':block.attn.cache_geometry=a.cache
 ids=fixed['ids'][:16,:-1];x=torch.nn.functional.embedding(ids,state['model.embeddings.weight']).cuda().requires_grad_();del state,fixed
 # Hooks see autograd saves, not context-owned geometry cache. Always report CUDA peak too.
 def step(measure_saved=False):
  block.zero_grad(set_to_none=True);x.grad=None;seen={}
  def pack(t):
   st=t.untyped_storage();seen[st.data_ptr()]=st.nbytes();return t
  import contextlib
  hooks=torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t) if measure_saved else contextlib.nullcontext()
  ev=[torch.cuda.Event(enable_timing=True) for _ in range(3)];torch.cuda.synchronize();start=time.perf_counter();ev[0].record()
  with hooks,torch.autocast('cuda',dtype=torch.bfloat16):y=block(x)[0]
  ev[1].record();y.float().square().mean().backward();ev[2].record();torch.cuda.synchronize()
  return dict(wall_seconds=time.perf_counter()-start,forward_ms=ev[0].elapsed_time(ev[1]),backward_ms=ev[1].elapsed_time(ev[2]),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),saved_storage_bytes=sum(seen.values()) if measure_saved else None)
 rows=[]
 for i in range(4):rows.append(dict(phase='cold' if i==0 else 'warmup',**step()))
 torch.cuda.reset_peak_memory_stats()
 for _ in range(10):rows.append(dict(phase='warm',**step()))
 rows.append(dict(phase='saved_storage_not_timing',**step(True)))
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:step()
 prof.export_chrome_trace(str(a.out/'trace.json'));(a.out/'kernels.txt').write_text(prof.key_averages().table(sort_by='self_device_time_total',row_limit=100))
 (a.out/'results.json').write_text(json.dumps(dict(variant=a.variant,cache=a.cache,prune_gc=a.prune_gc,shape=[16,2048,1024],fixture_sha=sha(a.fixture/'fixture.json'),scope='First block with real checkpoint embedding input; no DDP/optimizer/head; hooks omit context-owned cache',rows=rows),indent=2))
if __name__=='__main__':main()
