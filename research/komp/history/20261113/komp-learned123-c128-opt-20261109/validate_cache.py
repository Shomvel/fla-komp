"""Verify cache replay, retained backward, graph isolation and allocation lifetime."""
import argparse,gc,json,os,sys
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--snapshot',default='snapshot-cache');args=ap.parse_args()
sys.path.insert(0,str(ROOT/args.snapshot))
from fla.ops.komp.block_fast import block_komp_fast
from fla.ops.komp.geometry_cache import geometry_contexts

torch.manual_seed(884);torch.cuda.manual_seed_all(884);torch.set_num_threads(2)
B,T,H,D,M=2,257,3,64,32
base=[torch.randn(B,T,H,D,device='cuda',dtype=torch.bfloat16)*.2 for _ in range(3)]
base += [-torch.rand(B,T,H,M,device='cuda')*.01-.001,torch.rand(B,T,H,device='cuda')*.3,
         torch.randn(B,H,M,D,device='cuda')*.2,torch.randn(B,H,M,D,device='cuda')*.1,
         torch.randn(B,T,H,device='cuda')]
initial=torch.randn(B,H,M,device='cuda')
do=torch.randn_like(base[2])*.01

def build(seed,cache):
 xs=[x.detach().clone().requires_grad_() for x in base]
 # Distinct graph inputs, same signatures: proves values cannot leak between graphs.
 with torch.no_grad():xs[2].add_(seed*.01);xs[7].add_(seed*.3)
 def fn(*z):
  o,(k,v)=block_komp_fast(*z[:5],(z[5],z[6]),chunk_size=128,evict='learned',scores=z[7],initial_scores=initial)
  return o,k,v
 kwargs={'context_fn':geometry_contexts} if cache else {}
 out=checkpoint(fn,*xs,use_reentrant=False,**kwargs)
 loss=(out[0].float()*do).sum()+out[1].square().sum()*.01+out[2].square().sum()*.01
 return xs,out,loss

def capture(graph,retain=False):
 xs,out,loss=graph
 grads=torch.autograd.grad(loss,xs,allow_unused=True,retain_graph=retain)
 return [z.detach().cpu() for z in out]+[None if z is None else z.detach().cpu() for z in grads]
results=[]
def compare(label,ref,repeat,got):
 fields=[]
 for name,a,b,c in zip(['out','keys','values','dq','dk','dv','dg','dbeta','dkeys','dvalues','dscores'],ref,repeat,got):
  if a is None:assert c is None;continue
  a,b,c=a.float(),b.float(),c.float()
  scale=a.square().mean().sqrt().clamp_min(1e-30)
  rel=float((a-c).square().mean().sqrt()/scale);noise=float((a-b).square().mean().sqrt()/scale)
  tol=max(5e-4,noise*5+1e-5)
  ok=bool(torch.isfinite(c).all()) and rel<=tol
  if name in ('out','keys','values'):ok=ok and torch.equal(a,c)
  fields.append(dict(name=name,relative_rms=rel,baseline_repeat=noise,tolerance=tol,passed=ok))
 row=dict(case=label,passed=all(f['passed'] for f in fields),metrics=fields)
 results.append(row);print(json.dumps(row),flush=True)

for seed in (0,1):
 ref=capture(build(seed,False));rep=capture(build(seed,False))
 a=build(seed,True);b=build(1-seed,True)
 first=capture(a,True);second=capture(a)
 compare(f'graph{seed}-first',ref,rep,first);compare(f'graph{seed}-retained',ref,rep,second)
 other=capture(b)
 other_ref=capture(build(1-seed,False));other_rep=capture(build(1-seed,False))
 compare(f'graph{seed}-interleaved',other_ref,other_rep,other)
 del a,b,ref,rep,first,second,other,other_ref,other_rep
 gc.collect();torch.cuda.synchronize()
allocated=[]
for i in range(8):
 graph=build(i%2,True);capture(graph);del graph
 gc.collect();torch.cuda.synchronize();allocated.append(torch.cuda.memory_allocated())
assert max(allocated[2:])-min(allocated[2:])<=1024*1024,allocated
result=dict(passed=all(r['passed'] for r in results),results=results,post_graph_allocated_bytes=allocated)
(ROOT/'evidence/cache-differential.json').write_text(json.dumps(result,indent=2)+'\n')
assert result['passed']
print(json.dumps(dict(passed=True,post_graph_allocated_bytes=allocated)),flush=True)
