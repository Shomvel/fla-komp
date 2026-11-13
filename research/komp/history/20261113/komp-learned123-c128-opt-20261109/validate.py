"""Differential production-shape and tail tests against the pinned implementation."""
import gc, hashlib, importlib.util, json, os, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent
os.environ['KOMP_SNAPSHOT'] = str(ROOT/'snapshot')
from models import LM
import fla.ops.komp.block_fast as candidate
spec = importlib.util.spec_from_file_location('fla.ops.komp.block_fast_baseline', ROOT/'snapshot-baseline/fla/ops/komp/block_fast.py')
baseline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)
torch.set_num_threads(2)
torch.manual_seed(734); torch.cuda.manual_seed_all(734)
checkpoint = ROOT/'fixture/checkpoint-010240'
assert (checkpoint/'complete.json').exists()
saved = torch.load(checkpoint/'model.pt', map_location='cpu', weights_only=False)
model = LM('block'); model.load_state_dict(saved['model'], strict=True)
del saved
model = model.cuda().train()
manifest = json.loads((ROOT/'configs/data-manifest.json').read_text())
import numpy as np
data = np.memmap(manifest['validation_tokens_path'], dtype=np.uint16, mode='r')
ids = torch.tensor(np.array(data[:16*2049], dtype=np.int64).reshape(16,2049),device='cuda')
with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
    x = model.model.layers[0].attn_norm(model.model.embeddings(ids[:,:-1]))
    q,k,v,g,beta,keys,values,scores,initial = model.model.layers[0].attn.operands(x)
operands = [z.detach() for z in (q,k,v,g,beta,keys,values,scores)]
initial = initial.detach()
del model,x,ids,q,k,v,g,beta,keys,values,scores
gc.collect();torch.cuda.empty_cache()
results=[]
def run_case(label, src, init, chunk, eviction, final_loss):
    do = torch.randn_like(src[2])*.01
    dfk = torch.randn_like(src[5])*.01
    dfv = torch.randn_like(src[6])*.01
    def run(module):
        xs=[z.detach().clone().requires_grad_(True) for z in src]
        out,(fk,fv)=module.block_komp_fast(*xs[:5],(xs[5],xs[6]),chunk_size=chunk,
                                       evict=eviction,scores=xs[7] if eviction=='learned' else None,
                                       initial_scores=init if eviction=='learned' else None)
        loss=(out.float()*do.float()).sum()
        if final_loss:loss=loss+(fk*dfk).sum()+(fv*dfv).sum()
        loss.backward();torch.cuda.synchronize()
        return [out.detach().cpu(),fk.detach().cpu(),fv.detach().cpu()]+[x.grad.detach().cpu() if x.grad is not None else None for x in xs]
    ref=run(baseline); again=run(baseline); got=run(candidate)
    metrics=[]
    for name,a,b,c in zip(['out','final_keys','final_values','dq','dk','dv','dg','dbeta','dkeys','dvalues','dscores'],ref,again,got):
        if a is None:
            assert b is None and c is None,name
            continue
        assert torch.isfinite(c).all(),(label,name)
        a,b,c=a.float(),b.float(),c.float()
        scale=a.square().mean().sqrt().item()
        err=(c-a).square().mean().sqrt().item()/max(scale,1e-30)
        noise=(b-a).square().mean().sqrt().item()/max(scale,1e-30)
        mx=(c-a).abs().max().item()
        item=dict(name=name,relative_rms=err,baseline_repeat_relative_rms=noise,max_abs=mx,rms=scale)
        metrics.append(item)
        # The changed one-hot contraction is FP32. Account for existing atomics and BF16 rounding.
        limit=max(5e-4,noise*5+1e-5)
        assert err <= limit,(label,item,limit)
        if name in ('out','final_keys','final_values'):
            assert torch.equal(a,c),(label,name,'forward changed')
    result=dict(case=label,shape=list(src[0].shape),slots=src[5].shape[-2],chunk=chunk,
                eviction=eviction,final_state_loss=final_loss,metrics=metrics,passed=True)
    results.append(result);print(json.dumps(result),flush=True)
run_case('production-trained',operands,initial,128,'learned',False)
del operands,initial;gc.collect();torch.cuda.empty_cache()
for T,M,C,eviction,slot,final_loss in [(129,32,128,'learned',True,True),(257,32,128,'fifo',True,True),
                                      (65,16,64,'learned',True,True),(129,64,128,'learned',True,True),
                                      (17,32,128,'learned',True,True),(129,24,128,'learned',True,True),
                                      (129,32,128,'learned',False,True)]:
    B,H,D,V=2,3,64,64
    q,k,v=[torch.randn(B,T,H,D,device='cuda',dtype=torch.bfloat16)*.2 for _ in range(3)]
    g=-torch.rand((B,T,H,M) if slot else (B,T,H),device='cuda')*.01-.001
    beta=torch.rand(B,T,H,device='cuda')*.3
    keys=torch.randn(B,H,M,D,device='cuda')*.2
    values=torch.randn(B,H,M,V,device='cuda')*.1
    scores=torch.randn(B,T,H,device='cuda'); init=torch.randn(B,H,M,device='cuda')
    run_case(f'tail-{T}-m{M}-c{C}-{eviction}-slot{slot}',[q,k,v,g,beta,keys,values,scores],init,C,eviction,final_loss)
(ROOT/'evidence/differential.json').write_text(json.dumps(dict(passed=True,results=results,
    baseline_sha=hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest(),
    candidate_sha=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest()),indent=2)+'\n')
