"""Shared-cotangent backward localization, no optimizer or formal training."""
import hashlib,importlib,json,sys
from pathlib import Path
import torch,triton
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import ddp_train as d
from models import LM
import gsa2_packed_eval_candidate as candidate
configuration=candidate.install()
# L2 normalization has the same token-bucket autotuner as the output RMSNorm.
# Fix both forward/backward launch layouts; formulas/dtypes stay unchanged.
l2=importlib.import_module('fla.modules.norm.l2norm')
for name in ('l2norm_fwd_kernel','l2norm_bwd_kernel'):
    tuner=getattr(l2,name)
    assert hasattr(tuner,'configs') and hasattr(tuner,'cache')
    tuner.configs=[triton.Config({'BT':32},num_warps=4)];tuner.cache.clear()
    configuration.append({'kernel':name,'BT':32,'num_warps':4})
torch.set_num_threads(2);torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
model=LM('gsa2',731,checkpoint_layers=False).cuda().train()
rows=[d.base.Corpus('depo2','train','tune').row(i) for i in range(17)]
ids,mask=d.base.batch(rows)
valid=torch.arange(ids.shape[1])[None]<torch.tensor([len(r['input_ids']) for r in rows])[:,None]
torch.manual_seed(831)
upstream=torch.randn(*ids.shape,512,device='cuda')*valid.cuda().unsqueeze(-1)/valid.sum().item()
selected=('embed.weight','layers.0.mix.layer.o_proj.weight','layers.14.mix.layer.o_proj.weight')
trace={};current=[]

def capture(name,x):
    x.register_hook(lambda gradient:trace.__setitem__(name,gradient.detach().float().cpu()))

hooks=[layer.register_forward_hook(lambda m,i,o,n=f'layer{index}':capture(n,o)) for index,layer in enumerate(model.layers)]

def run(indices):
    global trace
    trace={}
    with d.amp('bf16'):
        output=model(ids[indices])
        loss=(output*upstream[indices]).sum()
    loss.backward()
    return output.detach().float().cpu(),trace

whole,ref=run(list(range(17)))
reference_grad={n:p.grad.detach().clone() for n,p in model.named_parameters() if n in selected}
model.zero_grad(set_to_none=True)
combined=torch.zeros_like(whole)
backward={name:torch.zeros_like(value) for name,value in ref.items()}
for rank in range(8):
    indices=list(range(rank,17,8))
    actual,values=run(indices)
    combined[indices]=actual
    for name,value in values.items():backward[name][indices]=value

def metric(a,b):
    return {'rms_relative':float((a-b).square().mean().sqrt()/a.square().mean().sqrt().clamp_min(1e-8)),'max_abs':float((a-b).abs().max()),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all())}
report={'status':'completed','sources':d.digest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'configuration':configuration,'optimizer_steps':0,'shared_upstream_seed':831,'forward':metric(whole[valid],combined[valid]),'activation_gradients':{name:metric(value[valid],backward[name][valid]) for name,value in ref.items()},'parameter_gradients':{n:metric(reference_grad[n],p.grad) for n,p in model.named_parameters() if n in selected}}
(ROOT/'evidence/gsa2-backward-stages.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report),flush=True)
