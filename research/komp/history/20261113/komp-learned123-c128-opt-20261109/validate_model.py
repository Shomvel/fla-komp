"""Full 1B model loss and all parameter-gradient differential on fixed trained weights."""
import gc, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
os.environ['KOMP_SNAPSHOT']=str(ROOT/'snapshot')
import models
import fla.ops.komp.block_fast as candidate
spec=importlib.util.spec_from_file_location('fla.ops.komp.block_fast_baseline',ROOT/'snapshot-baseline/fla/ops/komp/block_fast.py')
baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
torch.set_num_threads(2);torch.manual_seed(734);torch.cuda.manual_seed_all(734)
state=torch.load(ROOT/'fixture/checkpoint-010240/model.pt',map_location='cpu',weights_only=False)
model=models.LM('block');model.load_state_dict(state['model'],strict=True);del state
model=model.cuda().train()
manifest=json.loads((ROOT/'configs/data-manifest.json').read_text())
data=np.memmap(manifest['validation_tokens_path'],dtype=np.uint16,mode='r')
ids=torch.tensor(np.array(data[:16*2049],dtype=np.int64).reshape(16,2049),device='cuda')
def run(module):
    models.block_komp_fast=module.block_komp_fast;model.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(ids[:,:-1],ids[:,1:])/32768
    loss.backward();torch.cuda.synchronize()
    grads={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None}
    assert len(grads)==len(list(model.named_parameters()))
    print(json.dumps(dict(phase='backward',source=module.__file__,loss=float(loss))),flush=True)
    return float(loss),grads
loss0,ref=run(baseline);loss1,noise=run(baseline);loss2,got=run(candidate)
assert loss0==loss1==loss2,(loss0,loss1,loss2)
metrics=[];total=err2=repeat2=0.
for n,a in ref.items():
    b,c=noise[n],got[n]
    assert torch.isfinite(c).all(),n
    ss=a.double().square().sum().item();ee=(a.double()-c.double()).square().sum().item();nn=(a.double()-b.double()).square().sum().item()
    total+=ss;err2+=ee;repeat2+=nn
    rel=(ee/max(ss,1e-30))**.5;rep=(nn/max(ss,1e-30))**.5
    metrics.append(dict(name=n,relative_rms=rel,baseline_repeat_relative_rms=rep,max_abs=(a-c).abs().max().item()))
    assert rel<=max(.005,rep*5+.0001),(n,rel,rep)
result=dict(passed=True,loss=loss2,parameters=1003181056,tensors=len(metrics),relative_rms=(err2/total)**.5,
            baseline_repeat_relative_rms=(repeat2/total)**.5,metrics=metrics)
(ROOT/'evidence/model-differential.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='metrics'}),flush=True)
