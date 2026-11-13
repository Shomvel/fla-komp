"""Full 1B model loss and all parameter-gradient differential on fixed trained weights."""
import argparse, gc, importlib.util, json, os, sys, hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--snapshot',required=True);ap.add_argument('--tag',required=True);ap.add_argument('--optimizer-check',action='store_true');ap.add_argument('--cache-ut-layers',type=int,default=0);ap.add_argument('--isolate',choices=['all','ff','norm','cache'],default='all');ap.add_argument('--loss-atol',type=float,default=1e-5);args=ap.parse_args()
os.environ['KOMP_SNAPSHOT']=str(ROOT/args.snapshot)
os.environ['KOMP_UT_CACHE_LAYERS']=str(args.cache_ut_layers)
import models
import fla.ops.komp.block_fast as candidate
spec=importlib.util.spec_from_file_location('fla.ops.komp.block_fast_baseline',ROOT/'snapshot-baseline/fla/ops/komp/block_fast.py')
baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
torch.set_num_threads(2);torch.manual_seed(734);torch.cuda.manual_seed_all(734)
state=torch.load(ROOT/'fixture/checkpoint-010240/model.pt',map_location='cpu',weights_only=False)
model=models.LM('block');model.load_state_dict(state['model'],strict=True)
model=model.cuda().train()
source_state=state if args.optimizer_check else None
opt=models.optimizer(model) if args.optimizer_check else None
if not args.optimizer_check:del state
updates=[] 
manifest=json.loads((ROOT/'configs/data-manifest.json').read_text())
data=np.memmap(manifest['validation_tokens_path'],dtype=np.uint16,mode='r')
ids=torch.tensor(np.array(data[:16*2049],dtype=np.int64).reshape(16,2049),device='cuda')
def run(module, optimized):
    models.block_komp_fast=module.block_komp_fast
    import fla.models.komp.modeling_komp as architecture
    if hasattr(architecture,'ROUNDED_SWIGLU'):architecture.ROUNDED_SWIGLU=optimized if args.isolate in ('all','ff') else False
    if 'fla.modules.komp_rmsnorm' in sys.modules:
        sys.modules['fla.modules.komp_rmsnorm'].OPTIMIZED_NORM=optimized if args.isolate in ('all','norm') else False
    for layer in model.model.layers:
        layer.attn.cache_ut_layers=args.cache_ut_layers if (args.isolate!='cache' or optimized) else 0
    model.zero_grad(set_to_none=True)
    if args.optimizer_check:
        model.load_state_dict(source_state['model'],strict=True)
        opt.load_state_dict(source_state['optimizer'])
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(ids[:,:-1],ids[:,1:])/32768
    loss.backward();torch.cuda.synchronize()
    grads={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None}
    assert len(grads)==len(list(model.named_parameters()))
    if args.optimizer_check:
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();torch.cuda.synchronize()
        updates.append({n:p.detach().cpu() for n,p in model.named_parameters()})
    print(json.dumps(dict(phase='backward',source=module.__file__,loss=float(loss))),flush=True)
    return float(loss),grads
reference=baseline if args.isolate=='all' else candidate
loss0,ref=run(reference,False);loss1,noise=run(reference,False);loss2,got=run(candidate,True)
failures=[]
if not abs(loss0-loss2)<=args.loss_atol:failures.append(dict(field='loss',values=[loss0,loss1,loss2],tolerance=args.loss_atol))
metrics=[];total=err2=repeat2=0.
for n,a in ref.items():
    b,c=noise[n],got[n]
    finite=bool(torch.isfinite(c).all())
    if not finite:failures.append(dict(field=n,check='finite'))
    ss=a.double().square().sum().item();ee=(a.double()-c.double()).square().sum().item();nn=(a.double()-b.double()).square().sum().item()
    total+=ss;err2+=ee;repeat2+=nn
    rel=(ee/max(ss,1e-30))**.5;rep=(nn/max(ss,1e-30))**.5
    metrics.append(dict(name=n,relative_rms=rel,baseline_repeat_relative_rms=rep,max_abs=(a-c).abs().max().item()))
    tolerance=max(.005,rep*5+.0001)
    metrics[-1].update(tolerance=tolerance,passed=finite and rel<=tolerance)
    if not rel<=tolerance:failures.append(dict(field=n,check='gradient',relative_rms=rel,baseline_repeat_relative_rms=rep,tolerance=tolerance))
result=dict(passed=not failures,isolate=args.isolate,failures=failures,loss=loss2,reference_loss=loss0,loss_difference=loss2-loss0,loss_tolerance=args.loss_atol,parameters=1003181056,tensors=len(metrics),relative_rms=(err2/total)**.5,
            baseline_repeat_relative_rms=(repeat2/total)**.5,metrics=metrics)
if args.optimizer_check:
    update_metrics=[]
    for n,a in updates[1].items():
        b,c=updates[0][n],updates[2][n]
        diff=(a.double()-c.double()).square().mean().sqrt().item()
        noise=(a.double()-b.double()).square().mean().sqrt().item()
        rms=a.double().square().mean().sqrt().item()
        rel=diff/max(rms,1e-30);repeat=noise/max(rms,1e-30)
        update=(a.double()-source_state['model'][n].double()).square().mean().sqrt().item()
        update_rel=diff/max(update,1e-30);repeat_update=noise/max(update,1e-30)
        parameter_tol=max(5e-5,repeat*5+1e-7);update_tol=max(.02,repeat_update*3+.001)
        if not rel<=parameter_tol:failures.append(dict(field=n,check='parameter_update',relative_rms=rel,tolerance=parameter_tol))
        if not update_rel<=update_tol:failures.append(dict(field=n,check='adam_delta',relative_rms=update_rel,tolerance=update_tol))
        update_metrics.append(dict(name=n,relative_rms=rel,baseline_repeat_relative_rms=repeat,
                                   update_relative_rms=update_rel,baseline_repeat_update_relative_rms=repeat_update,
                                   max_abs=(a-c).abs().max().item()))
    result['optimizer_step']=dict(passed=not any(f['check'] in ('parameter_update','adam_delta') for f in failures if 'check' in f),strict_checkpoint_adam=True,metrics=update_metrics)
result['candidate_source_files']={str(p.relative_to(ROOT/args.snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/args.snapshot).rglob('*.py'))}
result['model_adapter_sha256']=hashlib.sha256((ROOT/'models.py').read_bytes()).hexdigest()
result['passed']=not failures
(ROOT/'evidence'/f'{args.tag}.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='metrics'}),flush=True)

assert result['passed'],failures[:10]
