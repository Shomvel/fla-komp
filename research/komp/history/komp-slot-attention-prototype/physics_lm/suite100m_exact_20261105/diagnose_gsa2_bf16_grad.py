"""Test the actual BF16 training path across equivalent batch partitions."""
import hashlib,json,math,random,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import ddp_train as d
from models import LM,optimizer
from tasks import Factory

torch.set_num_threads(2);torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
selected=('embed.weight','layers.0.mix.layer.o_proj.weight','head.weight')
prior=ROOT/'evidence/gsa2-bf16-gradient-v1.json'
report=json.loads(prior.read_text()) if prior.exists() else {'status':'running','sources':d.digest(),'cases':{}}
assert report['sources']==d.digest()
for case in ['mano','depo2']:
    if case in report['cases']:
        result=report['cases'][case]
        result['accepted']=result['loss_abs']<.005 and max(result['gradient_rms'].values())<.02 and max(result['parameter_max_abs'].values())<1e-4
        result['thresholds']={'loss_abs':.005,'gradient_rms':.02,'parameter_max_abs':1e-4}
        print('REUSED_NATIVE_DIAGNOSTIC',case,result['accepted'],flush=True)
        continue
    rows=([Factory('mano').row(random.Random(923+i)) for i in range(17)] if case=='mano' else [d.base.Corpus('depo2','train','tune').row(i) for i in range(17)])
    model=LM('gsa2',731).cuda().train()
    initial={n:x.detach().cpu().clone() for n,x in model.state_dict().items()}
    opt=optimizer(model,1e-4,.01)
    ids,mask=d.base.batch(rows)
    with d.amp('bf16'):loss,count=d.base.loss_sum(model,ids,mask);normalized=loss/count
    normalized.backward();whole_loss=normalized.item()
    whole_grad={n:p.grad.detach().clone() for n,p in model.named_parameters() if n in selected}
    whole_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.).item();opt.step()
    whole_param={n:p.detach().clone() for n,p in model.named_parameters() if n in selected}
    model.load_state_dict(initial);del initial
    opt=optimizer(model,1e-4,.01);opt.zero_grad(set_to_none=True)
    value=0.;padded=math.ceil(max(len(r['input_ids']) for r in rows)/16)*16
    micro=min(math.ceil(len(rows)/8),2**int(math.log2(max(1,8192/padded))))
    for rank in range(8):
        local=rows[rank::8]
        for start in range(0,len(local),micro):
            ids,mask=d.base.batch(local[start:start+micro])
            with d.amp('bf16'):loss,_=d.base.loss_sum(model,ids,mask);normalized=loss/count
            normalized.backward();value+=normalized.item()
    grads={}
    for n,p in model.named_parameters():
        if n in selected:
            ref=whole_grad[n];delta=p.grad-ref
            grads[n]=(delta.square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-8)).item()
    partition_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.).item();opt.step()
    params={n:(p-whole_param[n]).abs().max().item() for n,p in model.named_parameters() if n in selected}
    result={'whole_loss':whole_loss,'partition_loss':value,'loss_abs':abs(whole_loss-value),'gradient_rms':grads,'parameter_max_abs':params,'whole_grad_norm':whole_norm,'partition_grad_norm':partition_norm}
    report['cases'][case]=result
    print('BF16_BATCH_GRADIENT',case,json.dumps(result),flush=True)
    (ROOT/'evidence/gsa2-bf16-gradient-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    result['accepted']=result['loss_abs']<.005 and max(grads.values())<.02 and max(params.values())<1e-4
    result['thresholds']={'loss_abs':.005,'gradient_rms':.02,'parameter_max_abs':1e-4}
    del model,opt,whole_grad,whole_param,ids,mask,loss,normalized
    torch.cuda.empty_cache()
layout=json.loads((ROOT/'evidence/gsa2-layout-diagnostic.json').read_text())
assert layout['sources']==d.digest()
for kind,checks in layout['checks']['bf16'].items():
    assert all(row['max_abs']==0 for row in checks),(kind,checks)
accepted=all(x['accepted'] for x in report['cases'].values())
report.update(status='completed',accepted=accepted,verdict='PASSED' if accepted else 'FAILED',scope='BF16 training only; FP32 whole-batch shape sensitivity retained as a limitation',layout_sha256=hashlib.sha256((ROOT/'evidence/gsa2-layout-diagnostic.json').read_bytes()).hexdigest())
(ROOT/'evidence/gsa2-bf16-gradient-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
if accepted:
    (ROOT/'evidence/gsa2-layout-acceptance.json').write_text(json.dumps({**report,'status':'passed'},indent=2)+'\n')
print('NATIVE_GSA2_BF16_VERDICT',report['verdict'],flush=True)
