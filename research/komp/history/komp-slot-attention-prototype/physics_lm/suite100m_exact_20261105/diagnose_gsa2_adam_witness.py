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
selected=('embed.weight','layers.0.mix.layer.o_proj.weight','layers.0.mix.out.weight','head.weight')
prior=ROOT/'evidence/gsa2-bf16-gradient-v1.json'
report=json.loads(prior.read_text()) if prior.exists() else {'status':'running','sources':d.digest(),'cases':{}}
assert report['sources']==d.digest()
for kind,case in [(kind,case) for kind in ('gsa2','full') for case in ('mano','depo2')]:
    label=f'{kind}/{case}'
    if label in report['cases'] and 'adam_witness' in report['cases'][label]:
        result=report['cases'][label]
        result['accepted']=result['loss_abs']<.005 and max(result['gradient_rms'].values())<.02 and max(result['parameter_max_abs'].values())<1e-4
        result['thresholds']={'loss_abs':.005,'gradient_rms':.02,'parameter_max_abs':1e-4}
        print('REUSED_NATIVE_DIAGNOSTIC',label,result['accepted'],flush=True)
        continue
    rows=([Factory('mano').row(random.Random(923+i)) for i in range(17)] if case=='mano' else [d.base.Corpus('depo2','train','tune').row(i) for i in range(17)])
    model=LM(kind,731).cuda().train()
    initial={n:x.detach().cpu().clone() for n,x in model.state_dict().items()}
    initial_selected={n:p.detach().clone() for n,p in model.named_parameters() if n in selected}
    opt=optimizer(model,1e-4,.01)
    ids,mask=d.base.batch(rows)
    with d.amp('bf16'):loss,count=d.base.loss_sum(model,ids,mask);normalized=loss/count
    normalized.backward();whole_loss=normalized.item()
    whole_grad={n:p.grad.detach().clone() for n,p in model.named_parameters() if n in selected}
    whole_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.).item();opt.step()
    whole_param={n:p.detach().clone() for n,p in model.named_parameters() if n in selected}
    with torch.no_grad(),d.amp('bf16'):
        whole_post_loss,_=d.base.loss_sum(model,ids,mask)
        whole_post_loss=whole_post_loss.item()/count
        whole_hidden=model(ids)[:,:-1][mask[:,1:]].float()
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
    partition_raw={n:p.grad.detach().clone() for n,p in model.named_parameters() if n in selected}
    partition_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.).item();opt.step()
    params={n:(p-whole_param[n]).abs().max().item() for n,p in model.named_parameters() if n in selected}
    result={'whole_loss':whole_loss,'partition_loss':value,'loss_abs':abs(whole_loss-value),'gradient_rms':grads,'parameter_max_abs':params,'whole_grad_norm':whole_norm,'partition_grad_norm':partition_norm}
    witness={}
    for name,p in model.named_parameters():
        if name not in selected:continue
        delta=p-whole_param[name];index=delta.abs().argmax().item()
        grad0=whole_grad[name].flatten()[index].item();grad1=partition_raw[name].flatten()[index].item()
        update=whole_param[name]-initial_selected[name]
        witness[name]={'flat_index':index,'whole_raw_gradient':grad0,'partition_raw_gradient':grad1,'whole_clipped_gradient':grad0*min(1.,1/(whole_norm+1e-6)),'partition_clipped_gradient':grad1*min(1.,1/(partition_norm+1e-6)),'whole_gradient_rms':whole_grad[name].square().mean().sqrt().item(),'parameter_difference':delta.flatten()[index].item(),'parameter_rms_relative':(delta.square().mean().sqrt()/whole_param[name].square().mean().sqrt().clamp_min(1e-8)).item(),'update_rms_relative':(delta.square().mean().sqrt()/update.square().mean().sqrt().clamp_min(1e-8)).item(),'sign_disagreement_fraction':((whole_grad[name].sign()!=partition_raw[name].sign()) & ((whole_grad[name]!=0)|(partition_raw[name]!=0))).float().mean().item()}
    ids,mask=d.base.batch(rows)
    with torch.no_grad(),d.amp('bf16'):
        partition_post_loss,_=d.base.loss_sum(model,ids,mask)
        partition_post_loss=partition_post_loss.item()/count
        hidden=model(ids)[:,:-1][mask[:,1:]].float()
    result['adam_witness']=witness
    result['post_step_function']={'whole_loss':whole_post_loss,'partition_loss':partition_post_loss,'loss_abs':abs(whole_post_loss-partition_post_loss),'hidden_rms_relative':((hidden-whole_hidden).square().mean().sqrt()/whole_hidden.square().mean().sqrt().clamp_min(1e-8)).item()}
    report['cases'][label]=result
    print('BF16_BATCH_GRADIENT',label,json.dumps(result),flush=True)
    (ROOT/'evidence/gsa2-bf16-adam-witness.json').write_text(json.dumps(report,indent=2)+'\n')
    result['accepted']=result['loss_abs']<.005 and max(grads.values())<.02 and max(params.values())<1e-4
    result['thresholds']={'loss_abs':.005,'gradient_rms':.02,'parameter_max_abs':1e-4}
    del model,opt,whole_grad,whole_param,ids,mask,loss,normalized,whole_hidden,hidden,partition_raw,initial_selected
    torch.cuda.empty_cache()
layout=json.loads((ROOT/'evidence/gsa2-layout-diagnostic.json').read_text())
assert layout['sources']==d.digest()
for kind,checks in layout['checks']['bf16'].items():
    assert all(row['max_abs']==0 for row in checks),(kind,checks)
accepted=all(x['accepted'] for k,x in report['cases'].items() if k.startswith('gsa2/'))
report.update(status='completed',accepted=accepted,verdict='PASSED' if accepted else 'FAILED',scope='BF16 training only; FP32 whole-batch shape sensitivity retained as a limitation',layout_sha256=hashlib.sha256((ROOT/'evidence/gsa2-layout-diagnostic.json').read_bytes()).hexdigest())
(ROOT/'evidence/gsa2-bf16-adam-witness.json').write_text(json.dumps(report,indent=2)+'\n')
if accepted:
    (ROOT/'evidence/gsa2-layout-acceptance.json').write_text(json.dumps({**report,'status':'passed'},indent=2)+'\n')
print('NATIVE_GSA2_BF16_VERDICT',report['verdict'],flush=True)
