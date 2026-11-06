"""Attribute the first nonzero real-data layout difference to a layer substage."""
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import ddp_train as d
from models import LM
import fla.layers.gsa2 as native

torch.set_num_threads(2)
torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32=False
model=LM('gsa2',731,checkpoint_layers=False).cuda().train()
rows=[d.base.Corpus('depo2','train','tune').row(i) for i in range(17)]
ids,mask=d.base.batch(rows)
target=0
current_position=0
current_batch=17
last_output=None
length=len(rows[0]['input_ids'])
trace={}
full_inputs={}
capture_inputs=False


def first(x):
    if isinstance(x,(tuple,list)):
        x=x[0]
    if x.shape[0]==1 and x.shape[1]==current_batch*ids.shape[1]:
        x=x.reshape(current_batch,ids.shape[1],*x.shape[2:])
    return x[current_position,:length].detach().float().cpu()


def wrapper(name,original):
    def call(**kwargs):
        for key in ('q','k','v','g','gv','b','c','w'):
            if key in kwargs and torch.is_tensor(kwargs[key]):
                trace[name+'.'+key]=first(kwargs[key])
        if capture_inputs:
            full_inputs[name]={k:(v.detach().clone() if torch.is_tensor(v) else v) for k,v in kwargs.items()}
        result=original(**kwargs)
        trace[name+'.output']=first(result)
        return result
    return call


original_oja=native.chunk_gated_oja_rule2
original_gdn=native.chunk_gdn2
native.chunk_gated_oja_rule2=wrapper('oja',original_oja)
native.chunk_gdn2=wrapper('gdn2',original_gdn)
hooks=[]
layer=model.layers[0]
for name,module in layer.named_modules():
    if name and (isinstance(module,torch.nn.Linear) or name in ('n1','n2','mix.layer.q_conv1d','mix.layer.k_conv1d','mix.layer.v_conv1d','mix.layer.w_conv1d','mix.layer.o_norm')):
        hooks.append(module.register_forward_hook(lambda m,i,o,n=name:trace.__setitem__(n,first(o))))


def run(indices,training):
    global trace,current_position,current_batch,last_output
    trace={}
    current_position=indices.index(target) if target in indices else 0
    current_batch=len(indices)
    layer.train(training)
    with torch.no_grad(),d.amp('bf16'):
        output=layer(model.embed(ids[indices]))
    last_output=output.detach().float().cpu()
    trace['block.output']=first(output)
    return trace


def metric(a,b):
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {'finite':finite,'rms_relative':float((a-b).square().mean().sqrt()/a.square().mean().sqrt().clamp_min(1e-8)) if finite else None,
            'max_abs':float((a-b).abs().max()) if finite else None,'reference_rms':float(a.square().mean().sqrt())}


run(list(range(17)),True)
whole=last_output.clone()
per_doc={}
for rank in range(8):
    indices=list(range(rank,17,8))
    run(indices,True)
    for j,index in enumerate(indices):
        n=len(rows[index]['input_ids']);per_doc[index]=metric(whole[index,:n],last_output[j,:n])
target=max(per_doc,key=lambda i:per_doc[i]['rms_relative'])
length=len(rows[target]['input_ids'])
target_group=list(range(target%8,17,8))
print('SELECTED_DOCUMENT',target,per_doc[target],flush=True)
capture_inputs=True
baseline=run(list(range(17)),True)
capture_inputs=False
report={'status':'running','sources':d.digest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Maximum first-layer difference document from real Depo17-row witness; first block only, shared weights and exact document prefix, no training',
        'target_document':target,'per_document_first_layer':per_doc,'length':length,'padded_length':ids.shape[1],'checks':{},'norms':{}}
for name,indices,training in [('partition',target_group,True),('single',[target],True),('dense',list(range(17)),False)]:
    actual=run(indices,training)
    report['checks'][name]={key:metric(value,actual[key]) for key,value in baseline.items()}
for name in ['oja.q','oja.k','oja.v','oja.output','gdn2.q','gdn2.k','gdn2.output']:
    values=baseline[name].norm(dim=-1).flatten()
    report['norms'][name]={'min':float(values.min()),'p01':float(torch.quantile(values,.01)),
                           'p50':float(values.median()),'max':float(values.max())}
# All direct operator layouts below share exactly the captured per-document inputs.
for name,op in [('oja',original_oja),('gdn2',original_gdn)]:
    args=full_inputs[name]
    current_batch=17;current_position=target
    with torch.no_grad(),d.amp('bf16'):reference=first(op(**args))
    for label,indices,dense in [('subset_packed',target_group,False),('all_dense',list(range(17)),True),('single_dense',[target],True)]:
        current_batch=len(indices);current_position=indices.index(target)
        changed={}
        for key,value in args.items():
            if torch.is_tensor(value) and value.ndim>=3:
                value=value.reshape(17,ids.shape[1],*value.shape[2:])[indices].contiguous()
                if not dense:value=value.reshape(1,current_batch*ids.shape[1],*value.shape[2:])
            changed[key]=value
        changed['cu_seqlens']=None if dense else torch.arange(0,(current_batch+1)*ids.shape[1],ids.shape[1],device='cuda',dtype=torch.int32)
        changed['cu_seqlens_cpu']=None
        with torch.no_grad(),d.amp('bf16'):actual=first(op(**changed))
        report['checks'][name+'_identical_inputs_'+label]=metric(reference,actual)
report['status']='completed'
(ROOT/'evidence/gsa2-worst-layer.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report),flush=True)
