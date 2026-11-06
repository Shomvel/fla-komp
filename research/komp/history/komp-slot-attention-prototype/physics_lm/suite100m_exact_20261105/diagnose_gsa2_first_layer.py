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
length=len(rows[0]['input_ids'])
trace={}
full_inputs={}
capture_inputs=False


def first(x):
    if isinstance(x,(tuple,list)):
        x=x[0]
    return x[0,:length].detach().float().cpu()


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
    global trace
    trace={}
    layer.train(training)
    with torch.no_grad(),d.amp('bf16'):
        output=layer(model.embed(ids[indices]))
    trace['block.output']=first(output)
    return trace


def metric(a,b):
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {'finite':finite,'rms_relative':float((a-b).square().mean().sqrt()/a.square().mean().sqrt().clamp_min(1e-8)) if finite else None,
            'max_abs':float((a-b).abs().max()) if finite else None,'reference_rms':float(a.square().mean().sqrt())}


capture_inputs=True
baseline=run(list(range(17)),True)
capture_inputs=False
report={'status':'running','sources':d.digest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'First document of real Depo17-row witness; first block only, shared weights and exact document prefix, no training',
        'length':length,'padded_length':ids.shape[1],'checks':{},'norms':{}}
for name,indices,training in [('partition',[0,8,16],True),('single',[0],True),('dense',list(range(17)),False)]:
    actual=run(indices,training)
    report['checks'][name]={key:metric(value,actual[key]) for key,value in baseline.items()}
for name in ['oja.q','oja.k','oja.v','oja.output','gdn2.q','gdn2.k','gdn2.output']:
    values=baseline[name].norm(dim=-1).flatten()
    report['norms'][name]={'min':float(values.min()),'p01':float(torch.quantile(values,.01)),
                           'p50':float(values.median()),'max':float(values.max())}
# Freeze exact operator input values; dense versus packed now removes upstream projection differences.
for name,op in [('oja',original_oja),('gdn2',original_gdn)]:
    args=full_inputs[name]
    packed={k:(v[:,:ids.shape[1]].clone() if torch.is_tensor(v) and v.ndim>=3 else v) for k,v in args.items()}
    packed['cu_seqlens']=torch.tensor([0,ids.shape[1]],device='cuda',dtype=torch.int32)
    packed['cu_seqlens_cpu']=None
    dense={**packed,'cu_seqlens':None}
    with torch.no_grad(),d.amp('bf16'):
        a=first(op(**packed));b=first(op(**dense))
    report['checks'][name+'_same_input_dense_vs_packed']=metric(a,b)
report['status']='completed'
(ROOT/'evidence/gsa2-first-layer.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report),flush=True)
