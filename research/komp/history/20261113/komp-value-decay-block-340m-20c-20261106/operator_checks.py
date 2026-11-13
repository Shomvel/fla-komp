"""Frozen-value-decay oracle checks on random and real model operands."""
import json
import torch
from models import ROOT
from fla.ops.komp import block_komp_value_decay

def compare(xs, name):
    actual=[x.detach().requires_grad_() for x in xs]
    reference=[x.detach().double().requires_grad_() for x in xs]
    def run(a,backend):
        o,(k,s)=block_komp_value_decay(*a[:5],(a[5],a[6]),chunk_size=128,backend=backend)
        return o,k,s
    a=run(actual,'triton');r=run(reference,'recurrent')
    weights=[torch.randn_like(x) for x in a]
    da=torch.autograd.grad(a,actual,weights)
    dr=torch.autograd.grad(r,reference,[x.double() for x in weights])
    rows=[];tol=.025 if xs[0].dtype==torch.bfloat16 else .003
    for label,x,y in zip(('out','keys','state','dq','dk','dv','dg','dbeta','dk0','ds0'),(*a,*da),(*r,*dr)):
        x=x.double();d=x-y
        absolute=float(d.abs().max());relative=float(d.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-12))
        rows.append(dict(tensor=label,absolute=absolute,rmsrel=relative,passed=bool(torch.isfinite(x).all()) and (absolute<=3e-4 or relative<=tol)))
    return dict(case=name,tolerance=tol,rows=rows,passed=all(x['passed'] for x in rows))

def run_checks(model,stream):
    torch.manual_seed(1234)
    cases=[]
    for dtype in (torch.float32,torch.bfloat16):
        t=129
        xs=[torch.randn(1,t,2,64,device='cuda',dtype=dtype) for _ in range(3)]
        xs += [-torch.rand(1,t,2,64,device='cuda')*.1,torch.rand(1,t,2,device='cuda'),torch.randn(1,2,32,64,device='cuda'),torch.randn(1,2,32,64,device='cuda')*.1]
        cases.append(compare(xs,'random-tail-'+str(dtype)))
    saved=stream.state_dict();ids=stream.next()[:130].cuda()[None];stream.load_state_dict(saved)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        hidden=model.model.embeddings(ids[:,:-1]);hidden=model.model.layers[0].attn_norm(hidden)
        xs=list(model.model.layers[0].attn.operands(hidden))
    xs=[x[:,:,:2].contiguous() for x in xs[:5]]+[x[:,:2].contiguous() for x in xs[5:]]
    cases.append(compare(xs,'real-first-layer-tail'))
    out=dict(status='passed' if all(c['passed'] for c in cases) else 'failed',cases=cases)
    (ROOT/'evidence/operator-block.json').write_text(json.dumps(out,indent=2))
    assert out['status']=='passed',out
    print('VALUE_DECAY_OPERATOR_PASS',flush=True)
