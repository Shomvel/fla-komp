"""CPU/GPU PyTorch acceptance for value-coordinate KOMP; no training jobs.

Run: python3 validate_value_decay.py --output evidence/value-decay/acceptance.json
The token oracle and triangular implementation use different recurrences.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from komp_slot.block_reference import block_komp_reference, kernel
from komp_slot.value_decay import ValueDecayGate, block_komp_value_decay, exact_komp_value_decay

ROOT = Path(__file__).resolve().parent
CHECKS = []


def close(name, actual, expected, atol=2e-9, rtol=2e-8):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, check_dtype=False)
    assert bool(torch.isfinite(actual).all()), name
    CHECKS.append({'name': name, 'max_abs': float((actual.detach().double()-expected.detach().double()).abs().max()),
                   'atol': atol, 'rtol': rtol})


def sample(t=9, m=3, dk=5, dv=4, dtype=torch.float64, seed=42):
    gen = torch.Generator().manual_seed(seed)
    randn = lambda *s: torch.randn(*s, generator=gen, dtype=dtype)
    return [randn(1,t,2,dk), randn(1,t,2,dk), randn(1,t,2,dv),
            -torch.rand(1,t,2,dv,generator=gen,dtype=dtype)*.15,
            torch.rand(1,t,2,generator=gen,dtype=dtype),
            randn(1,2,m,dk), randn(1,2,m,dv)*.1]


def call(xs, chunk, backend='chunk'):
    out, (keys, coef) = block_komp_value_decay(*xs[:5], (xs[5], xs[6]),
                                             chunk_size=chunk, backend=backend)
    return out, keys, coef


def differential():
    # Unequal key/value/slot widths detect accidental reuse of the slot axis.
    for m, dk, dv, t, chunk in [(3,5,4,9,1),(3,5,4,9,4),(4,4,4,7,3),
                                (2,4,7,5,8),(1,3,2,1,1),(32,64,64,129,128)]:
        base = sample(t,m,dk,dv)
        got = [x.clone().requires_grad_() for x in base]
        ref = [x.clone().requires_grad_() for x in base]
        a, b = call(got,chunk), call(ref,chunk,'recurrent')
        tag = f'differential/M{m}K{dk}V{dv}T{t}C{chunk}'
        weights = [torch.randn_like(x) for x in a]
        for label,x,y in zip(('output','keys','coefficients'),a,b):close(tag+'/'+label,x,y)
        ga = torch.autograd.grad(a,got,weights)
        gb = torch.autograd.grad(b,ref,weights)
        for label,x,y in zip(('q','k','v','g','beta','initial_keys','initial_coef'),ga,gb):
            close(tag+'/grad_'+label,x,y,atol=2e-8,rtol=2e-7)
    # Independent finite differences, including both states and decay gradients.
    xs = [x.requires_grad_() for x in sample(t=3,m=2,dk=2,dv=3)]
    assert torch.autograd.gradcheck(lambda *a: call(a,2),tuple(xs),atol=2e-5,rtol=2e-4)
    CHECKS.append({'name':'finite_difference_gradcheck','passed':True})


def scalar_limit_and_streaming():
    for chunk in (1,4):
        xs=sample(t=9)
        scalar=xs[3][...,0]
        xs[3]=scalar[...,None].expand_as(xs[3])
        actual=call(xs,chunk)
        o,(k,c)=block_komp_reference(*xs[:3],scalar,xs[4],(xs[5],xs[6]),chunk_size=chunk)
        for label,x,y in zip(('out','keys','coef'),actual,(o,k,c)):
            close(f'scalar_limit/C{chunk}/{label}',x,y)
        # Split on a complete compression boundary; final tail remains a tail.
        xs=sample(t=9)
        cut=4
        a=call([x[:,:cut] for x in xs[:5]]+xs[5:],chunk)
        b=call([x[:,cut:] for x in xs[:5]]+[a[1],a[2]],chunk)
        full=call(xs,chunk)
        close(f'streaming/C{chunk}/output',torch.cat((a[0],b[0]),1),full[0])
        close(f'streaming/C{chunk}/coef',b[2],full[2])
    xs=sample()
    o,st=exact_komp_value_decay(*xs[:5],(xs[5],xs[6]))
    expected=call(xs,1)
    for label,x,y in zip(('out','keys','coef'),(o,*st),expected):close('exact_alias/'+label,x,y)


def causality_and_extremes():
    for chunk in (1,4):
        xs=sample(t=7)
        changed=[x.clone() for x in xs]
        for i in (0,1,2):changed[i][:,4:]+=3
        changed[3][:,4:]=-3
        changed[4][:,4:]=0
        close(f'causal/C{chunk}',call(changed,chunk)[0][:,:4],call(xs,chunk)[0][:,:4])
        for extreme in (0.,-1000.):
            xx=[x.clone() for x in xs];xx[3].fill_(extreme)
            xx[4][:,1]=0;xx[4][:,2]=1
            aa=[x.requires_grad_() for x in xx];bb=[x.clone().requires_grad_() for x in xx]
            a,b=call(aa,chunk),call(bb,chunk,'recurrent')
            for label,x,y in zip(('out','keys','coef'),a,b):close(f'extreme/{extreme}/C{chunk}/{label}',x,y)
            ga=torch.autograd.grad(sum(x.square().sum() for x in a),aa)
            gb=torch.autograd.grad(sum(x.square().sum() for x in b),bb)
            for i,(x,y) in enumerate(zip(ga,gb)):close(f'extreme/{extreme}/C{chunk}/grad{i}',x,y)
    xs=sample(t=7)
    xs[0][:,0]=0;xs[1][:,1]=0
    for x,y in zip(call(xs,4),call(xs,4,'recurrent')):close('zero_key_query',x,y)


def geometry_contracts():
    xs=sample();z=F.normalize(xs[5],dim=-1);c=xs[6]
    decay=xs[3][:,0].exp();gram=kernel(z,z,.5)
    energy=lambda a: torch.einsum('bhmv,bhmn,bhnv->bh',a,gram,a)
    before,after=energy(c),energy(c*decay[...,None,:])
    assert bool((after<=before+1e-12).all())
    CHECKS.append({'name':'RKHS_energy_contraction','max_energy_ratio':float((after/before).max())})
    kept=F.normalize(xs[1][:,0:3].transpose(1,2),dim=-1)
    p=torch.linalg.solve(kernel(kept,kept,.5)+1e-4*torch.eye(3),kernel(kept,z,.5))
    close('projection_commutes_with_value_decay',p@(c*decay[...,None,:]),(p@c)*decay[...,None,:])
    # Actual projection's nonlinearity in keys is also covered by differential gradients.


def precision_and_gate():
    for dtype in (torch.float32,torch.bfloat16):
        xs=[x.to(dtype) for x in sample(t=9)]
        actual=call(xs,4);expected=call([x.double() for x in xs],4,'recurrent')
        for i,(x,y) in enumerate(zip(actual,expected)):
            close(f'precision/{dtype}/{i}',x,y,atol=.004 if dtype==torch.bfloat16 else 2e-6,
                  rtol=.008 if dtype==torch.bfloat16 else 2e-5)
        assert actual[0].dtype==dtype and actual[1].dtype==torch.float32 and actual[2].dtype==torch.float32
    gate=ValueDecayGate(12,2,4,rank=3)
    x=torch.randn(2,7,12)
    with torch.autocast('cpu',dtype=torch.bfloat16):g=gate(x)
    close('gate_initial_alpha',g.exp(),torch.full_like(g,.995),atol=1e-7,rtol=1e-7)
    assert g.dtype==torch.float32 and g.shape==(2,7,2,4)
    # Real autograd path from token IDs -> projections/gate -> mixer -> CE -> Adam.
    for chunk in (1,4):
        torch.manual_seed(42)
        embed=nn.Embedding(23,12);qkv=nn.Linear(12,24);beta=nn.Linear(12,2)
        gate=ValueDecayGate(12,2,4,rank=3);head=nn.Linear(8,23)
        initial=nn.Parameter(torch.randn(1,2,3,4))
        params=[*embed.parameters(),*qkv.parameters(),*beta.parameters(),*gate.parameters(),*head.parameters(),initial]
        opt=torch.optim.AdamW(params,lr=3e-4)
        ids=torch.randint(0,23,(1,9));losses=[]
        for step in range(3):
            opt.zero_grad();hidden=embed(ids[:,:-1]);q,k,v=[a.reshape(1,8,2,4) for a in qkv(hidden).chunk(3,-1)]
            out,_=block_komp_value_decay(q,k,v,gate(hidden),beta(hidden).sigmoid(),
                      (initial,torch.zeros(1,2,3,4)),chunk_size=chunk)
            loss=F.cross_entropy(head(out.flatten(-2)).flatten(0,1),ids[:,1:].flatten())
            loss.backward()
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
            if step==0:assert bool((gate.up.weight.grad!=0).any())
            if step==2:assert bool((gate.down.weight.grad!=0).any())
            opt.step();losses.append(float(loss.detach()))
        CHECKS.append({'name':f'synthetic_LM_Adam_smoke/C{chunk}','losses':losses,'gate_both_factors_receive_gradient':True})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);start=time.monotonic()
    report={'status':'running','scope':'CPU synthetic PyTorch operator validation, not CUDA/DDP/native training acceptance',
            'torch':torch.__version__,'checks':CHECKS}
    try:
        for suite in (differential,scalar_limit_and_streaming,causality_and_extremes,geometry_contracts,precision_and_gate):
            suite();print('PASS',suite.__name__,flush=True)
        report['status']='passed'
    except Exception as exc:
        report['status']='failed';report['error']=repr(exc);raise
    finally:
        report['seconds']=time.monotonic()-start
        report['sources']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('komp_slot/value_decay.py','komp_slot/block_reference.py','validate_value_decay.py')}
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
