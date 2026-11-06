"""CPU FP64 proof checks, gradients, masks, baseline equivalence and counterexamples."""
import importlib.util
import json
from pathlib import Path
import torch
from prototype import forward,kernel,compress

torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parent;report={'checks':[]}

def record(name,error,tol=1e-9):
    row=dict(name=name,max_abs=float(error),tolerance=tol,passed=float(error)<=tol)
    report['checks'].append(row);assert row['passed'],row


def check(name,a,b,tol=1e-9):record(name,(a-b).detach().abs().max(),tol)


def inputs(t,m,d=5,v=3,slot=False):
    return [torch.randn(t,d),torch.randn(t,d),torch.randn(t,v)*.2,
            -torch.rand((t,m) if slot else (t,))*.1,torch.rand(t)*.7,
            torch.randn(m,d),torch.randn(m,v)*.1]


def main():
    torch.manual_seed(237)
    for slot in (False,True):
        for t,m,c,p in ((1,3,4,2),(11,3,4,2),(13,5,3,4),(17,3,6,2)):
            for pw,pc in ((False,False),(True,False),(False,True),(True,True)):
                raw=inputs(t,m,slot=slot)
                if t>1:raw[3][1]=-1000;raw[4][::3]=0
                a=[x.detach().requires_grad_() for x in raw];b=[x.detach().requires_grad_() for x in raw]
                kw=dict(chunk=c,p=p,protect_write=pw,protect_compress=pc)
                out=forward(*a,mode='token',**kw);ref=forward(*b,mode='chunk',**kw)
                ws=[torch.randn_like(x) for x in out];ga=torch.autograd.grad(out,a,ws);gb=torch.autograd.grad(ref,b,ws)
                tag=f'{slot}/{t}/{m}/{c}/{p}/write{pw}/compress{pc}'
                for i,(x,y) in enumerate(zip((*out,*ga),(*ref,*gb))):check(tag+f'/{i}',x,y)
    # p=0 must collapse to the frozen baseline (including gradients).
    path=Path('/Users/shom/Downloads/komp-accel-scratch/frozen-20261101-r3/komp_slot/block_reference.py')
    spec=importlib.util.spec_from_file_location('block_ref',path);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    for slot in (False,True):
        raw=inputs(13,4,slot=slot);a=[x.detach().requires_grad_() for x in raw];b=[x.detach().requires_grad_() for x in raw]
        out=forward(*a,chunk=6,p=0,protect_compress=True,mode='chunk')
        q,k,v,g,beta,keys,state=b
        bo,(bk,bs)=base.block_komp_reference(q[None,:,None],k[None,:,None],v[None,:,None],g[None,:,None],beta[None,:,None],(keys[None,None],state[None,None]),chunk_size=6)
        ref=(bo[0,:,0],bk[0,0],bs[0,0]);w=[torch.randn_like(x) for x in out]
        ga=torch.autograd.grad(out,a,w);gb=torch.autograd.grad(ref,b,w)
        for i,(x,y) in enumerate(zip((*out,*ga),(*ref,*gb))):check(f'p0_baseline/{slot}/{i}',x,y)
    # Suffix interventions and gradients across both sides of chunk boundaries.
    for slot in (False,True):
        for pc in (False,True):
            raw=inputs(13,4,slot=slot);kw=dict(chunk=4,p=2,protect_compress=pc,mode='chunk')
            with torch.no_grad():
                ref=forward(*raw,**kw)[0]
                for cut in (1,3,4,5,8,9,12):
                    for component in range(5):
                        changed=[x.clone() for x in raw];changed[component][cut:]+=.3
                        out=forward(*changed,**kw)[0]
                        check(f'causal/{slot}/{pc}/{cut}/{component}',out[:cut],ref[:cut],0)
            x=[a.detach().requires_grad_() for a in raw];out=forward(*x,**kw)
            grad=torch.autograd.grad(out[0][:5].square().sum(),x)
            for i,g in enumerate(grad[:5]):record(f'future_grad/{slot}/{pc}/{i}',g[5:].abs().max(),0)
    # The first one-token block has one real anchor, not p initial dictionary keys.
    raw=inputs(1,4)
    one=forward(*raw,chunk=4,p=1,protect_compress=True,mode='chunk')
    many=forward(*raw,chunk=4,p=3,protect_compress=True,mode='chunk')
    for i,(a,b) in enumerate(zip(one,many)):check(f'startup_real_anchors/{i}',a,b,0)
    # End-of-block protected compression contracts the anchor prediction error.
    torch.manual_seed(81);u=torch.nn.functional.normalize(torch.randn(8,5),dim=-1);s=torch.randn(8,3)
    j0=compress(u,4,4,2,.01,False,.05);jp=compress(u,4,4,2,.01,True,.05)
    a=kernel(u[-2:],u[4:]);target=kernel(u[-2:],u)@s
    old=(a@j0@s-target).norm();new=(a@jp@s-target).norm()
    assert new<=old
    report['compression_error']={'before':float(old),'after':float(new),'ratio':float(new/old)}
    # A numerical counterexample: full-block ridge metric includes a future atom.
    torch.manual_seed(91)
    centers=torch.nn.functional.normalize(torch.randn(4,5),dim=-1)
    anchor=centers[:2];query=torch.nn.functional.normalize(torch.randn(1,5),dim=-1)
    def leaked(future):
        u=torch.cat((centers, future));metric=kernel(u,u)+.01*torch.eye(len(u))
        basis=kernel(u,anchor);x=torch.linalg.solve(metric,basis)
        e=torch.tensor([1.,0.]);w=x@torch.linalg.solve(basis.T@x+.05*torch.eye(2),e)
        return (kernel(query,u)@w).squeeze()
    future1=torch.nn.functional.normalize(torch.randn(1,5),dim=-1)
    future2=torch.nn.functional.normalize(torch.randn(1,5),dim=-1)
    delta=abs(float(leaked(future1)-leaked(future2)));assert delta>1e-8
    report['naive_full_block_future_leak']=delta
    # Independent primal proximal normal equations for the local direction.
    gram=kernel(centers[:3],centers[:3]);rhs=torch.tensor([1.05,0.,0.]);coeff=torch.linalg.solve(gram+.05*torch.eye(3),rhs)
    # Min 1/2 c^T G c + 1/(2 lambda)||G c-rhs||^2.
    primal=torch.linalg.solve(gram+gram@gram/.05,gram@rhs/.05)
    check('local_KAPA_proximal',coeff,primal)
    # Identical keys: protection and fitting conflict but regularized direction finite.
    duplicate=centers[:1].expand(3,-1);a=torch.linalg.solve(kernel(duplicate,duplicate)+.05*torch.eye(3),rhs)
    responses=kernel(duplicate,duplicate)@a
    assert torch.isfinite(a).all();check('duplicate_equal_response',responses,responses[:1].expand_as(responses))
    report['duplicate_responses']=responses.tolist()

if __name__=='__main__':
    try:main();report['status']='passed'
    except Exception as exc:report.update(status='failed',error=repr(exc));raise
    finally:(ROOT/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}));print('checks',len(report['checks']))
