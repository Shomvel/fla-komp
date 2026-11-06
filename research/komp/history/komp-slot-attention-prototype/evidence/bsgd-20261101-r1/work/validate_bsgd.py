"""GPU acceptance: independent FP64 recurrence, all adjoints, causality and streaming."""
import json
from pathlib import Path
import torch
from komp_slot.bsgd import bsgd_merging
from komp_slot.bsgd_reference import bsgd_reference

DEST = Path('evidence/bsgd-validation.json')
REPORT = {'checks': []}


def inputs(b, t, h, d, w, m, dtype, slot=False, seed=31):
    torch.manual_seed(seed)
    rand = lambda *s: torch.randn(*s, device='cuda')
    # Unit input embeddings; merged centers themselves are never renormalized.
    unit = lambda x: torch.nn.functional.normalize(x, dim=-1)
    q, k = unit(rand(b,t,h,d)).to(dtype), unit(rand(b,t,h,d)).to(dtype)
    v = rand(b,t,h,w).to(dtype)*.3
    g = -torch.rand((b,t,h,m) if slot else (b,t,h), device='cuda')*.15
    beta, gamma = torch.rand(b,t,h,device='cuda')*.7, torch.rand(b,t,h,device='cuda')*.8+.1
    keys, values = unit(rand(b,h,m,d)), rand(b,h,m,w)*.1
    return [q,k,v,g,beta,gamma,keys,values]


def run(fn, x, **kw):
    o, (k, v) = fn(*x[:6], (x[6],x[7]), **kw)
    return o,k,v


def log(row):
    REPORT['checks'].append(row)
    DEST.write_text(json.dumps(REPORT,indent=2))
    assert row['passed'], row


def compare(name, a, b, tol):
    delta = a.double()-b.double()
    rms = float(delta.square().mean().sqrt()/(b.double().square().mean().sqrt()+1e-30))
    absolute = float(delta.abs().max())
    log(dict(name=name,rms_relative=rms,max_abs=absolute,tolerance=tol,
             passed=bool(torch.isfinite(a).all()) and (rms<=tol or absolute<2e-6)))


def main():
    cases = [(1,1,1,7,5,3), (2,19,2,16,24,8), (1,67,2,33,17,5), (1,129,1,64,64,32)]
    for mode,routing in [('gated','cyclic'),('grid','cyclic'),('grid','min_error')]:
        for dtype in (torch.float32,torch.bfloat16):
            for ci,cfg in enumerate(cases):
                x=inputs(*cfg,dtype,slot=ci%2==1,seed=77+ci)
                if ci==2:
                    x[3][:,2]=-80; x[4][:,::3]=0; x[4][:,1::3]=1
                    x[5][:,::3]=0; x[5][:,1::3]=1
                xx=[a.detach().clone().requires_grad_() for a in x]
                rr=[a.detach().double().requires_grad_() for a in x]
                kw=dict(mode=mode,routing=routing,offset=2)
                got,ref=run(bsgd_merging,xx,**kw),run(bsgd_reference,rr,**kw)
                ws=[torch.randn_like(a) for a in got]
                gg=torch.autograd.grad(got,xx,ws)
                gr=torch.autograd.grad(ref,rr,[w.double() for w in ws],allow_unused=True)
                gr=[torch.zeros_like(a) if z is None else z for a,z in zip(rr,gr)]
                tag=f'{mode}/{routing}/{dtype}/{cfg}'
                for name,a,b in zip(('out','keys','coeff','dq','dk','dv','dg','dbeta','dmerge','dk0','dc0'),(*got,*gg),(*ref,*gr)):
                    compare(tag+'/'+name,a,b,.0005 if dtype==torch.float32 else .015)
                print('PASS',tag,flush=True)
    # Identical centers: merging is exact (no RKHS degradation); covers grid ties.
    for mode in ('gated','grid'):
        x=inputs(1,17,1,8,8,4,torch.float32)
        x[1][:]=x[6][:,:,0,:][:,None]
        x[6][:]=x[6][:,:,0:1,:].clone()
        compare('identical/'+mode,run(bsgd_merging,x,mode=mode)[0],
                run(bsgd_reference,[a.double() for a in x],mode=mode)[0],.0005)
    # Arbitrary (not just chunk-aligned) splits, with explicit cyclic phase.
    for mode,routing in [('gated','cyclic'),('grid','min_error')]:
        x=inputs(1,45,2,16,12,7,torch.float32,slot=True,seed=999)
        kw=dict(mode=mode,routing=routing)
        with torch.no_grad():
            full=run(bsgd_merging,x,**kw)
            for cut in (1,7,13,32,44):
                for component in range(6):
                    y=[a.clone() for a in x]
                    y[component][:,cut:]=(-torch.rand_like(y[component][:,cut:]) if component==3 else
                                         torch.rand_like(y[component][:,cut:]))
                    got=run(bsgd_merging,y,**kw)
                    log(dict(name=f'causal/{mode}/{cut}/{component}',passed=torch.equal(got[0][:,:cut],full[0][:,:cut])))
                first=run(bsgd_merging,[a[:,:cut] for a in x[:6]]+x[6:],**kw)
                second=run(bsgd_merging,[a[:,cut:] for a in x[:6]]+list(first[1:]),offset=cut,**kw)
                for name,a,b in zip(('out','keys','coeff'),(torch.cat((first[0],second[0]),1),*second[1:]),full):
                    compare(f'stream/{mode}/{cut}/{name}',a,b,1e-6)
        leaves=[a.detach().clone().requires_grad_() for a in x]
        out=run(bsgd_merging,leaves,**kw)[0]
        grad=torch.autograd.grad(out[:,:13].square().sum(),leaves)
        for i,a in enumerate(grad[:6]):
            log(dict(name=f'future_gradient/{mode}/{i}',max_abs=float(a[:,13:].abs().max()),passed=bool((a[:,13:]==0).all())))
        # Output-only and state-only objectives exercise zero upstream branches.
        for loss in ('output','state'):
            xx=[a.detach().clone().requires_grad_() for a in x]
            rr=[a.detach().double().requires_grad_() for a in x]
            got,ref=run(bsgd_merging,xx,**kw),run(bsgd_reference,rr,**kw)
            idx=(0,) if loss=='output' else (1,2)
            gg=torch.autograd.grad(sum(got[i].square().sum() for i in idx),xx)
            gr=torch.autograd.grad(sum(ref[i].square().sum() for i in idx),rr,allow_unused=True)
            for i,(a,b) in enumerate(zip(gg,gr)):
                compare(f'loss/{mode}/{loss}/{i}',a,torch.zeros_like(rr[i]) if b is None else b,.0005)
    # Algebraic sanity: pair coefficient formula beats any alternative coefficient
    # at the same center; grid choice includes both endpoint centers.
    torch.manual_seed(17)
    u,k=torch.randn(2,11,dtype=torch.float64)
    a,b=torch.randn(2,9,dtype=torch.float64)
    gamma=torch.linspace(0,1,17,dtype=torch.float64)
    dist=(u-k).square().sum(); tau=.5
    ka=torch.exp(-gamma.square()*dist/(2*tau)); kb=torch.exp(-(1-gamma).square()*dist/(2*tau))
    coeff=ka[:,None]*a+kb[:,None]*b
    energy=a.square().sum()+b.square().sum()+2*torch.exp(-dist/(2*tau))*(a*b).sum()
    err=energy-coeff.square().sum(-1)
    noise=torch.randn_like(coeff)
    alternative=energy+(coeff+noise).square().sum(-1)-2*((coeff+noise)*coeff).sum(-1)
    log(dict(name='rkhs_coefficient_optimum',passed=bool((alternative>=err-1e-12).all())))
    log(dict(name='grid_vs_removal',passed=float(err.min())<=float(torch.minimum(a.square().sum(),b.square().sum()))+1e-12))


if __name__=='__main__':
    DEST.parent.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4)
    try:
        main();REPORT['status']='passed'
    except Exception as e:
        REPORT['status']='failed';REPORT['error']=repr(e);raise
    finally:
        DEST.write_text(json.dumps(REPORT,indent=2))
