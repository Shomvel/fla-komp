"""KAPA numerical, proximal, protection, degeneracy and causality acceptance."""
import json
from pathlib import Path
import torch
from komp_slot.kapa import protected_kapa,gaussian_protected_kapa
from komp_slot.kapa_reference import protected_kapa_reference,protected_direction,gaussian_features,replay_kapa_reference

DEST=Path('evidence/kapa-validation.json');REPORT={'checks':[]}


def log(row):
    REPORT['checks'].append(row);DEST.write_text(json.dumps(REPORT,indent=2))
    assert row['passed'],row


def compare(name,a,b,tol=.003):
    delta=a.detach().double()-b.detach().double()
    rel=float(delta.square().mean().sqrt()/(b.detach().double().square().mean().sqrt()+1e-30))
    absolute=float(delta.abs().max())
    log(dict(name=name,rms_relative=rel,max_abs=absolute,tolerance=tol,passed=bool(torch.isfinite(a).all()) and (rel<tol or absolute<2e-6)))


def inputs(b,t,h,m,v,dtype,slot=False,seed=317):
    torch.manual_seed(seed)
    rand=lambda *s:torch.randn(*s,device='cuda')
    q=torch.nn.functional.normalize(rand(b,t,h,m),dim=-1).to(dtype)
    k=torch.nn.functional.normalize(rand(b,t,h,m),dim=-1).to(dtype)
    value=(rand(b,t,h,v)*.3).to(dtype)
    g=-torch.rand((b,t,h,m) if slot else (b,t,h),device='cuda')*.2
    beta=torch.rand(b,t,h,device='cuda')*.7
    factor=rand(b,h,m,m)/m**.5
    p=factor@factor.transpose(-1,-2)+torch.eye(m,device='cuda')*.3
    initial=rand(b,h,m,v)*.1
    return [q,k,value,g,beta,p,initial]


def main():
    for dtype in (torch.float32,torch.bfloat16):
        for ci,cfg in enumerate(((1,1,1,5,7),(2,19,2,16,24),(1,67,1,17,33),(1,37,1,64,128))):
            x=inputs(*cfg,dtype,slot=ci%2==1,seed=44+ci)
            if ci==2:
                x[3][:,3]=-1000;x[4][:,::3]=0;x[4][:,1::3]=1
            for anchors in (0,2,4,7):
                a=[z.detach().clone().requires_grad_() for z in x]
                b=[z.detach().double().requires_grad_() for z in x]
                got=protected_kapa(*a,anchors=anchors);ref=protected_kapa_reference(*b,anchors=anchors)
                ws=[torch.randn_like(z) for z in got]
                ga=torch.autograd.grad(got,a,ws);gb=torch.autograd.grad(ref,b,[z.double() for z in ws])
                tag=f'{dtype}/{cfg}/anchors{anchors}'
                for name,u,v in zip(('out','state','dq','dh','dv','dg','dbeta','dmetric','dstate0'),(*got,*ga),(*ref,*gb)):
                    compare(tag+'/'+name,u,v,.003 if dtype==torch.float32 else .02)
                print('PASS',tag,flush=True)
    # p=0 is normalized kernel delta; this uses its direct formula independently.
    x=inputs(1,13,2,8,4,torch.float32,slot=True)
    h=x[1].double();p=x[5].double();raw=torch.einsum('bhmn,bthn->bthm',p,h)
    normalized=raw/((h*raw).sum(-1,keepdim=True)+.05)
    compare('p0_normalized_delta',protected_direction(h,p,0,.05),normalized,1e-11)
    # Independent primal proximal linear system equals multi-constraint KAPA.
    torch.manual_seed(66)
    m,r,v=8,3,4
    a=torch.randn(m,m,dtype=torch.float64);metric=a@a.T+torch.eye(m,dtype=torch.float64)
    inv=torch.linalg.inv(metric);basis=torch.randn(m,r,dtype=torch.float64)
    old=torch.randn(m,v,dtype=torch.float64);target=torch.randn(r,v,dtype=torch.float64);lam=.2
    update=old+inv@basis@torch.linalg.solve(basis.T@inv@basis+lam*torch.eye(r,dtype=torch.float64),target-basis.T@old)
    direct=torch.linalg.solve(metric+basis@basis.T/lam,metric@old+basis@target/lam)
    compare('kapa_proximal_normal_equations',update,direct,1e-11)
    # Hard protection is feasible for independent directions, but not identical
    # conflicting keys. The implemented regularized rule deliberately compromises.
    hnew=torch.tensor([1.,.2],dtype=torch.float64);anchor=torch.tensor([1.,0.],dtype=torch.float64)
    basis=torch.stack((hnew,anchor),1);p=torch.eye(2,dtype=torch.float64)
    w=p@basis@torch.linalg.solve(basis.T@p@basis,torch.tensor([1.,0.],dtype=torch.float64))
    compare('hard_feasible_constraints',basis.T@w,torch.tensor([1.,0.],dtype=torch.float64),1e-11)
    identical=torch.stack((anchor,anchor),1)
    soft=identical@torch.linalg.solve(identical.T@identical+.05*torch.eye(2,dtype=torch.float64),torch.tensor([1.,0.],dtype=torch.float64))
    responses=identical.T@soft
    log(dict(name='identical_conflict_is_not_exact_protection',responses=responses.tolist(),passed=bool(torch.isfinite(soft).all()) and abs(float(responses[0]-responses[1]))<1e-12 and abs(float(responses[1]))>.1))
    # Actual GPU duplicate/near-duplicate histories, nonzero starting coefficients.
    for noise in (0.,1e-3):
        x=inputs(1,41,2,16,12,torch.float32,slot=True)
        x[1][:]=x[1][:,:1].clone()+noise*x[1]
        a=[z.detach().clone().requires_grad_() for z in x];b=[z.detach().double().requires_grad_() for z in x]
        got=protected_kapa(*a,anchors=4,regularization=.01);ref=protected_kapa_reference(*b,anchors=4,regularization=.01)
        ws=[torch.randn_like(z) for z in got]
        ga=torch.autograd.grad(got,a,ws);gb=torch.autograd.grad(ref,b,[z.double() for z in ws])
        for i,(u,v) in enumerate(zip((*got,*ga),(*ref,*gb))):compare(f'duplicate/{noise}/{i}',u,v,.005)
    # Causality includes past-key windows and the geometry solve adjoint.
    for slot in (False,True):
        x=inputs(1,67,2,8,6,torch.float32,slot=slot)
        with torch.no_grad():
            full=protected_kapa(*x,anchors=4)[0]
            for cut in (1,3,4,5,16,33,66):
                for component in range(5):
                    y=[z.clone() for z in x]
                    y[component][:,cut:]=(-torch.rand_like(y[component][:,cut:]) if component==3 else torch.rand_like(y[component][:,cut:]))
                    out=protected_kapa(*y,anchors=4)[0]
                    err=float((out[:,:cut]-full[:,:cut]).abs().max())
                    log(dict(name=f'causality/{slot}/{cut}/{component}',max_abs=err,passed=err==0))
        for loss in ('prefix','output','state'):
            a=[z.detach().clone().requires_grad_() for z in x];b=[z.detach().double().requires_grad_() for z in x]
            got=protected_kapa(*a,anchors=4);ref=protected_kapa_reference(*b,anchors=4)
            objective=lambda o:o[1].square().sum() if loss=='state' else (o[0][:,:19] if loss=='prefix' else o[0]).square().sum()
            ga=torch.autograd.grad(objective(got),a);gb=torch.autograd.grad(objective(ref),b,allow_unused=True)
            for i,(u,v) in enumerate(zip(ga,gb)):compare(f'loss/{slot}/{loss}/{i}',u,torch.zeros_like(b[i]) if v is None else v)
            if loss=='prefix':
                for i,u in enumerate(ga[:5]):
                    err=float(u[:,19:].abs().max());log(dict(name=f'future_gradient/{slot}/{i}',max_abs=err,passed=err==0))
    # End-to-end trainable raw-key Gaussian adapter, including dictionary/Gram grads.
    torch.manual_seed(311)
    rand=lambda *s:torch.randn(*s,device='cuda')
    raw=[rand(1,17,2,8),rand(1,17,2,8),rand(1,17,2,5)*.2,-torch.rand(1,17,2,device='cuda')*.1,
         torch.rand(1,17,2,device='cuda')*.7,rand(1,2,6,8),rand(1,2,6,5)*.1]
    a=[z.requires_grad_() for z in raw];b=[z.detach().double().requires_grad_() for z in raw]
    got=gaussian_protected_kapa(*a,anchors=2)
    qf,h,p=gaussian_features(b[0],b[1],b[5]);ref=protected_kapa_reference(qf,h,b[2],b[3],b[4],p,b[6],anchors=2)
    ws=[torch.randn_like(z) for z in got]
    ga=torch.autograd.grad(got,a,ws);gb=torch.autograd.grad(ref,b,[z.double() for z in ws])
    for i,(u,v) in enumerate(zip((*got,*ga),(*ref,*gb))):compare(f'gaussian_adapter/{i}',u,v,.003)


if __name__=='__main__':
    DEST.parent.mkdir(exist_ok=True,parents=True);torch.set_num_threads(4)
    try:main();REPORT['status']='passed'
    except Exception as exc:REPORT['status']='failed';REPORT['error']=repr(exc);raise
    finally:DEST.write_text(json.dumps(REPORT,indent=2))
