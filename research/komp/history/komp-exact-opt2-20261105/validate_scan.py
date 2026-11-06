"""Hierarchical boundary/tail/causality acceptance against independent affine scans."""
import json
from pathlib import Path
import torch
from exact_komp.hierarchical import scan
from exact_komp import exact_komp
from baseline_komp import exact_komp as baseline
from validate_gpu import make,check,checks


def affine():
    torch.manual_seed(521)
    for nc,m,v in [(32,7,11),(33,32,64),(63,64,128),(129,32,64)]:
        f=.96*torch.eye(m,device='cuda')[None,None]+.01*torch.randn(2,nc,m,m,device='cuda')/m**.5
        u=.05*torch.randn(2,nc,m,v,device='cuda');initial=torch.randn(2,m,v,device='cuda')
        for reverse in (False,True):
            entry=torch.empty_like(u);final=torch.empty_like(initial)
            scan(f,u,initial,entry,final,reverse=reverse)
            state=initial.double();ref=torch.empty_like(u,dtype=torch.float64)
            for i in (range(nc-1,-1,-1) if reverse else range(nc)):
                ref[:,i]=state
                mat=f[:,i].double()
                state=(mat.transpose(-1,-2) if reverse else mat)@state+u[:,i].double()
            check(f'affine/nc{nc}/m{m}/v{v}/rev{reverse}/entry',entry,ref,1e-4,1e-4)
            check(f'affine/nc{nc}/m{m}/v{v}/rev{reverse}/final',final,state,1e-4,1e-4)


def end_to_end():
    for learned in (False,True):
        for t,chunk,scalar in [(513,16,False),(1025,32,True)]:
            xs=tuple(x.requires_grad_(i<8) for i,x in enumerate(make(522,2,t,2,7,9,11,torch.float32,scalar=scalar)))
            q,k,v,g,beta,k0,s0,scores,init=xs
            kw=dict(eviction='learned' if learned else 'fifo',scores=scores,initial_scores=init,compute_chunk=chunk)
            def run(fn):
                out,(kf,sf),_=fn(q,k,v,g,beta,(k0,s0),**kw)
                return out,kf,sf
            a,b=run(exact_komp),run(baseline)
            weights=[torch.randn_like(x) for x in a]
            ga=torch.autograd.grad(a,xs[:8],weights,allow_unused=True)
            gb=torch.autograd.grad(b,xs[:8],weights,allow_unused=True)
            for i,(x,y) in enumerate(zip((*a,*ga),(*b,*gb))):
                if x is not None and y is not None:check(f'tail/t{t}/c{chunk}/l{learned}/{i}',x,y,1e-4,1e-4)
        with torch.no_grad():
            xs=make(523,1,8192,2,32,64,64,torch.float32,scalar=True)
            def run(data):
                q,k,v,g,beta,k0,s0,score,init=data
                return exact_komp(q,k,v,g,beta,(k0,s0),eviction='learned' if learned else 'fifo',scores=score,initial_scores=init)[0]
            a=run(xs);changed=[x.clone() for x in xs]
            for i in (0,1,2,3,4,7):changed[i][:,5003:]+=.01
            b=run(changed)
            check(f'8k/prefix/l{learned}',a[:,:5003],b[:,:5003],0.,0.)


try:
    affine();end_to_end()
    Path('evidence/scan.json').write_text(json.dumps({'status':'passed','check_count':len(checks),'checks':checks},indent=2))
    print('PASS hierarchical scan',len(checks),flush=True)
except Exception as e:
    Path('evidence/scan-failure.json').write_text(json.dumps({'error':repr(e),'checks':checks},indent=2))
    raise
