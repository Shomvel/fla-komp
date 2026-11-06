"""Independent FP64 acceptance for chunk BSGD merging and its entire backward."""
import json
from pathlib import Path
import torch
from komp_slot.bsgd_chunk import chunk_bsgd_merging
from komp_slot.bsgd_reference import bsgd_reference
from validate_bsgd import inputs,run

DEST=Path('evidence/bsgd-chunk-validation.json')
REPORT={'checks':[]}


def log(row):
    REPORT['checks'].append(row)
    DEST.write_text(json.dumps(REPORT,indent=2))
    assert row['passed'],row


def compare(name,a,b,tol):
    delta=a.double()-b.double()
    rms=float(delta.square().mean().sqrt()/(b.double().square().mean().sqrt()+1e-30))
    absolute=float(delta.abs().max())
    log(dict(name=name,rms_relative=rms,max_abs=absolute,tolerance=tol,
             passed=bool(torch.isfinite(a).all()) and (rms<=tol or absolute<=2e-6)))


def main():
    cases=[(1,1,1,7,5,3),(2,37,2,16,24,8),(1,75,2,33,17,5),
           (1,130,1,64,64,32),(1,65,1,128,128,64),(1,521,1,16,8,4)]
    for dtype in (torch.float32,torch.bfloat16):
        for ci,cfg in enumerate(cases):
            x=inputs(*cfg,dtype,slot=ci%2==1,seed=237+ci)
            if ci==2:
                x[3][:,2]=-1000; x[4][:,::3]=0; x[4][:,1::3]=1
                x[5][:,::3]=0; x[5][:,1::3]=1
            rr=[a.detach().double().requires_grad_() for a in x]
            ref=run(bsgd_reference,rr,offset=2)
            weights=[torch.randn_like(a).float() for a in ref]
            gr=torch.autograd.grad(ref,rr,[w.double() for w in weights])
            for c in (16,32,64):
                xx=[a.detach().clone().requires_grad_() for a in x]
                got=run(chunk_bsgd_merging,xx,chunk_size=c,offset=2)
                gg=torch.autograd.grad(got,xx,[w.to(a.dtype) for w,a in zip(weights,got)])
                # Use the same BF16-rounded output upstream gradient in both graphs.
                if dtype==torch.bfloat16:
                    rr2=[a.detach().double().requires_grad_() for a in x]
                    ref2=run(bsgd_reference,rr2,offset=2)
                    gr2=torch.autograd.grad(ref2,rr2,[w.to(a.dtype).double() for w,a in zip(weights,got)])
                else:
                    gr2=gr
                tag=f'{dtype}/{cfg}/C{c}'
                for name,a,b in zip(('out','keys','coeff','dq','dk','dv','dg','dbeta','dmerge','dk0','dc0'),(*got,*gg),(*ref,*gr2)):
                    compare(tag+'/'+name,a,b,.0005 if dtype==torch.float32 else .015)
                print('PASS reference',tag,flush=True)
    # Intervene on every input independently, across intra- and inter-chunk cuts.
    for slot in (False,True):
        for c in (16,32,64):
            x=inputs(1,263,2,32,24,7,torch.float32,slot=slot,seed=992)
            with torch.no_grad():
                full=run(chunk_bsgd_merging,x,chunk_size=c,offset=3)
                for cut in (1,c-1,c,c+1,2*c+3,262):
                    for component in range(6):
                        y=[a.clone() for a in x]
                        y[component][:,cut:]=(-torch.rand_like(y[component][:,cut:])*4 if component==3 else
                                             torch.rand_like(y[component][:,cut:])*2)
                        got=run(chunk_bsgd_merging,y,chunk_size=c,offset=3)
                        err=float((got[0][:,:cut]-full[0][:,:cut]).abs().max())
                        log(dict(name=f'causal/{slot}/C{c}/{cut}/{component}',max_abs=err,passed=err==0))
                for cut in (13,c,2*c+3):
                    first=run(chunk_bsgd_merging,[a[:,:cut] for a in x[:6]]+x[6:],chunk_size=c,offset=3)
                    second=run(chunk_bsgd_merging,[a[:,cut:] for a in x[:6]]+list(first[1:]),chunk_size=c,offset=cut+3)
                    for name,a,b in zip(('out','keys','coeff'),(torch.cat((first[0],second[0]),1),*second[1:]),full):
                        compare(f'stream/{slot}/C{c}/{cut}/{name}',a,b,1e-4)
            for loss in ('prefix','output','state'):
                xx=[a.detach().clone().requires_grad_() for a in x]
                rr=[a.detach().double().requires_grad_() for a in x]
                got=run(chunk_bsgd_merging,xx,chunk_size=c,offset=3)
                ref=run(bsgd_reference,rr,offset=3)
                def objective(o):
                    if loss=='prefix':return o[0][:,:c+3].square().sum()
                    if loss=='output':return o[0].square().sum()
                    return o[1].square().sum()+o[2].square().sum()
                gg=torch.autograd.grad(objective(got),xx)
                gr=torch.autograd.grad(objective(ref),rr,allow_unused=True)
                for i,(a,b) in enumerate(zip(gg,gr)):
                    compare(f'loss/{slot}/C{c}/{loss}/{i}',a,torch.zeros_like(rr[i]) if b is None else b,.0005)
                if loss=='prefix':
                    for i,a in enumerate(gg[:6]):
                        err=float(a[:,c+3:].abs().max())
                        log(dict(name=f'future_grad/{slot}/C{c}/{i}',max_abs=err,passed=err<=1e-7))
    # Identical keys and endpoint gates, including slots not yet visited when T<M.
    x=inputs(1,3,1,16,16,8,torch.float32,slot=True,seed=62)
    x[1][:]=x[6][:,:,0,:][:,None]
    x[6][:]=x[6][:,:,0:1,:].clone()
    for gamma in (0.,1.):
        x[5].fill_(gamma)
        xx=[a.clone().requires_grad_() for a in x]
        rr=[a.double().requires_grad_() for a in x]
        got=run(chunk_bsgd_merging,xx,offset=6);ref=run(bsgd_reference,rr,offset=6)
        gg=torch.autograd.grad(sum(a.square().sum() for a in got),xx)
        gr=torch.autograd.grad(sum(a.square().sum() for a in ref),rr)
        for i,(a,b) in enumerate(zip((*got,*gg),(*ref,*gr))):
            compare(f'duplicate/merge{gamma}/{i}',a,b,.0005)


if __name__=='__main__':
    DEST.parent.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4)
    try:
        main();REPORT['status']='passed'
    except Exception as exc:
        REPORT['status']='failed';REPORT['error']=repr(exc);raise
    finally:
        DEST.write_text(json.dumps(REPORT,indent=2))
