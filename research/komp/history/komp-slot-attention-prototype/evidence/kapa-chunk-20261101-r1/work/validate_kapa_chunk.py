"""Chunk KAPA vs independent FP64 reference, raw geometry and causal boundaries."""
import json
from pathlib import Path
import torch
import validate_kapa as check
from validate_kapa import inputs
from validate_kapa_geometry import raw_inputs,grads
from komp_slot.kapa import protected_kapa
from komp_slot.kapa_chunk import chunk_protected_kapa,chunk_gaussian_protected_kapa
from komp_slot.kapa_reference import protected_kapa_reference,gaussian_features

check.DEST=Path('evidence/kapa-chunk-validation.json')


def main():
    cases=((1,1,1,1,1),(2,37,2,17,33),(1,67,2,32,64),
           (1,65,1,64,128),(1,521,1,8,9),(1,79,1,16,17))
    for dtype in (torch.float32,torch.bfloat16):
        for ci,cfg in enumerate(cases):
            x=inputs(*cfg,dtype,slot=ci%2==1,seed=912+ci)
            if ci in (2,3):
                x[3][:,15]=-1000;x[3][:,32]=-1000
                x[4][:,::3]=0;x[4][:,1::3]=1
            if ci==5:
                x[1][:]=x[1][:,:1].clone()+1e-3*x[1]
                x=[a.to(dtype) if i in (3,4,6) else a for i,a in enumerate(x)]
            if ci==1:x=[a.repeat_interleave(2,dim=-1)[...,::2] for a in x]
            for anchors in (0,2,7):
                ref_inputs=[a.detach().double().requires_grad_() for a in x]
                ref=protected_kapa_reference(*ref_inputs,anchors=anchors)
                weights=[torch.randn_like(a).to(dtype if i==0 else torch.float32) for i,a in enumerate(ref)]
                rg=grads(ref,ref_inputs,[a.double() for a in weights])
                for c in (16,32,64):
                    xx=[a.detach().requires_grad_() for a in x]
                    out=chunk_protected_kapa(*xx,anchors=anchors,chunk_size=c)
                    gg=grads(out,xx,weights)
                    tag=f'{dtype}/{cfg}/p{anchors}/C{c}'
                    for name,a,b in zip(('out','state','dq','dh','dv','dg','dbeta','dmetric','dstate0'),(*out,*gg),(*ref,*rg)):
                        check.compare(tag+'/'+name,a,b,.003 if dtype==torch.float32 else .02)
                    print('PASS',tag,flush=True)
    # Prefix loss with genuinely unused outputs exercises autograd zero adjoints.
    for slot in (False,True):
        for c in (16,32,64):
            x=inputs(1,263,2,8,9,torch.float32,slot=slot,seed=67)
            with torch.no_grad():
                full=chunk_protected_kapa(*x,anchors=7,chunk_size=c)
                serial=protected_kapa(*x,anchors=7)
                for i,(a,b) in enumerate(zip(full,serial)):check.compare(f'serial/{slot}/{c}/{i}',a,b,.0005)
                for cut in (1,7,c-1,c,c+1,2*c+3,262):
                    for i in range(5):
                        y=[a.clone() for a in x]
                        y[i][:,cut:]=(-torch.rand_like(y[i][:,cut:])*3 if i==3 else torch.rand_like(y[i][:,cut:]))
                        out=chunk_protected_kapa(*y,anchors=7,chunk_size=c)[0]
                        err=float((out[:,:cut]-full[0][:,:cut]).abs().max())
                        check.log(dict(name=f'causal/{slot}/C{c}/cut{cut}/{i}',max_abs=err,passed=err==0))
                    prefix=chunk_protected_kapa(*[a[:,:cut] for a in x[:5]],*x[5:],anchors=7,chunk_size=c)[0]
                    check.compare(f'truncated/{slot}/C{c}/cut{cut}',prefix,full[0][:,:cut],.0005)
            for loss in ('prefix','output','state'):
                xx=[a.detach().requires_grad_() for a in x];rr=[a.detach().double().requires_grad_() for a in x]
                out=chunk_protected_kapa(*xx,anchors=7,chunk_size=c);ref=protected_kapa_reference(*rr,anchors=7)
                def objective(o):
                    if loss=='state':return o[1].square().sum()
                    return (o[0][:,:c+3] if loss=='prefix' else o[0]).square().sum()
                gg=torch.autograd.grad(objective(out),xx)
                rg=torch.autograd.grad(objective(ref),rr,allow_unused=True)
                for i,(a,b) in enumerate(zip(gg,rg)):check.compare(f'loss/{slot}/C{c}/{loss}/{i}',a,torch.zeros_like(rr[i]) if b is None else b,.003)
                if loss=='prefix':
                    for i,a in enumerate(gg[:5]):
                        err=float(a[:,c+3:].abs().max())
                        check.log(dict(name=f'future_gradient/{slot}/C{c}/{i}',max_abs=err,passed=err==0))
    for dtype in (torch.float32,torch.bfloat16):
        for ci,(cfg,duplicate,ridge) in enumerate([
                ((2,37,2,7,5,9),None,.01),((1,67,1,128,64,128),None,.01),
                ((1,75,1,16,32,17),0.,.01),((1,79,1,31,17,33),1e-3,.001)]):
            x=raw_inputs(*cfg,dtype,slot=ci%2==1,duplicate=duplicate,noncontiguous=ci==0)
            for c in (16,32,64):
                xx=[a.detach().requires_grad_() for a in x];rr=[a.detach().double().requires_grad_() for a in x]
                out=chunk_gaussian_protected_kapa(*xx,gram_ridge=ridge,anchors=4,chunk_size=c)
                q,h,p=gaussian_features(rr[0],rr[1],rr[5],gram_ridge=ridge)
                ref=protected_kapa_reference(q,h,rr[2],rr[3],rr[4],p,rr[6],anchors=4)
                weights=[torch.randn_like(a) for a in out]
                gg=grads(out,xx,weights);rg=grads(ref,rr,[a.double() for a in weights])
                for name,a,b in zip(('out','state','dq','dk','dv','dg','dbeta','dcenters','dstate0'),(*out,*gg),(*ref,*rg)):
                    check.compare(f'raw/{dtype}/{cfg}/dup{duplicate}/C{c}/{name}',a,b,.01 if dtype==torch.float32 else .03)
                print('PASS raw',dtype,cfg,c,flush=True)


if __name__=='__main__':
    torch.manual_seed(127);torch.set_num_threads(4);check.DEST.parent.mkdir(exist_ok=True,parents=True)
    try:main();check.REPORT['status']='passed'
    except Exception as exc:check.REPORT['status']='failed';check.REPORT['error']=repr(exc);raise
    finally:check.DEST.write_text(json.dumps(check.REPORT,indent=2))
