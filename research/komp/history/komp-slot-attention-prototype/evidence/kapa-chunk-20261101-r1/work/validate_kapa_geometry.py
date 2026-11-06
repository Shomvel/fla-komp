"""Raw Gaussian KAPA acceptance against independent FP64 PyTorch autograd."""
import json
from pathlib import Path
import torch
import validate_kapa as check
from komp_slot.kapa_geometry import gaussian_features
from komp_slot.kapa_reference import gaussian_features as reference_features, protected_kapa_reference
from komp_slot.kapa import gaussian_protected_kapa

check.DEST=Path('evidence/kapa-geometry-validation.json')


def raw_inputs(b,t,h,d,m,v,dtype,slot=False,duplicate=None,noncontiguous=False):
    rand=lambda *s:torch.randn(*s,device='cuda')
    q,k=rand(b,t,h,d)/d**.5,rand(b,t,h,d)/d**.5
    u=rand(b,h,m,d)/d**.5
    if duplicate is not None:u=u[:,:,:1].expand_as(u).clone()+duplicate*u
    xs=[q,k,rand(b,t,h,v)*.2,-torch.rand((b,t,h,m) if slot else (b,t,h),device='cuda')*.1,
        torch.rand(b,t,h,device='cuda')*.7,u,rand(b,h,m,v)*.1]
    xs=[x.to(dtype) if i in (0,1,2,5) else x for i,x in enumerate(xs)]
    if noncontiguous:
        xs=[x.repeat_interleave(2,dim=-1)[...,::2] for x in xs]
    return xs


def grads(outputs,inputs,weights):
    gs=torch.autograd.grad(outputs,inputs,weights,allow_unused=True)
    return [torch.zeros_like(x) if g is None else g for x,g in zip(inputs,gs)]


def main():
    torch.manual_seed(728)
    for dtype in (torch.float32,torch.bfloat16):
        for ci,(cfg,duplicate,ridge) in enumerate([
                ((1,1,1,1,1,1),None,.01),((2,19,2,7,5,9),None,.01),
                ((1,33,2,128,64,17),None,.01),((1,17,1,31,17,7),0.,.01),
                ((1,21,1,16,32,8),1e-3,.001)]):
            raw=raw_inputs(*cfg,dtype,slot=ci%2==0,duplicate=duplicate,noncontiguous=ci==1)
            tag=f'{dtype}/{cfg}/dup{duplicate}/ridge{ridge}'
            tol=.003 if dtype==torch.float32 else .02
            if ridge==.001:tol=.01 if dtype==torch.float32 else .03
            for loss in ('all','qf','hf','inverse'):
                a=[raw[i].detach().requires_grad_() for i in (0,1,5)]
                b=[x.detach().double().requires_grad_() for x in a]
                out=gaussian_features(*a,gram_ridge=ridge);ref=reference_features(*b,gram_ridge=ridge)
                weights=[torch.randn_like(x) if loss=='all' or i=={'qf':0,'hf':1,'inverse':2}.get(loss) else torch.zeros_like(x) for i,x in enumerate(out)]
                ga=grads(out,a,weights);gb=grads(ref,b,[x.double() for x in weights])
                for name,x,y in zip(('qf','hf','inverse','dq','dk','dcenters'),(*out,*ga),(*ref,*gb)):
                    check.compare(tag+'/'+loss+'/'+name,x,y,tol)
                # Verify inverse against a separately constructed Gram residual.
                if loss=='all':
                    u=b[2].detach();gram=torch.exp(-(u[..., :,None,:]-u[...,None,:,:]).square().sum(-1))
                    eye=torch.eye(cfg[4],device='cuda',dtype=torch.float64)
                    residual=(gram+ridge*eye)@out[2].double()-eye
                    check.log(dict(name=tag+'/inverse_residual',max_abs=float(residual.abs().max()),passed=float(residual.abs().max())<.003))
            for loss in ('both','output','state','prefix'):
                a=[x.detach().requires_grad_() for x in raw];b=[x.detach().double().requires_grad_() for x in raw]
                out=gaussian_protected_kapa(*a,gram_ridge=ridge)
                qf,hf,p=reference_features(b[0],b[1],b[5],gram_ridge=ridge)
                ref=protected_kapa_reference(qf,hf,b[2],b[3],b[4],p,b[6])
                w=[torch.randn_like(x) for x in out]
                cut=max(1,cfg[1]//2)
                if loss=='state':w[0].zero_()
                if loss in ('output','prefix'):w[1].zero_()
                if loss=='prefix':w[0][:,cut:]=0
                ga=grads(out,a,w);gb=grads(ref,b,[x.double() for x in w])
                for name,x,y in zip(('output','state','dq','dk','dv','dg','dbeta','dcenters','dstate0'),(*out,*ga),(*ref,*gb)):
                    check.compare(tag+'/raw/'+loss+'/'+name,x,y,tol)
                if loss=='prefix' and cfg[1]>1:
                    for i,x in enumerate(ga[:5]):
                        err=float(x[:,cut:].abs().max())
                        check.log(dict(name=tag+f'/future_gradient/{i}',max_abs=err,passed=err==0))
            if cfg[1]>1:
                with torch.no_grad():
                    baseline=gaussian_protected_kapa(*raw,gram_ridge=ridge)[0]
                    for i in range(5):
                        changed=[x.clone() for x in raw];changed[i][:,cut:]+= .1
                        actual=gaussian_protected_kapa(*changed,gram_ridge=ridge)[0]
                        err=float((baseline[:,:cut]-actual[:,:cut]).abs().max())
                        check.log(dict(name=tag+f'/future_intervention/{i}',max_abs=err,passed=err==0))
            print('PASS',tag,flush=True)


if __name__=='__main__':
    check.DEST.parent.mkdir(exist_ok=True,parents=True);torch.set_num_threads(4)
    try:main();check.REPORT['status']='passed'
    except Exception as exc:check.REPORT['status']='failed';check.REPORT['error']=repr(exc);raise
    finally:check.DEST.write_text(json.dumps(check.REPORT,indent=2))
