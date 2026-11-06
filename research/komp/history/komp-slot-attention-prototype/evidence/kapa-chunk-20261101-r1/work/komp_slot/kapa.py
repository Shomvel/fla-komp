"""Feature-level soft-protected KAPA with Triton fwd and first bwd.

All direction solves, recurrence and gradients are Triton. The raw-key adapter
uses kapa_geometry.py for Triton Gaussian features and inverse Gram gradients.
Fixed dictionary only, anchors are past feature vectors.
"""
import math
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable
from .geometry_chunk import _sweep_inverse
from .utils import zero


@tr.jit
def _basis(Hist,t,bh,T:tl.constexpr,H:tl.constexpr,M:tl.constexpr,BM:tl.constexpr,R:tl.constexpr):
    m,r=tl.arange(0,BM),tl.arange(0,16)
    time=t-r
    tok=((bh//H)*T+time)*H+bh%H
    return tl.load(Hist+tok[None,:]*M+m[:,None],(m[:,None]<M)&(r[None,:]<R)&(time[None,:]>=0),0).to(tl.float32)


@tr.jit
def direction_fwd(Hist,Metric,Write,Inv,Coeff,T:tl.constexpr,H:tl.constexpr,M:tl.constexpr,
                   BM:tl.constexpr,R:tl.constexpr,LAM:tl.constexpr):
    t,bh=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64)
    m,r=tl.arange(0,BM),tl.arange(0,16)
    mask=(m[:,None]<M)&(m[None,:]<M)
    p=tl.load(Metric+bh*M*M+m[:,None]*M+m[None,:],mask,0).to(tl.float32)
    p=.5*(p+tl.trans(p))
    basis=_basis(Hist,t,bh,T,H,M,BM,R)
    x=tl.dot(p,basis,input_precision='tf32x3')
    gram=tl.dot(tl.trans(basis),x,input_precision='tf32x3')
    gram+=tl.where(r[:,None]==r[None,:],LAM,0.)
    inv=_sweep_inverse(gram,R,16)
    c=tl.sum(tl.where(r[None,:]==0,inv,0.),1)
    w=tl.sum(x*c[None,:],1)
    tok=((bh//H)*T+t)*H+bh%H
    tl.store(Write+tok*M+m,w,m<M)
    ix=bh*T+t
    tl.store(Inv+ix*R*R+r[:,None]*R+r[None,:],inv,(r[:,None]<R)&(r[None,:]<R))
    tl.store(Coeff+ix*R+r,c,r<R)


@tr.jit
def direction_bwd(Hist,Metric,Inv,Coeff,DWrite,DHist,DMetric,T:tl.constexpr,H:tl.constexpr,
                   M:tl.constexpr,BM:tl.constexpr,R:tl.constexpr):
    t,bh=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64)
    m,r=tl.arange(0,BM),tl.arange(0,16)
    mask=(m[:,None]<M)&(m[None,:]<M)
    p=tl.load(Metric+bh*M*M+m[:,None]*M+m[None,:],mask,0).to(tl.float32)
    p=.5*(p+tl.trans(p))
    basis=_basis(Hist,t,bh,T,H,M,BM,R)
    x=tl.dot(p,basis,input_precision='tf32x3')
    ix=bh*T+t;tok=((bh//H)*T+t)*H+bh%H
    inv=tl.load(Inv+ix*R*R+r[:,None]*R+r[None,:],(r[:,None]<R)&(r[None,:]<R),0)
    c=tl.load(Coeff+ix*R+r,r<R,0)
    dw=tl.load(DWrite+tok*M+m,m<M,0)
    dc=tl.sum(x*dw[:,None],0)
    adj=tl.sum(tl.trans(inv)*dc[None,:],1)
    dgram=-adj[:,None]*c[None,:]
    dx=dw[:,None]*c[None,:]+tl.dot(basis,dgram,input_precision='tf32x3')
    dbasis=tl.dot(x,tl.trans(dgram),input_precision='tf32x3')+tl.dot(tl.trans(p),dx,input_precision='tf32x3')
    dp=tl.dot(dx,tl.trans(basis),input_precision='tf32x3');dp=.5*(dp+tl.trans(dp))
    tl.atomic_add(DMetric+bh*M*M+m[:,None]*M+m[None,:],dp,mask,sem='relaxed')
    time=t-r;htok=((bh//H)*T+time)*H+bh%H
    tl.atomic_add(DHist+htok[None,:]*M+m[:,None],dbasis,
                  (m[:,None]<M)&(r[None,:]<R)&(time[None,:]>=0),sem='relaxed')


@tr.jit
def recurrence_fwd(Q,Hist,Value,G,Beta,Write,Initial,Out,Final,Tape,T:tl.constexpr,H:tl.constexpr,
                    M:tl.constexpr,V:tl.constexpr,BM:tl.constexpr,BV:tl.constexpr,SLOT:tl.constexpr,SAVE:tl.constexpr):
    bh=tl.program_id(0).to(tl.int64)
    m,v=tl.arange(0,BM),tl.arange(0,BV)
    mask=(m[:,None]<M)&(v[None,:]<V)
    state=tl.load(Initial+bh*M*V+m[:,None]*V+v[None,:],mask,0).to(tl.float32)
    for t in range(T):
        tok=((bh//H)*T+t)*H+bh%H
        if SLOT:alpha=tl.exp(tl.load(G+tok*M+m,m<M,0).to(tl.float32))
        else:alpha=tl.full((BM,),1.,tl.float32)*tl.exp(tl.load(G+tok).to(tl.float32))
        pre=alpha[:,None]*state
        h=tl.load(Hist+tok*M+m,m<M,0).to(tl.float32)
        w=tl.load(Write+tok*M+m,m<M,0)
        q=tl.load(Q+tok*M+m,m<M,0).to(tl.float32)
        beta=tl.load(Beta+tok).to(tl.float32)
        val=tl.load(Value+tok*V+v,v<V,0).to(tl.float32)
        err=val-tl.sum(pre*h[:,None],0)
        state=pre+beta*w[:,None]*err[None,:]
        if SAVE:tl.store(Tape+(bh*T+t)*M*V+m[:,None]*V+v[None,:],pre,mask)
        tl.store(Out+tok*V+v,tl.sum(state*q[:,None],0),v<V)
    tl.store(Final+bh*M*V+m[:,None]*V+v[None,:],state,mask)


@tr.jit
def recurrence_bwd(Q,Hist,Value,G,Beta,Write,Tape,DO,DF,DQ,DHist,DV,DG,DB,DW,DI,
                    T:tl.constexpr,H:tl.constexpr,M:tl.constexpr,V:tl.constexpr,
                    BM:tl.constexpr,BV:tl.constexpr,SLOT:tl.constexpr):
    bh=tl.program_id(0).to(tl.int64)
    m,v=tl.arange(0,BM),tl.arange(0,BV)
    mask=(m[:,None]<M)&(v[None,:]<V)
    ds=tl.load(DF+bh*M*V+m[:,None]*V+v[None,:],mask,0).to(tl.float32)
    for step in range(T):
        t=T-1-step;tok=((bh//H)*T+t)*H+bh%H
        pre=tl.load(Tape+(bh*T+t)*M*V+m[:,None]*V+v[None,:],mask,0)
        h=tl.load(Hist+tok*M+m,m<M,0).to(tl.float32)
        w=tl.load(Write+tok*M+m,m<M,0)
        q=tl.load(Q+tok*M+m,m<M,0).to(tl.float32)
        beta=tl.load(Beta+tok).to(tl.float32)
        val=tl.load(Value+tok*V+v,v<V,0).to(tl.float32)
        do=tl.load(DO+tok*V+v,v<V,0).to(tl.float32)
        err=val-tl.sum(pre*h[:,None],0)
        state=pre+beta*w[:,None]*err[None,:]
        dq=tl.sum(state*do[None,:],1)
        ds+=q[:,None]*do[None,:]
        selected=tl.sum(w[:,None]*ds,0)
        dw=beta*tl.sum(ds*err[None,:],1)
        de=beta*selected
        db=tl.sum(selected*err,0)
        dh=-tl.sum(pre*de[None,:],1)
        dpre=ds-h[:,None]*de[None,:]
        if SLOT:
            alpha=tl.exp(tl.load(G+tok*M+m,m<M,0).to(tl.float32))
            tl.store(DG+tok*M+m,tl.sum(dpre*pre,1),m<M)
        else:
            alpha=tl.full((BM,),1.,tl.float32)*tl.exp(tl.load(G+tok).to(tl.float32))
            tl.store(DG+tok,tl.sum(tl.sum(dpre*pre,1),0))
        ds=alpha[:,None]*dpre
        tl.store(DQ+tok*M+m,dq,m<M);tl.store(DHist+tok*M+m,dh,m<M)
        tl.store(DW+tok*M+m,dw,m<M);tl.store(DV+tok*V+v,de,v<V);tl.store(DB+tok,db)
    tl.store(DI+bh*M*V+m[:,None]*V+v[None,:],ds,mask)


@tr.jit
def add_grad(A,B,C,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
    tl.store(C+i,tl.load(A+i,i<N,0)+tl.load(B+i,i<N,0),i<N)


class _Protected(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,h,v,g,beta,p,initial,anchors,regularization,save):
        b,t,heads,m=h.shape;width=v.shape[-1];r=anchors+1
        alloc=lambda *shape:torch.empty(shape,device=q.device,dtype=torch.float32)
        write=alloc(b,t,heads,m);inv=alloc(b*heads,t,r,r);coeff=alloc(b*heads,t,r)
        bm,bv=max(16,tr.next_power_of_2(m)),tr.next_power_of_2(width)
        direction_fwd[(t,b*heads)](h,p,write,inv,coeff,t,heads,m,bm,r,regularization,num_warps=4)
        tape=alloc(b*heads,t,m,width) if save else initial
        out,final=torch.empty_like(v),alloc(b,heads,m,width)
        recurrence_fwd[(b*heads,)](q,h,v,g,beta,write,initial,out,final,tape,t,heads,m,width,bm,bv,g.ndim==4,save,num_warps=4)
        ctx.save_for_backward(q,h,v,g,beta,p,initial,write,inv,coeff,tape)
        ctx.r=r
        return out,final

    @staticmethod
    @once_differentiable
    def backward(ctx,do,df):
        q,h,v,g,beta,p,initial,write,inv,coeff,tape=ctx.saved_tensors
        b,t,heads,m=h.shape;width=v.shape[-1];bm,bv=max(16,tr.next_power_of_2(m)),tr.next_power_of_2(width)
        dq,dv,dg,db,di=[torch.empty_like(x) for x in (q,v,g,beta,initial)]
        dh_direct=torch.empty(h.shape,device=h.device,dtype=torch.float32)
        dh_geometry=zero(torch.empty_like(dh_direct));dw=torch.empty_like(dh_direct)
        dp=zero(torch.empty_like(p));dh=torch.empty_like(h)
        recurrence_bwd[(b*heads,)](q,h,v,g,beta,write,tape,do.contiguous(),df.contiguous(),dq,dh_direct,dv,dg,db,dw,di,
                                  t,heads,m,width,bm,bv,g.ndim==4,num_warps=4)
        direction_bwd[(t,b*heads)](h,p,inv,coeff,dw,dh_geometry,dp,t,heads,m,bm,ctx.r,num_warps=4)
        add_grad[(tr.cdiv(h.numel(),256),)](dh_direct,dh_geometry,dh,h.numel(),256)
        return dq,dh,dv,dg,db,dp,di,None,None,None


def protected_kapa(q,h,v,g,beta,inverse_metric,initial_state,*,anchors=2,regularization=.05):
    """Soft-protected rank-1 KAPA over supplied fixed-dictionary features.

    q,h [B,T,H,M], v [B,T,H,V], beta [B,T,H], g [B,T,H] or
    [B,T,H,M]; inverse_metric [B,H,M,M], initial_state [B,H,M,V].
    P must be SPD, is symmetrized differentiably; metric must be FP32.
    Previous `anchors` key features are protected relative to the decayed state.
    Positive regularization softens constraints; no claim of exact protection.
    """
    if q.ndim!=4 or q.shape!=h.shape or v.ndim!=4 or v.shape[:3]!=q.shape[:3]:raise ValueError('q/h/v shapes disagree')
    b,t,heads,m=q.shape;width=v.shape[-1]
    if inverse_metric.shape!=(b,heads,m,m) or initial_state.shape!=(b,heads,m,width):raise ValueError('metric/state shape mismatch')
    if beta.shape!=(b,t,heads) or g.shape not in ((b,t,heads),(b,t,heads,m)):raise ValueError('gate shape mismatch')
    if min(b,t,heads,m,width)<1 or m>64 or width>128:raise ValueError('nonempty tensors, M<=64 and V<=128 required')
    if not isinstance(anchors,int) or not 0<=anchors<=7:raise ValueError('anchors must be 0..7')
    if not math.isfinite(regularization) or regularization<=0:raise ValueError('regularization must be positive; hard constraints are not silently approximated')
    xs=(q,h,v,g,beta,inverse_metric,initial_state)
    if q.device.type!='cuda' or any(x.device!=q.device or x.dtype not in (torch.float32,torch.bfloat16) for x in xs):raise ValueError('one CUDA device and FP32/BF16 tensors required')
    if inverse_metric.dtype!=torch.float32:raise ValueError('inverse_metric must be FP32')
    save=torch.is_grad_enabled() and any(x.requires_grad for x in xs)
    with torch.cuda.device(q.device):
        return _Protected.apply(*[x.contiguous() for x in xs],anchors,regularization,save)


def gaussian_protected_kapa(q,k,v,g,beta,centers,initial_state,*,temperature=.5,gram_ridge=.01,anchors=2,regularization=.05):
    """All-Triton Gaussian geometry and protected KAPA forward/backward.

    Centers remain fixed for the entire call, with gradients through Gaussian
    features and the inverse Gram. Raw key width D<=128; numeric ops use Triton.
    """
    if not math.isfinite(temperature) or temperature<=0 or not math.isfinite(gram_ridge) or gram_ridge<=0:
        raise ValueError('temperature and Gram ridge must be positive and finite')
    from .kapa_geometry import gaussian_features
    qf,h,p=gaussian_features(q,k,centers,temperature=temperature,gram_ridge=gram_ridge)
    return protected_kapa(qf,h,v,g,beta,p,initial_state,anchors=anchors,regularization=regularization)
