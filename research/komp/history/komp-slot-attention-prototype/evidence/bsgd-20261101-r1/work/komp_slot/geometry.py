"""Parallel kernel geometry and analytic solve/normalization gradients in Triton."""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable
from .utils import zero


@tr.jit
def normalize_kernel(Q,K,Initial,QN,Pool,T:tl.constexpr,H:tl.constexpr,M:tl.constexpr,D:tl.constexpr,BD:tl.constexpr):
    row=tl.program_id(0).to(tl.int64); bh=tl.program_id(1).to(tl.int64)
    d=tl.arange(0,BD)
    if row<T:
        tok=((bh//H)*T+row)*H+bh%H
        q=tl.load(Q+tok*D+d,d<D,0).to(tl.float32)
        k=tl.load(K+tok*D+d,d<D,0).to(tl.float32)
        q=q/tl.maximum(tl.sqrt(tl.sum(q*q,0)),1.e-6)
        k=k/tl.maximum(tl.sqrt(tl.sum(k*k,0)),1.e-6)
        tl.store(QN+tok*D+d,q,d<D)
        tl.store(Pool+(bh*(T+M)+M+row)*D+d,k,d<D)
    else:
        m=row-T
        k=tl.load(Initial+(bh*M+m)*D+d,d<D,0).to(tl.float32)
        k=k/tl.maximum(tl.sqrt(tl.sum(k*k,0)),1.e-6)
        tl.store(Pool+(bh*(T+M)+m)*D+d,k,d<D)


@tr.jit
def routes_kernel(Slot,Source,Removed,Last,T:tl.constexpr,H:tl.constexpr,M:tl.constexpr,BM:tl.constexpr):
    bh=tl.program_id(0).to(tl.int64); m=tl.arange(0,BM)
    src=m
    for t in range(T):
        tok=((bh//H)*T+t)*H+bh%H
        j=tl.load(Slot+tok)
        old=tl.sum(tl.where(m==j,src,0),0)
        src=tl.where(m==j,M+t,src)
        tl.store(Source+tok*M+m,src,m<M)
        tl.store(Removed+tok,old)
    tl.store(Last+bh*M+m,src,m<M)


@tr.jit
def _imax(a,b):
    return tl.maximum(a,b)


@tr.jit
def routes_local_kernel(Slot,Source,Summary,T:tl.constexpr,H:tl.constexpr,M:tl.constexpr,
                        C:tl.constexpr,BM:tl.constexpr,NC:tl.constexpr):
    block=tl.program_id(0).to(tl.int64);bh=tl.program_id(1).to(tl.int64)
    c=tl.arange(0,C);m=tl.arange(0,BM)
    t=block*C+c;tok=((bh//H)*T+t)*H+bh%H
    slot=tl.load(Slot+tok,t<T,other=-1)
    latest=tl.where((slot[:,None]==m[None,:])&(t[:,None]<T),M+t[:,None],-1).to(tl.int32)
    prefix=tl.associative_scan(latest,0,_imax)
    tl.store(Source+tok[:,None]*M+m[None,:],prefix,(t[:,None]<T)&(m[None,:]<M))
    end=tl.max(prefix,0)
    tl.store(Summary+(bh*NC+block)*M+m,end,m<M)


@tr.jit
def routes_carry_kernel(Summary,Carry,H:tl.constexpr,M:tl.constexpr,NC:tl.constexpr,
                        BN:tl.constexpr,BM:tl.constexpr):
    bh=tl.program_id(0).to(tl.int64)
    n=tl.arange(0,BN);m=tl.arange(0,BM)
    # Loading the preceding summary makes this an exclusive block prefix.
    previous=tl.load(Summary+(bh*NC+n[:,None]-1)*M+m[None,:],
                     (n[:,None]>0)&(n[:,None]<=NC)&(m[None,:]<M),other=-1)
    previous=tl.maximum(previous,m[None,:])
    carry=tl.associative_scan(previous,0,_imax)
    tl.store(Carry+(bh*NC+n[:,None])*M+m[None,:],carry,(n[:,None]<NC)&(m[None,:]<M))


@tr.jit
def routes_finish_kernel(Slot,Source,Removed,Last,Carry,T:tl.constexpr,H:tl.constexpr,
                         M:tl.constexpr,C:tl.constexpr,BM:tl.constexpr,NC:tl.constexpr):
    block=tl.program_id(0).to(tl.int64);bh=tl.program_id(1).to(tl.int64)
    c=tl.arange(0,C);m=tl.arange(0,BM)
    t=block*C+c;tok=((bh//H)*T+t)*H+bh%H
    local=tl.load(Source+tok[:,None]*M+m[None,:],(t[:,None]<T)&(m[None,:]<M),other=-1)
    carry=tl.load(Carry+(bh*NC+block)*M+m,m<M,other=-1)
    # Previous local row is read from this CTA's already loaded tensor. There
    # are no cross-block reads from Source while it is overwritten in place.
    previous=tl.gather(local,tl.broadcast_to(tl.maximum(c-1,0)[:,None],(C,BM)),0)
    previous=tl.where(c[:,None]>0,previous,-1)
    previous=tl.maximum(previous,carry[None,:])
    slot=tl.load(Slot+tok,t<T,other=-1)
    removed=tl.sum(tl.where(m[None,:]==slot[:,None],previous,0),1)
    source=tl.maximum(local,carry[None,:])
    tl.store(Source+tok[:,None]*M+m[None,:],source,(t[:,None]<T)&(m[None,:]<M))
    tl.store(Removed+tok,removed,t<T)
    if block==NC-1:
        final=tl.max(source,0)
        tl.store(Last+bh*M+m,final,m<M)


def routes_parallel_into(slots,sources,removed,last,m,block_size=128):
    """Exact integer route history; three scans, no serial token loop."""
    b,t,h=slots.shape;nc=tr.cdiv(t,block_size);bm=tr.next_power_of_2(m)
    summary=torch.empty((b*h,nc,m),device=slots.device,dtype=torch.int32)
    carry=torch.empty_like(summary)
    routes_local_kernel[(nc,b*h)](slots,sources,summary,t,h,m,block_size,bm,nc,num_warps=4)
    routes_carry_kernel[(b*h,)](summary,carry,h,m,nc,tr.next_power_of_2(nc),bm,num_warps=4)
    routes_finish_kernel[(nc,b*h)](slots,sources,removed,last,carry,t,h,m,block_size,bm,nc,num_warps=4)


@tr.jit
def _factor_solve(gram,rhs,M:tl.constexpr,BM:tl.constexpr):
    m=tl.arange(0,BM)
    fact=tl.zeros((BM,BM),tl.float32)
    a=gram
    b=rhs
    for j in range(M):
        col=tl.sum(tl.where(m[None,:]==j,a,0.),1)
        diag=tl.sum(tl.where(m==j,col,0.),0)
        lj=tl.where(m>j,col/diag,0.)
        bj=tl.sum(tl.where(m==j,b,0.),0)
        fact+=tl.where(m[None,:]==j,tl.where(m[:,None]==j,diag,lj[:,None]),0.)
        a-=tl.where((m[:,None]>j)&(m[None,:]>j),col[:,None]*lj[None,:],0.)
        b-=lj*bj
    d=tl.sum(tl.where(m[:,None]==m[None,:],fact,0.),1)
    sol=b/tl.where(m<M,d,1.)
    for step in range(M):
        j=M-1-step
        col=tl.sum(tl.where(m[None,:]==j,fact,0.),1)
        correction=tl.sum(tl.where(m>j,col*sol,0.),0)
        sol=tl.where(m==j,sol-correction,sol)
    return fact,sol


@tr.jit
def _solve(fact,rhs,M:tl.constexpr,BM:tl.constexpr):
    m=tl.arange(0,BM)
    b=rhs
    for j in range(M):
        col=tl.sum(tl.where(m[None,:]==j,fact,0.),1)
        bj=tl.sum(tl.where(m==j,b,0.),0)
        b-=tl.where(m>j,col*bj,0.)
    d=tl.sum(tl.where(m[:,None]==m[None,:],fact,0.),1)
    sol=b/tl.where(m<M,d,1.)
    for step in range(M):
        j=M-1-step
        col=tl.sum(tl.where(m[None,:]==j,fact,0.),1)
        correction=tl.sum(tl.where(m>j,col*sol,0.),0)
        sol=tl.where(m==j,sol-correction,sol)
    return sol


@tr.jit
def _column(a, j, BM:tl.constexpr):
    # Gather one column instead of reducing a masked BM x BM tile each pivot.
    return tl.reshape(tl.gather(a, tl.full((BM,1),j,tl.int32),1),(BM,))


@tr.jit
def _element(a, j):
    return tl.sum(tl.gather(a,tl.full((1,),j,tl.int32),0),0)


@tr.jit
def _factor_solve_gather(gram,rhs,M:tl.constexpr,BM:tl.constexpr):
    m=tl.arange(0,BM)
    fact=tl.zeros((BM,BM),tl.float32)
    a=gram
    b=rhs
    for j in range(M):
        col=_column(a,j,BM)
        diag=_element(col,j)
        lj=tl.where(m>j,col/diag,0.)
        bj=_element(b,j)
        fact+=tl.where(m[None,:]==j,tl.where(m[:,None]==j,diag,lj[:,None]),0.)
        a-=tl.where((m[:,None]>j)&(m[None,:]>j),col[:,None]*lj[None,:],0.)
        b-=lj*bj
    d=tl.sum(tl.where(m[:,None]==m[None,:],fact,0.),1)
    sol=b/tl.where(m<M,d,1.)
    for step in range(M):
        j=M-1-step
        col=_column(fact,j,BM)
        correction=tl.sum(tl.where(m>j,col*sol,0.),0)
        sol=tl.where(m==j,sol-correction,sol)
    return fact,sol


@tr.jit
def _solve_gather(fact,rhs,M:tl.constexpr,BM:tl.constexpr):
    m=tl.arange(0,BM)
    b=rhs
    for j in range(M):
        col=_column(fact,j,BM)
        bj=_element(b,j)
        b-=tl.where(m>j,col*bj,0.)
    d=tl.sum(tl.where(m[:,None]==m[None,:],fact,0.),1)
    sol=b/tl.where(m<M,d,1.)
    for step in range(M):
        j=M-1-step
        col=_column(fact,j,BM)
        correction=tl.sum(tl.where(m>j,col*sol,0.),0)
        sol=tl.where(m==j,sol-correction,sol)
    return sol


@tr.jit
def geometry_fwd_kernel(QN,Pool,Source,Removed,Slot,Hist,Proj,Read,Factor,
                        T:tl.constexpr,H:tl.constexpr,M:tl.constexpr,D:tl.constexpr,
                        BM:tl.constexpr,BD:tl.constexpr,TAU:tl.constexpr,RIDGE:tl.constexpr,SAVE:tl.constexpr,
                        GATHER:tl.constexpr=False,DOT_PRECISION:tl.constexpr='ieee'):
    tok=tl.program_id(0).to(tl.int64)
    head=tok%H; batch=tok//(T*H); time=tok//H%T; bh=batch*H+head
    m=tl.arange(0,BM); d=tl.arange(0,BD)
    src=tl.load(Source+tok*M+m,m<M,0)
    removed=tl.load(Removed+tok)
    j=tl.load(Slot+tok)
    keys=tl.load(Pool+(bh*(T+M)+src[:,None])*D+d[None,:],(m[:,None]<M)&(d[None,:]<D),0)
    old=tl.load(Pool+(bh*(T+M)+removed)*D+d,d<D,0)
    key=tl.load(Pool+(bh*(T+M)+M+time)*D+d,d<D,0)
    q=tl.load(QN+tok*D+d,d<D,0)
    oldkeys=tl.where(m[:,None]==j,old[None,:],keys)
    hist=tl.exp((tl.sum(oldkeys*key[None,:],1)-1.)/TAU)
    read=tl.exp((tl.sum(keys*q[None,:],1)-1.)/TAU)
    rhs=tl.exp((tl.sum(keys*old[None,:],1)-1.)/TAU)
    gram=tl.exp((tl.dot(keys,tl.trans(keys),input_precision=DOT_PRECISION)-1.)/TAU)
    gram=tl.where((m[:,None]<M)&(m[None,:]<M),gram,0.)+(m[:,None]==m[None,:])*RIDGE
    if GATHER:
        fact,p=_factor_solve_gather(gram,tl.where(m<M,rhs,0.),M,BM)
    else:
        fact,p=_factor_solve(gram,tl.where(m<M,rhs,0.),M,BM)
    tl.store(Hist+tok*M+m,hist,m<M); tl.store(Proj+tok*M+m,p,m<M)
    tl.store(Read+tok*M+m,read,m<M)
    if SAVE:
        tl.store(Factor+tok*M*M+m[:,None]*M+m[None,:],fact,(m[:,None]<M)&(m[None,:]<M))


@tr.jit
def final_keys_kernel(Pool,Last,Final,T:tl.constexpr,M:tl.constexpr,D:tl.constexpr,BD:tl.constexpr):
    bh=tl.program_id(0).to(tl.int64); m=tl.program_id(1); d=tl.arange(0,BD)
    src=tl.load(Last+bh*M+m)
    val=tl.load(Pool+(bh*(T+M)+src)*D+d,d<D,0)
    tl.store(Final+(bh*M+m)*D+d,val,d<D)


@tr.jit
def geometry_bwd_kernel(QN,Pool,Source,Removed,Slot,Hist,Proj,Read,Factor,DH,DP,DR,DQN,DPool,
                        T:tl.constexpr,H:tl.constexpr,M:tl.constexpr,D:tl.constexpr,
                        BM:tl.constexpr,BD:tl.constexpr,TAU:tl.constexpr,
                        GATHER:tl.constexpr=False,DOT_PRECISION:tl.constexpr='ieee'):
    tok=tl.program_id(0).to(tl.int64)
    head=tok%H; batch=tok//(T*H); time=tok//H%T; bh=batch*H+head
    m=tl.arange(0,BM); d=tl.arange(0,BD)
    src=tl.load(Source+tok*M+m,m<M,0); oldsrc=tl.load(Removed+tok); j=tl.load(Slot+tok)
    keys=tl.load(Pool+(bh*(T+M)+src[:,None])*D+d[None,:],(m[:,None]<M)&(d[None,:]<D),0)
    old=tl.load(Pool+(bh*(T+M)+oldsrc)*D+d,d<D,0)
    key=tl.load(Pool+(bh*(T+M)+M+time)*D+d,d<D,0)
    q=tl.load(QN+tok*D+d,d<D,0)
    hist=tl.load(Hist+tok*M+m,m<M,0); p=tl.load(Proj+tok*M+m,m<M,0)
    read=tl.load(Read+tok*M+m,m<M,0)
    dh=tl.load(DH+tok*M+m,m<M,0)*hist/TAU
    dp=tl.load(DP+tok*M+m,m<M,0)
    dr=tl.load(DR+tok*M+m,m<M,0)*read/TAU
    fact=tl.load(Factor+tok*M*M+m[:,None]*M+m[None,:],(m[:,None]<M)&(m[None,:]<M),0)
    if GATHER:
        z=_solve_gather(fact,dp,M,BM)
    else:
        z=_solve(fact,dp,M,BM)
    gram=tl.exp((tl.dot(keys,tl.trans(keys),input_precision=DOT_PRECISION)-1.)/TAU)
    dg=-(z[:,None]*p[None,:]+p[:,None]*z[None,:])*gram/TAU
    dg=tl.where((m[:,None]<M)&(m[None,:]<M),dg,0.)
    rhs=tl.exp((tl.sum(keys*old[None,:],1)-1.)/TAU)
    db=z*rhs/TAU
    dkeys=tl.dot(dg,keys,input_precision=DOT_PRECISION)+db[:,None]*old[None,:]+dr[:,None]*q[None,:]
    dkeys+=tl.where(m[:,None]!=j,dh[:,None]*key[None,:],0.)
    dold=tl.sum(db[:,None]*keys,0)+tl.sum(tl.where(m==j,dh,0.),0)*key
    oldkeys=tl.where(m[:,None]==j,old[None,:],keys)
    dkey=tl.sum(dh[:,None]*oldkeys,0)
    dq=tl.sum(dr[:,None]*keys,0)
    tl.store(DQN+tok*D+d,dq,d<D)
    tl.atomic_add(DPool+(bh*(T+M)+src[:,None])*D+d[None,:],dkeys,(m[:,None]<M)&(d[None,:]<D),sem='relaxed')
    tl.atomic_add(DPool+(bh*(T+M)+oldsrc)*D+d,dold,d<D,sem='relaxed')
    tl.atomic_add(DPool+(bh*(T+M)+M+time)*D+d,dkey,d<D,sem='relaxed')


@tr.jit
def final_bwd_kernel(Last,DF,DPool,T:tl.constexpr,M:tl.constexpr,D:tl.constexpr,BD:tl.constexpr):
    bh=tl.program_id(0).to(tl.int64); m=tl.program_id(1); d=tl.arange(0,BD)
    src=tl.load(Last+bh*M+m)
    g=tl.load(DF+(bh*M+m)*D+d,d<D,0)
    tl.atomic_add(DPool+(bh*(T+M)+src)*D+d,g,d<D,sem='relaxed')


@tr.jit
def _norm_grad(raw,g):
    norm=tl.sqrt(tl.sum(raw*raw,0))
    denom=tl.maximum(norm,1.e-6)
    normalized=raw/denom
    return (g-tl.where(norm>=1.e-6,normalized*tl.sum(normalized*g,0),0.))/denom


@tr.jit
def normalize_bwd_kernel(Q,K,Initial,DQN,DPool,DQ,DK,DI,
                          T:tl.constexpr,H:tl.constexpr,M:tl.constexpr,D:tl.constexpr,BD:tl.constexpr):
    row=tl.program_id(0).to(tl.int64); bh=tl.program_id(1).to(tl.int64); d=tl.arange(0,BD)
    if row<T:
        tok=((bh//H)*T+row)*H+bh%H
        q=tl.load(Q+tok*D+d,d<D,0).to(tl.float32); k=tl.load(K+tok*D+d,d<D,0).to(tl.float32)
        dq=tl.load(DQN+tok*D+d,d<D,0)
        dk=tl.load(DPool+(bh*(T+M)+M+row)*D+d,d<D,0)
        tl.store(DQ+tok*D+d,_norm_grad(q,dq),d<D)
        tl.store(DK+tok*D+d,_norm_grad(k,dk),d<D)
    else:
        m=row-T
        k=tl.load(Initial+(bh*M+m)*D+d,d<D,0).to(tl.float32)
        dk=tl.load(DPool+(bh*(T+M)+m)*D+d,d<D,0)
        tl.store(DI+(bh*M+m)*D+d,_norm_grad(k,dk),d<D)


class _Geometry(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,slots,initial,tau,ridge,save,gather,precision,parallel_routes):
        b,t,h,d=q.shape; m=initial.shape[-2]; bm=max(16,tr.next_power_of_2(m)); bd=max(16,tr.next_power_of_2(d))
        alloc=lambda *s:torch.empty(s,device=q.device,dtype=torch.float32)
        qn=alloc(b,t,h,d); pool=alloc(b,h,t+m,d)
        sources=torch.empty((b,t,h,m),device=q.device,dtype=torch.int32)
        removed=torch.empty((b,t,h),device=q.device,dtype=torch.int32)
        last=torch.empty((b,h,m),device=q.device,dtype=torch.int32)
        hist,proj,read=[alloc(b,t,h,m) for _ in range(3)]
        factor=alloc(b,t,h,m,m) if save else alloc(1)
        final=alloc(b,h,m,d)
        normalize_kernel[(t+m,b*h)](q,k,initial,qn,pool,t,h,m,d,bd)
        if parallel_routes:
            routes_parallel_into(slots,sources,removed,last,m)
        else:
            routes_kernel[(b*h,)](slots,sources,removed,last,t,h,m,tr.next_power_of_2(m))
        geometry_fwd_kernel[(b*t*h,)](qn,pool,sources,removed,slots,hist,proj,read,factor,t,h,m,d,bm,bd,tau,ridge,save,gather,precision,num_warps=4)
        final_keys_kernel[(b*h,m)](pool,last,final,t,m,d,bd)
        ctx.save_for_backward(q,k,initial,slots,qn,pool,sources,removed,last,hist,proj,read,factor)
        ctx.tau=tau; ctx.gather=gather; ctx.precision=precision
        return hist,proj,read,final

    @staticmethod
    @once_differentiable
    def backward(ctx,dh,dp,dr,df):
        q,k,initial,slots,qn,pool,sources,removed,last,hist,proj,read,factor=ctx.saved_tensors
        b,t,h,d=q.shape; m=initial.shape[-2]; bm=max(16,tr.next_power_of_2(m)); bd=max(16,tr.next_power_of_2(d))
        dpool=zero(torch.empty_like(pool)); dqn=torch.empty_like(qn)
        dq,dk,di=[torch.empty_like(x) for x in (q,k,initial)]
        geometry_bwd_kernel[(b*t*h,)](qn,pool,sources,removed,slots,hist,proj,read,factor,
            dh.contiguous(),dp.contiguous(),dr.contiguous(),dqn,dpool,t,h,m,d,bm,bd,ctx.tau,ctx.gather,ctx.precision,num_warps=4)
        final_bwd_kernel[(b*h,m)](last,df.contiguous(),dpool,t,m,d,bd)
        normalize_bwd_kernel[(t+m,b*h)](q,k,initial,dqn,dpool,dq,dk,di,t,h,m,d,bd)
        return dq,dk,None,di,None,None,None,None,None,None


def triton_geometry_serial(q,k,slots,initial,temperature=.5,ridge=1e-4):
    """Frozen serial-route/IEEE baseline for numerical and timing comparisons."""
    save=torch.is_grad_enabled() and any(x.requires_grad for x in (q,k,initial))
    with torch.cuda.device(q.device):
        return _Geometry.apply(q.contiguous(),k.contiguous(),slots.contiguous(),initial.contiguous(),temperature,ridge,save,False,'ieee',False)


def triton_geometry_fast(q,k,slots,initial,temperature=.5,ridge=1e-4,dot_precision='tf32x3'):
    """Opt-in gather LDL solve; explicit IEEE or three-component TF32 dot products.

    The exponential kernel, ridge, normalization, route tracking, and analytic
    gradient are unchanged. TF32x3 is a different FP32 arithmetic realization,
    so callers must validate its error on their conditioning regime.
    """
    if dot_precision not in ('ieee','tf32x3'):
        raise ValueError("dot_precision must be 'ieee' or 'tf32x3'")
    save=torch.is_grad_enabled() and any(x.requires_grad for x in (q,k,initial))
    with torch.cuda.device(q.device):
        return _Geometry.apply(q.contiguous(),k.contiguous(),slots.contiguous(),initial.contiguous(),
                               temperature,ridge,save,True,dot_precision,False)


def triton_geometry_parallel(q,k,slots,initial,temperature=.5,ridge=1e-4,
                             gather=False,dot_precision='ieee'):
    """Opt-in parallel route scan, independently selectable solver/arithmetic.

    Default uses the original LDL solve and IEEE products, isolating the exact
    integer history optimization from arithmetic candidates.
    """
    if dot_precision not in ('ieee','tf32x3'):
        raise ValueError("dot_precision must be 'ieee' or 'tf32x3'")
    save=torch.is_grad_enabled() and any(x.requires_grad for x in (q,k,initial))
    with torch.cuda.device(q.device):
        return _Geometry.apply(q.contiguous(),k.contiguous(),slots.contiguous(),initial.contiguous(),
                               temperature,ridge,save,gather,dot_precision,True)


def triton_geometry(q,k,slots,initial,temperature=.5,ridge=1e-4):
    """IEEE geometry with exact parallel route scans for T>=192 and M>=16.

    The conservative threshold is empirically selected on H100. It affects only
    integer source history; small shapes retain the serial baseline. The LDL
    solver, floating-point arithmetic and gradients are unchanged.
    """
    if q.shape[1]>=192 and initial.shape[-2]>=16:
        return triton_geometry_parallel(q,k,slots,initial,temperature,ridge)
    return triton_geometry_serial(q,k,slots,initial,temperature,ridge)
