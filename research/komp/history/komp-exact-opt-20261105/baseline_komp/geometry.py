"""Triton geometry for per-token KOMP; projections use FP64 throughout."""
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def normalize(Q,K,K0,QN,Pool,T:tl.constexpr,M:tl.constexpr,D:tl.constexpr,BD:tl.constexpr):
    row,bh=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64)
    d=tl.arange(0,BD)
    if row<T:
        q=tl.load(Q+(bh*T+row)*D+d,d<D,0).to(tl.float64)
        k=tl.load(K+(bh*T+row)*D+d,d<D,0).to(tl.float64)
        q=q/tl.maximum(libdevice.sqrt(tl.sum(q*q,0)),1.e-6)
        k=k/tl.maximum(libdevice.sqrt(tl.sum(k*k,0)),1.e-6)
        tl.store(QN+(bh*T+row)*D+d,q,d<D)
        tl.store(Pool+(bh*(T+M)+M+row)*D+d,k,d<D)
    else:
        j=row-T
        k=tl.load(K0+(bh*M+j)*D+d,d<D,0).to(tl.float64)
        k=k/tl.maximum(libdevice.sqrt(tl.sum(k*k,0)),1.e-6)
        tl.store(Pool+(bh*(T+M)+j)*D+d,k,d<D)


@tr.jit
def select(Score,Score0,Birth0,IDs,DropSlot,SF,BF,T:tl.constexpr,M:tl.constexpr,
           BM:tl.constexpr,LEARNED:tl.constexpr,START:tl.constexpr):
    bh=tl.program_id(0).to(tl.int64)
    r=tl.arange(0,BM);valid=r<M
    ids=r
    sc=tl.load(Score0+bh*M+r,valid,0).to(tl.float32)
    born=tl.load(Birth0+bh*M+r,valid,0).to(tl.int64)
    tl.store(IDs+(bh*(T+1))*M+r,ids,valid)
    for t in range(T):
        if LEARNED:
            s=tl.load(Score+bh*T+t).to(tl.float32)
            low=tl.min(tl.where(valid,sc,float('inf')),0)
            age=tl.min(tl.where(valid & (sc==low),born,9223372036854775807),0)
            slot=tl.min(tl.where(valid & (sc==low) & (born==age),r,2147483647),0)
            slot=tl.where(s>=low,slot,-1)
        else:
            slot=(START+t)%M
            s=0.
        ids=tl.where(r==slot,M+t,ids)
        sc=tl.where(r==slot,s,sc)
        born=tl.where(r==slot,START+t,born)
        tl.store(DropSlot+bh*T+t,slot)
        tl.store(IDs+(bh*(T+1)+t+1)*M+r,ids,valid)
    tl.store(SF+bh*M+r,sc,valid)
    tl.store(BF+bh*M+r,born,valid)


@tr.jit
def effective_beta(Beta,Score,BE,N:tl.constexpr,GATE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    b=tl.load(Beta+i,i<N,0).to(tl.float32)
    if GATE:
        s=tl.load(Score+i,i<N,0).to(tl.float32)
        b=b*2*tl.sigmoid(s)
    tl.store(BE+i,b,i<N)


@tr.jit
def inverse_sweep(gram,M:tl.constexpr,BM:tl.constexpr):
    # Symmetric sweep in FP64, adapted from the existing KOMP Gram solver.
    r=tl.arange(0,BM)
    a=gram
    for j in range(M):
        col=tl.sum(tl.where(r[None,:]==j,a,0.),1)
        pivot=tl.sum(tl.where(r==j,col,0.),0)
        c=col/pivot
        a=a-col[:,None]*c[None,:]
        a=tl.where(r[:,None]==j,c[None,:],tl.where(r[None,:]==j,c[:,None],a))
        a=tl.where((r[:,None]==j)&(r[None,:]==j),-1./pivot,a)
    return tl.where((r[:,None]<M)&(r[None,:]<M),-a,0.)


@tr.jit
def forward(QN,Pool,IDs,DropSlot,P,H,R,N,Inv,Gram,Cross,
            T:tl.constexpr,M:tl.constexpr,D:tl.constexpr,BM:tl.constexpr,
            TAU:tl.constexpr,RIDGE:tl.constexpr):
    t,bh=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64)
    r=tl.arange(0,BM);valid=r<M
    old=tl.load(IDs+(bh*(T+1)+t)*M+r,valid,0)
    kept=tl.load(IDs+(bh*(T+1)+t+1)*M+r,valid,0)
    slot=tl.load(DropSlot+bh*T+t)
    dropped=tl.where(slot>=0,tl.sum(tl.where(r==slot,old,0),0),M+t)
    gram=tl.zeros((BM,BM),tl.float64)
    cross=tl.zeros((BM,),tl.float64)
    hist=tl.zeros((BM,),tl.float64)
    read=tl.zeros((BM,),tl.float64)
    qnew=tl.full((),0,tl.float64)
    base=bh*(T+M)
    for d in range(D):
        nk=tl.load(Pool+(base+kept)*D+d,valid,0)
        ok=tl.load(Pool+(base+old)*D+d,valid,0)
        drop=tl.load(Pool+(base+dropped)*D+d)
        new=tl.load(Pool+(base+M+t)*D+d)
        q=tl.load(QN+(bh*T+t)*D+d)
        gram=gram+nk[:,None]*nk[None,:]
        cross=cross+nk*drop
        hist=hist+ok*new
        read=read+ok*q
        qnew=qnew+q*new
    both=valid[:,None]&valid[None,:]
    gram=tl.where(both,libdevice.exp((gram-1.)/TAU),0.)
    cross=tl.where(valid,libdevice.exp((cross-1.)/TAU),0.)
    hist=tl.where(valid,libdevice.exp((hist-1.)/TAU),0.)
    read=tl.where(valid,libdevice.exp((read-1.)/TAU),0.)
    qnew=libdevice.exp((qnew-1.)/TAU)
    inv=inverse_sweep(gram+tl.where(r[:,None]==r[None,:],RIDGE,0.),M,BM)
    p=tl.sum(inv*cross[None,:],1)
    idx=bh*T+t
    tl.store(P+idx*M+r,p,valid)
    tl.store(H+idx*M+r,hist,valid)
    tl.store(R+idx*M+r,read,valid)
    tl.store(N+idx,qnew)
    tl.store(Cross+idx*M+r,cross,valid)
    tl.store(Inv+idx*M*M+r[:,None]*M+r[None,:],inv,both)
    tl.store(Gram+idx*M*M+r[:,None]*M+r[None,:],gram,both)


@tr.jit
def final_keys(Pool,IDs,KF,T:tl.constexpr,M:tl.constexpr,D:tl.constexpr,BLOCK:tl.constexpr):
    bh=tl.program_id(1).to(tl.int64)
    x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=x//D;d=x%D
    source=tl.load(IDs+(bh*(T+1)+T)*M+row,row<M,0)
    k=tl.load(Pool+(bh*(T+M)+source)*D+d,row<M,0)
    tl.store(KF+bh*M*D+x,k,row<M)


@tr.jit
def final_keys_backward(DKF,IDs,DPool,T:tl.constexpr,M:tl.constexpr,D:tl.constexpr,BLOCK:tl.constexpr):
    bh=tl.program_id(1).to(tl.int64)
    x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=x//D;d=x%D
    source=tl.load(IDs+(bh*(T+1)+T)*M+row,row<M,0)
    grad=tl.load(DKF+bh*M*D+x,row<M,0).to(tl.float64)
    tl.atomic_add(DPool+(bh*(T+M)+source)*D+d,grad,row<M,sem='relaxed')


@tr.jit
def backward(QN,Pool,IDs,DropSlot,P,H,R,N,Inv,Gram,Cross,DP,DH,DR,DN,DQN,DPool,
             T:tl.constexpr,M:tl.constexpr,D:tl.constexpr,BM:tl.constexpr,TAU:tl.constexpr):
    t,bh=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64)
    r=tl.arange(0,BM);valid=r<M;both=valid[:,None]&valid[None,:]
    idx=bh*T+t;base=bh*(T+M)
    old=tl.load(IDs+(bh*(T+1)+t)*M+r,valid,0)
    kept=tl.load(IDs+(bh*(T+1)+t+1)*M+r,valid,0)
    slot=tl.load(DropSlot+idx)
    dropped=tl.where(slot>=0,tl.sum(tl.where(r==slot,old,0),0),M+t)
    inv=tl.load(Inv+idx*M*M+r[:,None]*M+r[None,:],both,0)
    gram=tl.load(Gram+idx*M*M+r[:,None]*M+r[None,:],both,0)
    p=tl.load(P+idx*M+r,valid,0)
    dp=tl.load(DP+idx*M+r,valid,0).to(tl.float64)
    z=tl.sum(inv*dp[None,:],1)
    dc=z*tl.load(Cross+idx*M+r,valid,0)/TAU
    dmat=-(z[:,None]*p[None,:]+p[:,None]*z[None,:])*gram/TAU
    dh=tl.load(DH+idx*M+r,valid,0).to(tl.float64)*tl.load(H+idx*M+r,valid,0)/TAU
    dr=tl.load(DR+idx*M+r,valid,0).to(tl.float64)*tl.load(R+idx*M+r,valid,0)/TAU
    dn=tl.load(DN+idx).to(tl.float64)*tl.load(N+idx)/TAU
    for d in range(D):
        nk=tl.load(Pool+(base+kept)*D+d,valid,0)
        ok=tl.load(Pool+(base+old)*D+d,valid,0)
        drop=tl.load(Pool+(base+dropped)*D+d)
        new=tl.load(Pool+(base+M+t)*D+d)
        q=tl.load(QN+idx*D+d)
        dkeep=tl.sum(dmat*nk[None,:],1)+dc*drop
        dold=dh*new+dr*q
        ddrop=tl.sum(dc*nk,0)
        dnew=tl.sum(dh*ok,0)+dn*q
        dq=tl.sum(dr*ok,0)+dn*new
        tl.atomic_add(DPool+(base+kept)*D+d,dkeep,valid,sem='relaxed')
        tl.atomic_add(DPool+(base+old)*D+d,dold,valid,sem='relaxed')
        tl.atomic_add(DPool+(base+dropped)*D+d,ddrop,sem='relaxed')
        tl.atomic_add(DPool+(base+M+t)*D+d,dnew,sem='relaxed')
        tl.store(DQN+idx*D+d,dq)


@tr.jit
def normalize_backward(Q,K,K0,QN,Pool,DQN,DPool,DQ,DK,DK0,
                       T:tl.constexpr,M:tl.constexpr,D:tl.constexpr,BD:tl.constexpr):
    row,bh=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64)
    d=tl.arange(0,BD)
    if row<T:
        off=(bh*T+row)*D+d
        q=tl.load(Q+off,d<D,0).to(tl.float64)
        n=tl.load(QN+off,d<D,0)
        grad=tl.load(DQN+off,d<D,0)
        norm=libdevice.sqrt(tl.sum(q*q,0))
        dq=(grad-tl.where(norm>=1.e-6,n*tl.sum(n*grad,0),0.))/tl.maximum(norm,1.e-6)
        tl.store(DQ+off,dq,d<D)
        raw=tl.load(K+off,d<D,0).to(tl.float64)
        poff=(bh*(T+M)+M+row)*D+d
        n=tl.load(Pool+poff,d<D,0);grad=tl.load(DPool+poff,d<D,0)
        norm=libdevice.sqrt(tl.sum(raw*raw,0))
        dk=(grad-tl.where(norm>=1.e-6,n*tl.sum(n*grad,0),0.))/tl.maximum(norm,1.e-6)
        tl.store(DK+off,dk,d<D)
    else:
        j=row-T;off=(bh*M+j)*D+d;poff=(bh*(T+M)+j)*D+d
        raw=tl.load(K0+off,d<D,0).to(tl.float64)
        n=tl.load(Pool+poff,d<D,0);grad=tl.load(DPool+poff,d<D,0)
        norm=libdevice.sqrt(tl.sum(raw*raw,0))
        dk=(grad-tl.where(norm>=1.e-6,n*tl.sum(n*grad,0),0.))/tl.maximum(norm,1.e-6)
        tl.store(DK0+off,dk,d<D)
