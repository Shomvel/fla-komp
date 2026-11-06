"""Triton geometry for per-token KOMP; FP32 production geometry, optional FP64 numerical reference."""
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def normalize(Q,K,K0,QN,Pool,T:tl.constexpr,M:tl.constexpr,D:tl.constexpr,BD:tl.constexpr):
    row,bh=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64)
    d=tl.arange(0,BD)
    if row<T:
        q=tl.load(Q+(bh*T+row)*D+d,d<D,0).to(Pool.dtype.element_ty)
        k=tl.load(K+(bh*T+row)*D+d,d<D,0).to(Pool.dtype.element_ty)
        q=q/tl.maximum(libdevice.sqrt(tl.sum(q*q,0)),1.e-6)
        k=k/tl.maximum(libdevice.sqrt(tl.sum(k*k,0)),1.e-6)
        tl.store(QN+(bh*T+row)*D+d,q,d<D)
        tl.store(Pool+(bh*(T+M)+M+row)*D+d,k,d<D)
    else:
        j=row-T
        k=tl.load(K0+(bh*M+j)*D+d,d<D,0).to(Pool.dtype.element_ty)
        k=k/tl.maximum(libdevice.sqrt(tl.sum(k*k,0)),1.e-6)
        tl.store(Pool+(bh*(T+M)+j)*D+d,k,d<D)


@tr.jit
def select(Score,Score0,Birth0,IDs,DropSlot,SF,BF,T:tl.constexpr,M:tl.constexpr,
           BM:tl.constexpr,START:tl.constexpr,BT:tl.constexpr):
    bh=tl.program_id(0).to(tl.int64)
    r=tl.arange(0,BM);c=tl.arange(0,BT);valid=r<M
    ids=r;sc=tl.load(Score0+bh*M+r,valid,0).to(tl.float32)
    born=tl.load(Birth0+bh*M+r,valid,0).to(tl.int64)
    tl.store(IDs+bh*(T+1)*M+r,ids,valid)
    low=tl.min(tl.where(valid,sc,float('inf')),0)
    age=tl.min(tl.where(valid & (sc==low),born,9223372036854775807),0)
    slot=tl.min(tl.where(valid & (sc==low) & (born==age),r,2147483647),0)
    for ch in range(tl.cdiv(T,BT)):
        t=ch*BT+c
        scores=tl.load(Score+bh*T+t,t<T,float('-inf')).to(tl.float32)
        tl.store(DropSlot+bh*T+t,-1,t<T)
        pos=0
        while pos<BT:
            nxt=tl.min(tl.where((c>=pos)&(t<T)&(scores>=low),c,BT),0)
            if nxt>pos:
                # All rejected tokens in this interval share an unchanged dictionary.
                tl.store(IDs+(bh*(T+1)+t[:,None]+1)*M+r[None,:],
                         tl.broadcast_to(ids[None,:],(BT,BM)),
                         (c[:,None]>=pos)&(c[:,None]<nxt)&(t[:,None]<T)&valid[None,:])
            if nxt<BT:
                ti=ch*BT+nxt
                score=tl.sum(tl.where(c==nxt,scores,0.),0)
                ids=tl.where(r==slot,M+ti,ids)
                sc=tl.where(r==slot,score,sc);born=tl.where(r==slot,START+ti,born)
                tl.store(DropSlot+bh*T+ti,slot)
                tl.store(IDs+(bh*(T+1)+ti+1)*M+r,ids,valid)
                low=tl.min(tl.where(valid,sc,float('inf')),0)
                age=tl.min(tl.where(valid & (sc==low),born,9223372036854775807),0)
                slot=tl.min(tl.where(valid & (sc==low) & (born==age),r,2147483647),0)
            pos=nxt+1
    tl.store(SF+bh*M+r,sc,valid);tl.store(BF+bh*M+r,born,valid)


@tr.jit
def effective_beta(Beta,Score,BE,N:tl.constexpr,GATE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    b=tl.load(Beta+i,i<N,0).to(tl.float32)
    if GATE:
        s=tl.load(Score+i,i<N,0).to(tl.float32)
        b=b*2*tl.sigmoid(s)
    tl.store(BE+i,b,i<N)


@tr.jit
def fifo_ids(Score0,Birth0,IDs,DropSlot,SF,BF,T:tl.constexpr,M:tl.constexpr,
             BM:tl.constexpr,START:tl.constexpr,BT:tl.constexpr):
    bh=tl.program_id(1).to(tl.int64)
    t=tl.program_id(0)*BT+tl.arange(0,BT);r=tl.arange(0,BM)
    first=(r-START%M+M)%M
    # For t <= first the initial atom is still alive. Use nonnegative division.
    last=first[None,:]+((tl.maximum(t[:,None]-1-first[None,:],0))//M)*M
    replaced=t[:,None]>first[None,:]
    ids=tl.where(replaced,M+last,r[None,:])
    tl.store(IDs+(bh*(T+1)+t[:,None])*M+r[None,:],ids,(t[:,None]<=T)&(r[None,:]<M))
    tl.store(DropSlot+bh*T+t,(START+t)%M,t<T)
    if tl.program_id(0)==0:
        done=T>first
        final=first+(tl.maximum(T-1-first,0)//M)*M
        sc=tl.load(Score0+bh*M+r,r<M,0)
        born=tl.load(Birth0+bh*M+r,r<M,0)
        tl.store(SF+bh*M+r,tl.where(done,0.,sc),r<M)
        tl.store(BF+bh*M+r,tl.where(done,START+final,born),r<M)


@tr.jit
def direct_sweep(gram,rhs,M:tl.constexpr,BM:tl.constexpr):
    # Symmetric elimination solves the RHS directly, avoiding inverse @ near-ones.
    r=tl.arange(0,BM)
    a=gram
    sol=rhs
    for j in tl.static_range(M):
        col=tl.sum(tl.where(r[None,:]==j,a,0.),1)
        pivot=tl.sum(tl.where(r==j,col,0.),0)
        c=col/pivot
        bj=tl.sum(tl.where(r==j,sol,0.),0)/pivot
        sol=tl.where(r==j,bj,sol-col*bj)
        a=a-col[:,None]*c[None,:]
        a=tl.where(r[:,None]==j,c[None,:],tl.where(r[None,:]==j,c[:,None],a))
        a=tl.where((r[:,None]==j)&(r[None,:]==j),-1./pivot,a)
    return sol


@tr.jit
def factor_ldlt(gram,M:tl.constexpr,BM:tl.constexpr):
    r=tl.arange(0,BM)
    a=gram
    for j in tl.static_range(M):
        col=tl.reshape(tl.gather(a,tl.full((BM,1),j,tl.int32),1),(BM,))
        pivot=tl.sum(tl.gather(col,tl.full((1,),j,tl.int32),0),0)
        lower=col/pivot
        a=tl.where((r[:,None]>j)&(r[None,:]>j),a-col[:,None]*lower[None,:],a)
        a=tl.where((r[:,None]>j)&(r[None,:]==j),lower[:,None],a)
    return a


@tr.jit
def solve_ldlt_multi(factor,rhs,M:tl.constexpr,BM:tl.constexpr,BC:tl.constexpr):
    r=tl.arange(0,BM);x=rhs
    for j in tl.static_range(M):
        col=tl.reshape(tl.gather(factor,tl.full((BM,1),j,tl.int32),1),(BM,))
        pivot=tl.sum(tl.gather(col,tl.full((1,),j,tl.int32),0),0)
        val=tl.reshape(tl.gather(x,tl.full((1,BC),j,tl.int32),0),(BC,))
        x=tl.where(r[:,None]>j,x-col[:,None]*val[None,:],tl.where(r[:,None]==j,val[None,:]/pivot,x))
    for step in tl.static_range(M):
        j=M-1-step
        row=tl.reshape(tl.gather(factor,tl.full((1,BM),j,tl.int32),0),(BM,))
        val=tl.reshape(tl.gather(x,tl.full((1,BC),j,tl.int32),0),(BC,))
        x=tl.where(r[:,None]<j,x-row[:,None]*val[None,:],x)
    return x


@tr.jit
def geometry_values(QN,Pool,IDs,DropSlot,t,bh,T:tl.constexpr,M:tl.constexpr,D:tl.constexpr,
                    BM:tl.constexpr,BD:tl.constexpr,TAU:tl.constexpr,RIDGE:tl.constexpr):
    r=tl.arange(0,BM);d=tl.arange(0,BD);valid=r<M
    old=tl.load(IDs+(bh*(T+1)+t)*M+r,valid,0)
    kept=tl.load(IDs+(bh*(T+1)+t+1)*M+r,valid,0)
    slot=tl.load(DropSlot+bh*T+t)
    dropped=tl.where(slot>=0,tl.sum(tl.where(r==slot,old,0),0),M+t)
    base=bh*(T+M)
    if Pool.dtype.element_ty == tl.float64:
        gram=tl.zeros((BM,BM),tl.float64)
        cross=tl.zeros((BM,),tl.float64);hist=tl.zeros((BM,),tl.float64)
        read=tl.zeros((BM,),tl.float64);qnew=tl.full((),0,tl.float64)
        for j in range(D):
            nk=tl.load(Pool+(base+kept)*D+j,valid,0)
            ok=tl.load(Pool+(base+old)*D+j,valid,0)
            drop=tl.load(Pool+(base+dropped)*D+j)
            new=tl.load(Pool+(base+M+t)*D+j)
            q=tl.load(QN+(bh*T+t)*D+j)
            gram+=nk[:,None]*nk[None,:]
            cross+=nk*drop;hist+=ok*new;read+=ok*q;qnew+=q*new
    else:
        mask=valid[:,None] & (d[None,:]<D)
        nk=tl.load(Pool+(base+kept[:,None])*D+d[None,:],mask,0)
        ok=tl.load(Pool+(base+old[:,None])*D+d[None,:],mask,0)
        drop=tl.load(Pool+(base+dropped)*D+d,d<D,0)
        new=tl.load(Pool+(base+M+t)*D+d,d<D,0)
        q=tl.load(QN+(bh*T+t)*D+d,d<D,0)
        # Unit-normalized Gaussian kernel, evaluated from centered squared distances.
        # Centering avoids subtracting nearly equal O(1) dot products for duplicate keys.
        anchor=tl.sum(tl.where(r[:,None]==0,nk,0.),0)
        centered=nk-anchor[None,:]
        norms=tl.sum(centered*centered,1)
        gram=-.5*tl.maximum(norms[:,None]+norms[None,:]-2*tl.dot(centered,tl.trans(centered),input_precision='tf32x3'),0.)
        cross=-.5*tl.sum((nk-drop[None,:])*(nk-drop[None,:]),1)
        hist=-.5*tl.sum((ok-new[None,:])*(ok-new[None,:]),1)
        read=-.5*tl.sum((ok-q[None,:])*(ok-q[None,:]),1)
        qnew=-.5*tl.sum((q-new)*(q-new),0)
        # The normalization clamp permits sub-unit/zero vectors. Preserve their
        # dot-kernel semantics; ordinary unit vectors need no noisy correction.
        nn=tl.sum(nk*nk,1);on=tl.sum(ok*ok,1)
        dn0=tl.sum(drop*drop,0);nn0=tl.sum(new*new,0);qn0=tl.sum(q*q,0)
        cn=tl.where(nn<.99999,nn-1.,0.);co=tl.where(on<.99999,on-1.,0.)
        cd=tl.where(dn0<.99999,dn0-1.,0.);cnew=tl.where(nn0<.99999,nn0-1.,0.)
        cq=tl.where(qn0<.99999,qn0-1.,0.)
        gram+=.5*(cn[:,None]+cn[None,:]);cross+=.5*(cn+cd)
        hist+=.5*(co+cnew);read+=.5*(co+cq);qnew+=.5*(cq+cnew)
    both=valid[:,None]&valid[None,:]
    if Pool.dtype.element_ty == tl.float64:
        gram=gram-1.;cross=cross-1.;hist=hist-1.;read=read-1.;qnew=qnew-1.
    gram=tl.where(both,libdevice.exp(gram/TAU),0.)
    cross=tl.where(valid,libdevice.exp(cross/TAU),0.)
    hist=tl.where(valid,libdevice.exp(hist/TAU),0.)
    read=tl.where(valid,libdevice.exp(read/TAU),0.)
    qnew=libdevice.exp(qnew/TAU)
    # Gram/RHS/kernel tapes are represented in BF16; normalized vectors stay FP32.
    gram=gram.to(tl.bfloat16).to(tl.float32)
    cross=cross.to(tl.bfloat16).to(tl.float32)
    hist=hist.to(tl.bfloat16).to(tl.float32)
    read=read.to(tl.bfloat16).to(tl.float32)
    qnew=qnew.to(tl.bfloat16).to(tl.float32)
    return old,kept,dropped,gram,cross,hist,read,qnew


@tr.jit
def forward(QN,Pool,IDs,DropSlot,P,H,R,N,
            T:tl.constexpr,M:tl.constexpr,D:tl.constexpr,BM:tl.constexpr,BD:tl.constexpr,
            TAU:tl.constexpr,RIDGE:tl.constexpr,SPARSE:tl.constexpr=False):
    t,bh=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64)
    if SPARSE:
        ct=(t//32)*32+tl.arange(0,32)
        ds=tl.load(DropSlot+bh*T+ct,ct<T,-1)
        enabled=tl.sum((ds>=0).to(tl.int32),0)>4
    else:
        enabled=True
    if enabled:
        r=tl.arange(0,BM)
        old,kept,dropped,gram,cross,hist,read,qnew=geometry_values(
            QN,Pool,IDs,DropSlot,t,bh,T,M,D,BM,BD,TAU,RIDGE)
        p=direct_sweep(gram+tl.where(r[:,None]==r[None,:],RIDGE,0.),cross,M,BM)
        idx=bh*T+t
        tl.store(P+idx*M+r,p,r<M);tl.store(H+idx*M+r,hist,r<M)
        tl.store(R+idx*M+r,read,r<M);tl.store(N+idx,qnew)


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
    grad=tl.load(DKF+bh*M*D+x,row<M,0).to(DPool.dtype.element_ty)
    tl.atomic_add(DPool+(bh*(T+M)+source)*D+d,grad,row<M,sem='relaxed')


@tr.jit
def backward(QN,Pool,IDs,DropSlot,P,H,R,N,DP,DH,DR,DN,DQN,DPool,
             T:tl.constexpr,M:tl.constexpr,D:tl.constexpr,BM:tl.constexpr,BD:tl.constexpr,
             TAU:tl.constexpr,RIDGE:tl.constexpr,SPARSE:tl.constexpr=False):
    t,bh=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64)
    if SPARSE:
        ct=(t//32)*32+tl.arange(0,32)
        ds=tl.load(DropSlot+bh*T+ct,ct<T,-1)
        enabled=tl.sum((ds>=0).to(tl.int32),0)>4
    else:
        enabled=True
    if enabled:
        r=tl.arange(0,BM);d=tl.arange(0,BD);valid=r<M;both=valid[:,None]&valid[None,:]
        idx=bh*T+t;base=bh*(T+M)
        old,kept,dropped,gram,cross,unused_h,unused_r,unused_n=geometry_values(
            QN,Pool,IDs,DropSlot,t,bh,T,M,D,BM,BD,TAU,RIDGE)
        p=tl.load(P+idx*M+r,valid,0)
        dp=tl.load(DP+idx*M+r,valid,0).to(Pool.dtype.element_ty)
        z=direct_sweep(gram+tl.where(r[:,None]==r[None,:],RIDGE,0.),dp,M,BM)
        hist=tl.load(H+idx*M+r,valid,0)
        read=tl.load(R+idx*M+r,valid,0)
        qnew=tl.load(N+idx)
        dc=z*cross/TAU
        dmat=-(z[:,None]*p[None,:]+p[:,None]*z[None,:])*gram/TAU
        dh=tl.load(DH+idx*M+r,valid,0).to(Pool.dtype.element_ty)*hist/TAU
        dr=tl.load(DR+idx*M+r,valid,0).to(Pool.dtype.element_ty)*read/TAU
        dn=tl.load(DN+idx).to(Pool.dtype.element_ty)*qnew/TAU
        if Pool.dtype.element_ty == tl.float64:
            for j in range(D):
                nk=tl.load(Pool+(base+kept)*D+j,valid,0)
                ok=tl.load(Pool+(base+old)*D+j,valid,0)
                drop=tl.load(Pool+(base+dropped)*D+j)
                new=tl.load(Pool+(base+M+t)*D+j)
                q=tl.load(QN+idx*D+j)
                dkeep=tl.sum(dmat*nk[None,:],1)+dc*drop
                dold=dh*new+dr*q
                tl.atomic_add(DPool+(base+kept)*D+j,dkeep,valid,sem='relaxed')
                tl.atomic_add(DPool+(base+old)*D+j,dold,valid,sem='relaxed')
                tl.atomic_add(DPool+(base+dropped)*D+j,tl.sum(dc*nk,0),sem='relaxed')
                tl.atomic_add(DPool+(base+M+t)*D+j,tl.sum(dh*ok,0)+dn*q,sem='relaxed')
                tl.store(DQN+idx*D+j,tl.sum(dr*ok,0)+dn*new)
        else:
            mask=valid[:,None] & (d[None,:]<D)
            nk=tl.load(Pool+(base+kept[:,None])*D+d[None,:],mask,0)
            ok=tl.load(Pool+(base+old[:,None])*D+d[None,:],mask,0)
            drop=tl.load(Pool+(base+dropped)*D+d,d<D,0)
            new=tl.load(Pool+(base+M+t)*D+d,d<D,0)
            q=tl.load(QN+idx*D+d,d<D,0)
            anchor=tl.sum(tl.where(r[:,None]==0,nk,0.),0)
            centered=nk-anchor[None,:]
            dkeep=tl.dot(dmat,centered,input_precision='tf32x3')-tl.sum(dmat,1)[:,None]*centered+dc[:,None]*(drop[None,:]-nk)
            dold=dh[:,None]*(new[None,:]-ok)+dr[:,None]*(q[None,:]-ok)
            dkeep+=tl.where((tl.sum(nk*nk,1)<.99999)[:,None],(tl.sum(dmat,1)+dc)[:,None]*nk,0.)
            dold+=tl.where((tl.sum(ok*ok,1)<.99999)[:,None],(dh+dr)[:,None]*ok,0.)
            ddrop=tl.sum(dc[:,None]*(nk-drop[None,:]),0)+tl.where(tl.sum(drop*drop,0)<.99999,tl.sum(dc,0)*drop,0.)
            dnew=tl.sum(dh[:,None]*(ok-new[None,:]),0)+dn*(q-new)+tl.where(tl.sum(new*new,0)<.99999,(tl.sum(dh,0)+dn)*new,0.)
            dq=tl.sum(dr[:,None]*(ok-q[None,:]),0)+dn*(new-q)+tl.where(tl.sum(q*q,0)<.99999,(tl.sum(dr,0)+dn)*q,0.)
            tl.atomic_add(DPool+(base+kept[:,None])*D+d[None,:],dkeep,mask,sem='relaxed')
            tl.atomic_add(DPool+(base+old[:,None])*D+d[None,:],dold,mask,sem='relaxed')
            tl.atomic_add(DPool+(base+dropped)*D+d,ddrop,d<D,sem='relaxed')
            tl.atomic_add(DPool+(base+M+t)*D+d,dnew,d<D,sem='relaxed')
            tl.store(DQN+idx*D+d,dq,d<D)


@tr.jit
def normalize_backward(Q,K,K0,QN,Pool,DQN,DPool,DQ,DK,DK0,
                       T:tl.constexpr,M:tl.constexpr,D:tl.constexpr,BD:tl.constexpr):
    row,bh=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64)
    d=tl.arange(0,BD)
    if row<T:
        off=(bh*T+row)*D+d
        q=tl.load(Q+off,d<D,0).to(Pool.dtype.element_ty)
        n=tl.load(QN+off,d<D,0)
        grad=tl.load(DQN+off,d<D,0)
        norm=libdevice.sqrt(tl.sum(q*q,0))
        dq=(grad-tl.where(norm>=1.e-6,n*tl.sum(n*grad,0),0.))/tl.maximum(norm,1.e-6)
        tl.store(DQ+off,dq,d<D)
        raw=tl.load(K+off,d<D,0).to(Pool.dtype.element_ty)
        poff=(bh*(T+M)+M+row)*D+d
        n=tl.load(Pool+poff,d<D,0);grad=tl.load(DPool+poff,d<D,0)
        norm=libdevice.sqrt(tl.sum(raw*raw,0))
        dk=(grad-tl.where(norm>=1.e-6,n*tl.sum(n*grad,0),0.))/tl.maximum(norm,1.e-6)
        tl.store(DK+off,dk,d<D)
    else:
        j=row-T;off=(bh*M+j)*D+d;poff=(bh*(T+M)+j)*D+d
        raw=tl.load(K0+off,d<D,0).to(Pool.dtype.element_ty)
        n=tl.load(Pool+poff,d<D,0);grad=tl.load(DPool+poff,d<D,0)
        norm=libdevice.sqrt(tl.sum(raw*raw,0))
        dk=(grad-tl.where(norm>=1.e-6,n*tl.sum(n*grad,0),0.))/tl.maximum(norm,1.e-6)
        tl.store(DK0+off,dk,d<D)
