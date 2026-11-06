"""Rank-two chunk solve, chunk state propagation and exact recurrence adjoint.

Like MesaNet, chunk boundaries retain a state scan. Projection systems and
intra-chunk maps are computed in parallel across time. The first implementation
stores per-token decayed states for a direct analytical reverse recurrence.
"""
import triton as tr
import triton.language as tl


@tr.jit
def maps(P,H,BE,G,DropSlot,U,GC,A,B,F,FV,
         T:tl.constexpr,M:tl.constexpr,C:tl.constexpr,NC:tl.constexpr,BM:tl.constexpr):
    chunk,bh=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64)
    c=tl.arange(0,C);m=tl.arange(0,BM);z=tl.arange(0,2*C)
    t=chunk*C+c;tz=chunk*C+z//2;rank=z%2
    g=tl.load(G+(bh*T+t[:,None])*M+m[None,:],(t[:,None]<T)&(m[None,:]<M),0).to(tl.float32)
    gc=tl.cumsum(g,0)
    tl.store(GC+((bh*NC+chunk)*C+c[:,None])*M+m[None,:],gc,m[None,:]<M)
    gz=tl.gather(gc,tl.broadcast_to((z//2)[:,None],(2*C,BM)),0)
    glast=tl.sum(tl.where(c[:,None]==C-1,gc,0.),0)
    p=tl.load(P+(bh*T+tz[:,None])*M+m[None,:],(tz[:,None]<T)&(m[None,:]<M),0).to(tl.float32)
    h=tl.load(H+(bh*T+tz[:,None])*M+m[None,:],(tz[:,None]<T)&(m[None,:]<M),0).to(tl.float32)
    beta=tl.load(BE+bh*T+tz,tz<T,0).to(tl.float32)
    slot=tl.load(DropSlot+bh*T+tz,tz<T,-1)
    accept=slot>=0
    e=(m[None,:]==slot[:,None]).to(tl.float32)
    u=tl.where(rank[:,None]==0,tl.where(accept[:,None],p-e,0.),tl.where(accept[:,None],e,p))
    vf=tl.where(rank[:,None]==0,e,-beta[:,None]*h)
    valid=(tz[:,None]<T)&(m[None,:]<M)
    u=tl.where(valid,u,0.);vf=tl.where(valid,vf,0.)
    tl.store(U+((bh*T+tz[:,None])*2+rank[:,None])*M+m[None,:],u,valid)
    lower=(z[:,None]//2>z[None,:]//2)&(tz[:,None]<T)&(tz[None,:]<T)
    coupling=tl.zeros((2*C,2*C),tl.float32)
    for j in range(M):
        uj=tl.sum(tl.where(m[None,:]==j,u,0.),1)
        vj=tl.sum(tl.where(m[None,:]==j,vf,0.),1)
        gj=tl.sum(tl.where(m[None,:]==j,gz,0.),1)
        decay=tl.exp(tl.where(lower,gj[:,None]-gj[None,:],0.))
        coupling=coupling+tl.where(lower,vj[:,None]*uj[None,:]*decay,0.)
    ident=(z[:,None]==z[None,:]).to(tl.float32)
    inv=ident+coupling
    power=coupling
    for _ in range(4 if C==16 else 5):
        power=tl.dot(power,power,input_precision='tf32x3')
        inv=inv+tl.dot(power,inv,input_precision='tf32x3')
    w=vf*tl.exp(gz)
    write=((rank[:,None]==1)&(z[:,None]//2==c[None,:])).to(tl.float32)*beta[:,None]
    amap=tl.dot(inv,w,input_precision='tf32x3')
    bmap=tl.dot(inv,write,input_precision='tf32x3')
    last=tl.trans(u*tl.exp(glast[None,:]-gz))
    f=tl.dot(last,amap,input_precision='tf32x3')+(m[:,None]==m[None,:]).to(tl.float32)*tl.exp(glast)[:,None]
    fv=tl.dot(last,bmap,input_precision='tf32x3')
    idx=bh*NC+chunk
    tl.store(A+idx*(2*C)*M+z[:,None]*M+m[None,:],amap,m[None,:]<M)
    tl.store(B+idx*(2*C)*C+z[:,None]*C+c[None,:],bmap)
    tl.store(F+idx*M*M+m[:,None]*M+m[None,:],f,(m[:,None]<M)&(m[None,:]<M))
    tl.store(FV+idx*M*C+m[:,None]*C+c[None,:],fv,m[:,None]<M)


@tr.jit
def state_scan(A,B,F,FV,V,S0,Entry,Z,SF,
               T:tl.constexpr,M:tl.constexpr,DV:tl.constexpr,C:tl.constexpr,NC:tl.constexpr,BM:tl.constexpr,BV:tl.constexpr):
    bh,tile=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64)
    m=tl.arange(0,BM);c=tl.arange(0,C);z=tl.arange(0,2*C);v=tile*BV+tl.arange(0,BV)
    mask=(m[:,None]<M)&(v[None,:]<DV)
    state=tl.load(S0+bh*M*DV+m[:,None]*DV+v[None,:],mask,0).to(tl.float32)
    for chunk in range(NC):
        idx=bh*NC+chunk;t=chunk*C+c
        tl.store(Entry+idx*M*DV+m[:,None]*DV+v[None,:],state,mask)
        values=tl.load(V+(bh*T+t[:,None])*DV+v[None,:],(t[:,None]<T)&(v[None,:]<DV),0).to(tl.float32)
        a=tl.load(A+idx*(2*C)*M+z[:,None]*M+m[None,:],m[None,:]<M,0)
        b=tl.load(B+idx*(2*C)*C+z[:,None]*C+c[None,:])
        update=tl.dot(a,state,input_precision='tf32x3')+tl.dot(b,values,input_precision='tf32x3')
        tz=chunk*C+z//2
        tl.store(Z+((bh*T+tz[:,None])*2+z[:,None]%2)*DV+v[None,:],update,(tz[:,None]<T)&(v[None,:]<DV))
        f=tl.load(F+idx*M*M+m[:,None]*M+m[None,:],(m[:,None]<M)&(m[None,:]<M),0)
        fv=tl.load(FV+idx*M*C+m[:,None]*C+c[None,:],m[:,None]<M,0)
        state=tl.dot(f,state,input_precision='tf32x3')+tl.dot(fv,values,input_precision='tf32x3')
    tl.store(SF+bh*M*DV+m[:,None]*DV+v[None,:],state,mask)


@tr.jit
def expand_states(GC,U,Z,Entry,X,T:tl.constexpr,M:tl.constexpr,DV:tl.constexpr,C:tl.constexpr,
                  NC:tl.constexpr,BV:tl.constexpr,NV:tl.constexpr):
    chunk,bh,sv=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64),tl.program_id(2).to(tl.int64)
    slot=sv//NV;tile=sv%NV
    c=tl.arange(0,C);z=tl.arange(0,2*C);v=tile*BV+tl.arange(0,BV)
    t=chunk*C+c;tz=chunk*C+z//2;idx=bh*NC+chunk
    gc=tl.load(GC+(idx*C+c)*M+slot)
    gz=tl.gather(gc,z//2,0)
    u=tl.load(U+((bh*T+tz)*2+z%2)*M+slot,tz<T,0)
    causal=(c[:,None]>z[None,:]//2)&(t[:,None]<T)&(tz[None,:]<T)
    w=tl.where(causal,tl.exp(tl.where(causal,gc[:,None]-gz[None,:],0.))*u[None,:],0.)
    values=tl.load(Z+((bh*T+tz[:,None])*2+z[:,None]%2)*DV+v[None,:],(tz[:,None]<T)&(v[None,:]<DV),0)
    entry=tl.load(Entry+(idx*M+slot)*DV+v,v<DV,0)
    x=tl.exp(gc)[:,None]*entry[None,:]+tl.dot(w,values,input_precision='tf32x3')
    tl.store(X+((bh*T+t[:,None])*M+slot)*DV+v[None,:],x,(t[:,None]<T)&(v[None,:]<DV))


@tr.jit
def outputs(X,H,R,N,BE,V,O,T:tl.constexpr,M:tl.constexpr,DV:tl.constexpr,BM:tl.constexpr,BV:tl.constexpr):
    t,bh,tile=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64),tl.program_id(2).to(tl.int64)
    m=tl.arange(0,BM);v=tile*BV+tl.arange(0,BV);idx=bh*T+t
    x=tl.load(X+(idx*M+m[:,None])*DV+v[None,:],(m[:,None]<M)&(v[None,:]<DV),0)
    h=tl.load(H+idx*M+m,m<M,0).to(tl.float32)
    r=tl.load(R+idx*M+m,m<M,0).to(tl.float32)
    n=tl.load(N+idx).to(tl.float32);beta=tl.load(BE+idx)
    val=tl.load(V+idx*DV+v,v<DV,0).to(tl.float32)
    innovation=beta*(val-tl.sum(h[:,None]*x,0))
    out=tl.sum(r[:,None]*x,0)+n*innovation
    tl.store(O+idx*DV+v,out,v<DV)


@tr.jit
def backward(X,P,H,R,N,BE,G,V,DropSlot,DO,DSF,DP,DH,DR,DN,DB,DG,DVout,DS0,
             T:tl.constexpr,M:tl.constexpr,DV:tl.constexpr,BM:tl.constexpr,BV:tl.constexpr):
    bh,tile=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64)
    m=tl.arange(0,BM);v=tile*BV+tl.arange(0,BV)
    mask=(m[:,None]<M)&(v[None,:]<DV)
    adj=tl.load(DSF+bh*M*DV+m[:,None]*DV+v[None,:],mask,0).to(tl.float32)
    for step in range(T):
        t=T-1-step;idx=bh*T+t
        x=tl.load(X+(idx*M+m[:,None])*DV+v[None,:],mask,0)
        p=tl.load(P+idx*M+m,m<M,0).to(tl.float32)
        h=tl.load(H+idx*M+m,m<M,0).to(tl.float32)
        r=tl.load(R+idx*M+m,m<M,0).to(tl.float32)
        n=tl.load(N+idx).to(tl.float32);beta=tl.load(BE+idx)
        g=tl.load(G+idx*M+m,m<M,0).to(tl.float32)
        do=tl.load(DO+idx*DV+v,v<DV,0).to(tl.float32)
        val=tl.load(V+idx*DV+v,v<DV,0).to(tl.float32)
        slot=tl.load(DropSlot+idx)
        e=(m==slot).to(tl.float32)
        a=tl.where(slot>=0,p-e,0.);b=tl.where(slot>=0,e,p)
        residual=val-tl.sum(h[:,None]*x,0)
        innovation=beta*residual
        deleted=tl.sum(e[:,None]*x,0)
        di=n*do+tl.sum(b[:,None]*adj,0)
        dc=tl.sum(a[:,None]*adj,0)
        dx=adj+r[:,None]*do[None,:]+e[:,None]*dc[None,:]-h[:,None]*(beta*di)[None,:]
        dp=tl.sum(adj*tl.where(slot>=0,deleted,innovation)[None,:],1)
        dh=-tl.sum(x*(beta*di)[None,:],1)
        dr=tl.sum(x*do[None,:],1)
        dg=tl.sum(dx*x,1)
        tl.atomic_add(DP+idx*M+m,dp.to(tl.float64),m<M,sem='relaxed')
        tl.atomic_add(DH+idx*M+m,dh.to(tl.float64),m<M,sem='relaxed')
        tl.atomic_add(DR+idx*M+m,dr.to(tl.float64),m<M,sem='relaxed')
        tl.atomic_add(DG+idx*M+m,dg.to(tl.float64),m<M,sem='relaxed')
        tl.atomic_add(DN+idx,tl.sum(innovation*do,0).to(tl.float64),sem='relaxed')
        tl.atomic_add(DB+idx,tl.sum(residual*di,0).to(tl.float64),sem='relaxed')
        tl.store(DVout+idx*DV+v,beta*di,v<DV)
        adj=tl.exp(g)[:,None]*dx
    tl.store(DS0+bh*M*DV+m[:,None]*DV+v[None,:],adj,mask)


@tr.jit
def beta_backward(DB,Beta,Score,DBeta,DScore,N:tl.constexpr,GATE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    grad=tl.load(DB+i,i<N,0)
    if GATE:
        beta=tl.load(Beta+i,i<N,0).to(tl.float32)
        s=tl.load(Score+i,i<N,0).to(tl.float32)
        sig=tl.sigmoid(s)
        tl.store(DBeta+i,grad*2*sig,i<N)
        tl.store(DScore+i,grad*beta*2*sig*(1-sig),i<N)
    else:
        tl.store(DBeta+i,grad,i<N)
        tl.store(DScore+i,0.,i<N)
