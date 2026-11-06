"""Rank-two chunk solve, chunk state propagation and exact recurrence adjoint.

Like MesaNet, chunk boundaries retain a state scan. Projection systems and
intra-chunk maps are computed in parallel across time. Only chunk boundary states and rank-two updates are retained.
Reverse boundary scan and chunk-local gradients follow the GDN decomposition.
"""
import triton as tr
import triton.language as tl


@tr.jit
def maps(P,H,R,N,BE,G,DropSlot,U,GC,A,B,F,FV,Qmap,Omap,
         T:tl.constexpr,M:tl.constexpr,C:tl.constexpr,NC:tl.constexpr,BM:tl.constexpr,SCALAR:tl.constexpr,STORE_GC:tl.constexpr=True):
    chunk,bh=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64)
    c=tl.arange(0,C);m=tl.arange(0,BM);z=tl.arange(0,2*C)
    t=chunk*C+c;tz=chunk*C+z//2;rank=z%2
    g=tl.load(G+(bh*T+t[:,None])*M+m[None,:],(t[:,None]<T)&(m[None,:]<M),0).to(tl.float32)
    gc=tl.cumsum(g,0)
    if STORE_GC:
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
    if SCALAR:
        gz0=tl.sum(tl.where(m[None,:]==0,gz,0.),1)
        coupling=tl.dot(vf,tl.trans(u),input_precision='tf32x3')
        coupling=tl.where(lower,coupling*tl.exp(tl.where(lower,gz0[:,None]-gz0[None,:],0.)),0.)
    else:
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
    for _ in range(3 if C==16 else 4):
        power=tl.dot(power,power,input_precision='tf32x3')
        inv=inv+tl.dot(power,inv,input_precision='tf32x3')
    w=vf*tl.exp(gz)
    write=((rank[:,None]==1)&(z[:,None]//2==c[None,:])).to(tl.float32)*beta[:,None]
    amap=tl.dot(inv,w,input_precision='tf32x3')
    bmap=tl.dot(inv,write,input_precision='tf32x3')
    last=tl.trans(u*tl.exp(glast[None,:]-gz))
    f=tl.dot(last,amap,input_precision='tf32x3')+(m[:,None]==m[None,:]).to(tl.float32)*tl.exp(glast)[:,None]
    fv=tl.dot(last,bmap,input_precision='tf32x3')
    # Contract the per-token read with the rank-two WY representation.
    # O = Qmap @ chunk_entry + Omap @ values; no token M x V tape.
    ht=tl.load(H+(bh*T+t[:,None])*M+m[None,:],(t[:,None]<T)&(m[None,:]<M),0).to(tl.float32)
    rt=tl.load(R+(bh*T+t[:,None])*M+m[None,:],(t[:,None]<T)&(m[None,:]<M),0).to(tl.float32)
    nt=tl.load(N+bh*T+t,t<T,0).to(tl.float32)
    bt=tl.load(BE+bh*T+t,t<T,0)
    read=rt-(nt*bt)[:,None]*ht
    causal=(c[:,None]>z[None,:]//2)&(t[:,None]<T)&(tz[None,:]<T)
    if SCALAR:
        gt0=tl.sum(tl.where(m[None,:]==0,gc,0.),1)
        rw=tl.dot(read,tl.trans(u),input_precision='tf32x3')
        rw=tl.where(causal,rw*tl.exp(tl.where(causal,gt0[:,None]-gz0[None,:],0.)),0.)
    else:
        rw=tl.zeros((C,2*C),tl.float32)
        for j in range(M):
            rj=tl.sum(tl.where(m[None,:]==j,read,0.),1)
            uj=tl.sum(tl.where(m[None,:]==j,u,0.),1)
            gt=tl.sum(tl.where(m[None,:]==j,gc,0.),1)
            gj=tl.sum(tl.where(m[None,:]==j,gz,0.),1)
            decay=tl.exp(tl.where(causal,gt[:,None]-gj[None,:],0.))
            rw+=tl.where(causal,rj[:,None]*uj[None,:]*decay,0.)
    qm=read*tl.exp(gc)+tl.dot(rw,amap,input_precision='tf32x3')
    om=tl.dot(rw,bmap,input_precision='tf32x3')+(c[:,None]==c[None,:]).to(tl.float32)*(nt*bt)[:,None]
    idx=bh*NC+chunk
    tl.store(Qmap+idx*C*M+c[:,None]*M+m[None,:],qm,m[None,:]<M)
    tl.store(Omap+idx*C*C+c[:,None]*C+c[None,:],om)
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
