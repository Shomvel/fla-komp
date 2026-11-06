"""Two-level exact affine scan. Only the group boundaries are serial."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def sources(FV,V,U,T:tl.constexpr,M:tl.constexpr,DV:tl.constexpr,C:tl.constexpr,
            NC:tl.constexpr,BM:tl.constexpr,BV:tl.constexpr):
    ch,bh,tile=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64),tl.program_id(2).to(tl.int64)
    m=tl.arange(0,BM);c=tl.arange(0,C);v=tile*BV+tl.arange(0,BV)
    idx=bh*NC+ch;t=ch*C+c
    a=tl.load(FV+idx*M*C+m[:,None]*C+c[None,:],m[:,None]<M,0)
    val=tl.load(V+(bh*T+t[:,None])*DV+v[None,:],(t[:,None]<T)&(v[None,:]<DV),0).to(tl.float32)
    u=tl.dot(a,val,input_precision='tf32x3')
    tl.store(U+idx*M*DV+m[:,None]*DV+v[None,:],u,(m[:,None]<M)&(v[None,:]<DV))


@tr.jit
def summarize(F,U,FG,UG,M:tl.constexpr,DV:tl.constexpr,NC:tl.constexpr,
              NG:tl.constexpr,G:tl.constexpr,BM:tl.constexpr,BV:tl.constexpr,REVERSE:tl.constexpr):
    group,bh,tile=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64),tl.program_id(2).to(tl.int64)
    m=tl.arange(0,BM);v=tile*BV+tl.arange(0,BV)
    mask=(m[:,None]<M)&(v[None,:]<DV)
    fm=(m[:,None]<M)&(m[None,:]<M)
    prod=(m[:,None]==m[None,:]).to(tl.float32)
    state=tl.zeros((BM,BV),tl.float32)
    for j in range(G):
        seq=group*G+j
        if seq<NC:
            ch=NC-1-seq if REVERSE else seq
            idx=bh*NC+ch
            f=tl.load(F+idx*M*M+m[:,None]*M+m[None,:],fm,0)
            if REVERSE:f=tl.trans(f)
            u=tl.load(U+idx*M*DV+m[:,None]*DV+v[None,:],mask,0)
            state=tl.dot(f,state,input_precision='tf32x3')+u
            prod=tl.dot(f,prod,input_precision='tf32x3')
    ix=bh*NG+group
    if tile==0:
        tl.store(FG+ix*M*M+m[:,None]*M+m[None,:],prod,fm)
    tl.store(UG+ix*M*DV+m[:,None]*DV+v[None,:],state,mask)


@tr.jit
def coarse(FG,UG,Initial,GroupEntry,Final,M:tl.constexpr,DV:tl.constexpr,
           NG:tl.constexpr,BM:tl.constexpr,BV:tl.constexpr):
    bh,tile=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64)
    m=tl.arange(0,BM);v=tile*BV+tl.arange(0,BV)
    mask=(m[:,None]<M)&(v[None,:]<DV)
    state=tl.load(Initial+bh*M*DV+m[:,None]*DV+v[None,:],mask,0).to(tl.float32)
    for group in range(NG):
        ix=bh*NG+group
        tl.store(GroupEntry+ix*M*DV+m[:,None]*DV+v[None,:],state,mask)
        f=tl.load(FG+ix*M*M+m[:,None]*M+m[None,:],(m[:,None]<M)&(m[None,:]<M),0)
        u=tl.load(UG+ix*M*DV+m[:,None]*DV+v[None,:],mask,0)
        state=tl.dot(f,state,input_precision='tf32x3')+u
    tl.store(Final+bh*M*DV+m[:,None]*DV+v[None,:],state,mask)


@tr.jit
def expand(F,U,GroupEntry,Entry,M:tl.constexpr,DV:tl.constexpr,NC:tl.constexpr,
           NG:tl.constexpr,G:tl.constexpr,BM:tl.constexpr,BV:tl.constexpr,REVERSE:tl.constexpr):
    group,bh,tile=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64),tl.program_id(2).to(tl.int64)
    m=tl.arange(0,BM);v=tile*BV+tl.arange(0,BV)
    mask=(m[:,None]<M)&(v[None,:]<DV)
    state=tl.load(GroupEntry+(bh*NG+group)*M*DV+m[:,None]*DV+v[None,:],mask,0)
    for j in range(G):
        seq=group*G+j
        if seq<NC:
            ch=NC-1-seq if REVERSE else seq;idx=bh*NC+ch
            tl.store(Entry+idx*M*DV+m[:,None]*DV+v[None,:],state,mask)
            f=tl.load(F+idx*M*M+m[:,None]*M+m[None,:],(m[:,None]<M)&(m[None,:]<M),0)
            if REVERSE:f=tl.trans(f)
            u=tl.load(U+idx*M*DV+m[:,None]*DV+v[None,:],mask,0)
            state=tl.dot(f,state,input_precision='tf32x3')+u


@tr.jit
def updates(A,B,V,Entry,Z,T:tl.constexpr,M:tl.constexpr,DV:tl.constexpr,C:tl.constexpr,
            NC:tl.constexpr,BM:tl.constexpr,BV:tl.constexpr):
    ch,bh,tile=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64),tl.program_id(2).to(tl.int64)
    m=tl.arange(0,BM);c=tl.arange(0,C);z=tl.arange(0,2*C);v=tile*BV+tl.arange(0,BV)
    idx=bh*NC+ch;t=ch*C+c;tz=ch*C+z//2
    state=tl.load(Entry+idx*M*DV+m[:,None]*DV+v[None,:],(m[:,None]<M)&(v[None,:]<DV),0)
    a=tl.load(A+idx*2*C*M+z[:,None]*M+m[None,:],m[None,:]<M,0)
    b=tl.load(B+idx*2*C*C+z[:,None]*C+c[None,:])
    val=tl.load(V+(bh*T+t[:,None])*DV+v[None,:],(t[:,None]<T)&(v[None,:]<DV),0).to(tl.float32)
    update=tl.dot(a,state,input_precision='tf32x3')+tl.dot(b,val,input_precision='tf32x3')
    tl.store(Z+((bh*T+tz[:,None])*2+z[:,None]%2)*DV+v[None,:],update,(tz[:,None]<T)&(v[None,:]<DV))


def scan(f,u,initial,entry,final,reverse=False,group=16):
    bh,nc,m,_=f.shape;dv=initial.shape[-1]
    ng=tr.cdiv(nc,group);bm=max(16,tr.next_power_of_2(m));bv=32;nv=tr.cdiv(dv,bv)
    fg=torch.empty((bh,ng,m,m),device=f.device,dtype=torch.float32)
    ug=torch.empty((bh,ng,m,dv),device=f.device,dtype=torch.float32)
    ge=torch.empty_like(ug)
    summarize[(ng,bh,nv)](f,u,fg,ug,m,dv,nc,ng,group,bm,bv,reverse,num_warps=4)
    coarse[(bh,nv)](fg,ug,initial,ge,final,m,dv,ng,bm,bv,num_warps=4)
    expand[(ng,bh,nv)](f,u,ge,entry,m,dv,nc,ng,group,bm,bv,reverse,num_warps=4)
