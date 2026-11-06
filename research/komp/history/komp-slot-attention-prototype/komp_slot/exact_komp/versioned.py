"""Bounded learned dictionary-version reuse: batch RHS and aggregate Gram adjoints.

Sparse 32-token windows reuse an LDL factor within each unchanged dictionary
run. Dense windows use the ordinary parallel token kernels. No host counts or
per-token factor tape is needed, including during CUDA Graph capture.
"""
import triton as tr
import triton.language as tl
from .geometry import geometry_values,factor_ldlt,solve_ldlt_multi


@tr.jit
def prepare(QN,Pool,IDs,Drop,P,H,R,N,Cross,T:tl.constexpr,M:tl.constexpr,D:tl.constexpr,
            BM:tl.constexpr,BD:tl.constexpr,TAU:tl.constexpr,RIDGE:tl.constexpr):
    t,bh=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64)
    ct=(t//32)*32+tl.arange(0,32)
    ds=tl.load(Drop+bh*T+ct,ct<T,-1)
    if tl.sum((ds>=0).to(tl.int32),0)<=4:
        old,kept,dropped,unused,cross,h,r,n=geometry_values(QN,Pool,IDs,Drop,t,bh,T,M,D,BM,BD,TAU,RIDGE)
        m=tl.arange(0,BM);idx=bh*T+t
        tl.store(Cross+idx*M+m,cross,m<M)
        tl.store(H+idx*M+m,h,m<M);tl.store(R+idx*M+m,r,m<M);tl.store(N+idx,n)


@tr.jit
def project(QN,Pool,IDs,Drop,Cross,P,DP,DPool,T:tl.constexpr,M:tl.constexpr,D:tl.constexpr,
            BM:tl.constexpr,BD:tl.constexpr,TAU:tl.constexpr,RIDGE:tl.constexpr,BACKWARD:tl.constexpr):
    window,bh=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64)
    c=tl.arange(0,32);m=tl.arange(0,BM);d=tl.arange(0,BD)
    t=window*32+c;base=bh*(T+M)
    ds=tl.load(Drop+bh*T+t,t<T,-1)
    if tl.sum((ds>=0).to(tl.int32),0)<=4:
        pos=tl.full((),0,tl.int64)
        limit=tl.minimum(32,T-window*32)
        while pos<limit:
            end=tl.min(tl.where((c>pos)&(ds>=0)&(t<T),c,limit),0)
            first=window*32+pos
            old,kept,unused_drop,gram,unused_c,unused_h,unused_r,unused_n=geometry_values(
                QN,Pool,IDs,Drop,first,bh,T,M,D,BM,BD,TAU,RIDGE)
            factor=factor_ldlt(gram+tl.where(m[:,None]==m[None,:],RIDGE,0.),M,BM)
            active=(c>=pos)&(c<end)&(t<T)
            mask=(m[:,None]<M)&active[None,:]
            cross=tl.load(Cross+(bh*T+t[None,:])*M+m[:,None],mask,0)
            if BACKWARD:
                rhs=tl.load(DP+(bh*T+t[None,:])*M+m[:,None],mask,0)
            else:
                rhs=cross
            z=solve_ldlt_multi(factor,rhs,M,BM,32)
            if BACKWARD:
                p=tl.load(P+(bh*T+t[None,:])*M+m[:,None],mask,0)
                dc=z*cross/TAU
                zp=tl.dot(z,tl.trans(p),input_precision='tf32x3')
                dmat=-(zp+tl.trans(zp))*gram/TAU
                nk=tl.load(Pool+(base+kept[:,None])*D+d[None,:],(m[:,None]<M)&(d[None,:]<D),0)
                dropped=tl.load(IDs+(bh*(T+1)+t)*M+tl.maximum(ds,0),active&(ds>=0),0)
                dropped=tl.where(ds>=0,dropped,M+t)
                dk=tl.load(Pool+(base+dropped[:,None])*D+d[None,:],active[:,None]&(d[None,:]<D),0)
                anchor=tl.sum(tl.where(m[:,None]==0,nk,0.),0)
                nc=nk-anchor[None,:];dd=dk-anchor[None,:]
                dkeep=tl.dot(dmat,nc,input_precision='tf32x3')-tl.sum(dmat,1)[:,None]*nc
                dkeep+=tl.dot(dc,dd,input_precision='tf32x3')-tl.sum(dc,1)[:,None]*nc
                dkeep+=tl.where((tl.sum(nk*nk,1)<.99999)[:,None],(tl.sum(dmat,1)+tl.sum(dc,1))[:,None]*nk,0.)
                ddrop=tl.dot(tl.trans(dc),nc,input_precision='tf32x3')-tl.sum(dc,0)[:,None]*dd
                ddrop+=tl.where((tl.sum(dk*dk,1)<.99999)[:,None],tl.sum(dc,0)[:,None]*dk,0.)
                tl.atomic_add(DPool+(base+kept[:,None])*D+d[None,:],dkeep,(m[:,None]<M)&(d[None,:]<D),sem='relaxed')
                tl.atomic_add(DPool+(base+dropped[:,None])*D+d[None,:],ddrop,active[:,None]&(d[None,:]<D),sem='relaxed')
            else:
                tl.store(P+(bh*T+t[None,:])*M+m[:,None],z,mask)
            pos=end


@tr.jit
def read_backward(QN,Pool,IDs,Drop,H,R,N,DH,DR,DN,DQN,DPool,
                  T:tl.constexpr,M:tl.constexpr,D:tl.constexpr,BM:tl.constexpr,BD:tl.constexpr,TAU:tl.constexpr):
    t,bh=tl.program_id(0).to(tl.int64),tl.program_id(1).to(tl.int64)
    ct=(t//32)*32+tl.arange(0,32)
    ds=tl.load(Drop+bh*T+ct,ct<T,-1)
    if tl.sum((ds>=0).to(tl.int32),0)<=4:
        m=tl.arange(0,BM);d=tl.arange(0,BD);idx=bh*T+t;base=bh*(T+M)
        old=tl.load(IDs+(bh*(T+1)+t)*M+m,m<M,0)
        ok=tl.load(Pool+(base+old[:,None])*D+d[None,:],(m[:,None]<M)&(d[None,:]<D),0)
        new=tl.load(Pool+(base+M+t)*D+d,d<D,0)
        q=tl.load(QN+idx*D+d,d<D,0)
        dh=tl.load(DH+idx*M+m,m<M,0)*tl.load(H+idx*M+m,m<M,0)/TAU
        dr=tl.load(DR+idx*M+m,m<M,0)*tl.load(R+idx*M+m,m<M,0)/TAU
        dn=tl.load(DN+idx)*tl.load(N+idx)/TAU
        dold=dh[:,None]*(new[None,:]-ok)+dr[:,None]*(q[None,:]-ok)
        dold+=tl.where((tl.sum(ok*ok,1)<.99999)[:,None],(dh+dr)[:,None]*ok,0.)
        dnew=tl.sum(dh[:,None]*(ok-new[None,:]),0)+dn*(q-new)+tl.where(tl.sum(new*new,0)<.99999,(tl.sum(dh,0)+dn)*new,0.)
        dq=tl.sum(dr[:,None]*(ok-q[None,:]),0)+dn*(new-q)+tl.where(tl.sum(q*q,0)<.99999,(tl.sum(dr,0)+dn)*q,0.)
        tl.atomic_add(DPool+(base+old[:,None])*D+d[None,:],dold,(m[:,None]<M)&(d[None,:]<D),sem='relaxed')
        tl.atomic_add(DPool+(base+M+t)*D+d,dnew,d<D,sem='relaxed')
        tl.store(DQN+idx*D+d,dq,d<D)
