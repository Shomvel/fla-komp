"""Autograd interface. All numerical forward/backward stages run in Triton."""
import torch
import triton as tr
from torch.autograd.function import once_differentiable
from . import geometry as geo
from . import recurrence as rec
from . import chunk_backward as cb
from . import hierarchical as hs
from . import versioned as vg


class _Exact(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,g,beta,k0,s0,scores,scores0,births0,learned,gate,start,chunk,tau,ridge,fp64,scalar):
        bh,t,d=q.shape;m=k0.shape[1];dv=v.shape[-1]
        c=chunk;nc=tr.cdiv(t,c);bm=max(16,tr.next_power_of_2(m));bd=max(16,tr.next_power_of_2(d));bv=32;nv=tr.cdiv(dv,bv)
        dtype=torch.float64 if fp64 else torch.float32
        def alloc(shape,dt=torch.float32,zero=False):
            return (torch.zeros if zero else torch.empty)(shape,device=q.device,dtype=dt)
        qn=alloc((bh,t,d),dtype);pool=alloc((bh,t+m,d),dtype)
        ids=alloc((bh,t+1,m),torch.int32);drop=alloc((bh,t),torch.int32)
        sf=alloc((bh,m));bf=alloc((bh,m),torch.int64);be=alloc((bh,t))
        p,h,r=[alloc((bh,t,m),dtype) for _ in range(3)];n=alloc((bh,t),dtype)
        geo.normalize[(t+m,bh)](q,k,k0,qn,pool,t,m,d,bd,num_warps=4,enable_fp_fusion=False)
        if learned:
            geo.select[(bh,)](scores,scores0,births0,ids,drop,sf,bf,t,m,bm,start,64,num_warps=4)
        else:
            geo.fifo_ids[(tr.cdiv(t+1,32),bh)](scores0,births0,ids,drop,sf,bf,t,m,bm,start,32,num_warps=4)
        geo.effective_beta[(tr.cdiv(bh*t,256),)](beta,scores,be,bh*t,gate,256)
        reuse=learned and not fp64 and m<=32
        cross=alloc((bh,t,m)) if reuse else alloc((0,))
        geo.forward[(t,bh)](qn,pool,ids,drop,p,h,r,n,t,m,d,bm,bd,tau,ridge,reuse,num_warps=4,enable_fp_fusion=False)
        if reuse:
            vg.prepare[(t,bh)](qn,pool,ids,drop,p,h,r,n,cross,t,m,d,bm,bd,tau,ridge,num_warps=4,enable_fp_fusion=False)
            vg.project[(tr.cdiv(t,32),bh)](qn,pool,ids,drop,cross,p,p,p,t,m,d,bm,bd,tau,ridge,False,num_warps=4,enable_fp_fusion=False)
        u=alloc((bh,t,2,m));gc=alloc((bh,nc,c,m))
        a=alloc((bh,nc,2*c,m));bmap=alloc((bh,nc,2*c,c))
        f=alloc((bh,nc,m,m));fv=alloc((bh,nc,m,c))
        qm=alloc((bh,nc,c,m));om=alloc((bh,nc,c,c))
        rec.maps[(nc,bh)](p,h,r,n,be,g,drop,u,gc,a,bmap,f,fv,qm,om,t,m,c,nc,bm,scalar,num_warps=4)
        entry=alloc((bh,nc,m,dv));z=alloc((bh,t,2,dv));final=alloc((bh,m,dv))
        if nc>=32:
            source=alloc((bh,nc,m,dv))
            hs.sources[(nc,bh,nv)](fv,v,source,t,m,dv,c,nc,bm,bv,num_warps=4)
            hs.scan(f,source,s0,entry,final)
            del source
            hs.updates[(nc,bh,nv)](a,bmap,v,entry,z,t,m,dv,c,nc,bm,bv,num_warps=4)
        else:
            rec.state_scan[(bh,nv)](a,bmap,f,fv,v,s0,entry,z,final,t,m,dv,c,nc,bm,bv,num_warps=4)
        del a,bmap,fv
        out=alloc((bh,t,dv),v.dtype);keys=alloc((bh,m,d))
        cb.outputs[(nc,bh,nv)](qm,om,entry,v,out,t,m,dv,c,nc,bm,bv,num_warps=4)
        geo.final_keys[(tr.cdiv(m*d,256),bh)](pool,ids,keys,t,m,d,256)
        ctx.save_for_backward(q,k,v,g,beta,k0,s0,scores,qn,pool,ids,drop,p,h,r,n,be,u,gc,f,qm,entry,z,cross)
        ctx.options=(gate,tau,ridge,bm,bd,bv,c,nc,reuse)
        ctx.mark_non_differentiable(sf,bf)
        return out,keys,final,sf,bf

    @staticmethod
    @once_differentiable
    def backward(ctx,do,dkf,dsf,dscores_final,dbirths_final):
        q,k,v,g,beta,k0,s0,scores,qn,pool,ids,drop,p,h,r,n,be,u,gc,f,qm,entry,z,cross=ctx.saved_tensors
        gate,tau,ridge,bm,bd,bv,c,nc,reuse=ctx.options
        bh,t,d=q.shape;m=k0.shape[1];dv=v.shape[-1];nv=tr.cdiv(dv,bv)
        def alloc(shape,dt=torch.float32,zero=False):
            return (torch.zeros if zero else torch.empty)(shape,device=q.device,dtype=dt)
        do=torch.zeros_like(v) if do is None else do.contiguous()
        dkf=torch.zeros((bh,m,d),device=q.device) if dkf is None else dkf.contiguous()
        dsf=torch.zeros((bh,m,dv),device=q.device) if dsf is None else dsf.contiguous()
        dp,dh,dr,dg=[alloc((bh,t,m),zero=True) for _ in range(4)]
        dn=alloc((bh,t));db=alloc((bh,t),zero=True)
        dvalue=torch.empty_like(v);ds0=torch.empty_like(s0)
        j=alloc((bh,nc,m,dv));exit=alloc((bh,nc,m,dv))
        di,dc=[alloc((bh,t,dv)) for _ in range(2)]
        cb.boundary_source[(nc,bh,nv)](qm,do,j,t,m,dv,c,nc,bm,bv,num_warps=4)
        if nc>=32:
            hs.scan(f,j,dsf,exit,ds0,reverse=True)
        else:
            cb.boundary_scan[(bh,nv)](f,j,dsf,exit,ds0,m,dv,nc,bm,bv,num_warps=4)
        del j
        cb.local_adjoint[(nc,bh,nv)](p,h,r,n,be,g,drop,do,exit,di,dc,dvalue,t,m,dv,c,nc,bm,bv,num_warps=4)
        cb.local_gradients_scan[(nc,bh,m*nv)](g,u,z,entry,exit,h,r,be,drop,do,di,dc,dp,dh,dr,dg,db,
                                       t,m,dv,c,nc,bv,nv,num_warps=4)
        cb.scalar_gradients[(t,bh)](v,z,do,di,dn,db,t,dv,tr.next_power_of_2(dv),num_warps=4)
        del exit,di,dc
        dqn=alloc(qn.shape,qn.dtype);dpool=alloc(pool.shape,pool.dtype,zero=True)
        geo.final_keys_backward[(tr.cdiv(m*d,256),bh)](dkf,ids,dpool,t,m,d,256)
        geo.backward[(t,bh)](qn,pool,ids,drop,p,h,r,n,dp,dh,dr,dn,dqn,dpool,
                              t,m,d,bm,bd,tau,ridge,reuse,num_warps=4,enable_fp_fusion=False)
        if reuse:
            vg.project[(tr.cdiv(t,32),bh)](qn,pool,ids,drop,cross,p,dp,dpool,t,m,d,bm,bd,tau,ridge,True,num_warps=4,enable_fp_fusion=False)
            vg.read_backward[(t,bh)](qn,pool,ids,drop,h,r,n,dh,dr,dn,dqn,dpool,t,m,d,bm,bd,tau,num_warps=4,enable_fp_fusion=False)
        dq,dk,dk0=torch.empty_like(q),torch.empty_like(k),torch.empty_like(k0)
        geo.normalize_backward[(t+m,bh)](q,k,k0,qn,pool,dqn,dpool,dq,dk,dk0,t,m,d,bd,
                                         num_warps=4,enable_fp_fusion=False)
        dbeta,dscore=torch.empty_like(beta),torch.empty_like(scores)
        rec.beta_backward[(tr.cdiv(bh*t,256),)](db,beta,scores,dbeta,dscore,bh*t,gate,256)
        return dq,dk,dvalue,dg.to(g.dtype),dbeta,dk0,ds0,dscore,None,None,None,None,None,None,None,None,None,None


def exact_komp(q,k,v,log_decay,beta,initial_state,*,eviction='fifo',scores=None,
               initial_scores=None,initial_births=None,start_position=0,compute_chunk=16,
               score_write_gate=None,temperature=.5,ridge=1e-4,projection_dtype="float32"):
    """Per-token Gaussian KOMP with FIFO or input-score learned deletion.

    q/k [B,T,H,K], v [B,T,H,V], log_decay [B,T,H,M] or [B,T,H],
    beta/scores [B,T,H], state=(keys [B,H,M,K], coefficients [B,H,M,V]).
    Returns (output, (keys,coefficients), {scores,births}). Carry metadata and
    start_position when streaming learned states. beta is the BASE strength;
    learned applies 2*sigmoid(scores) by default, matching the experiment.

    Algorithmic compression is ALWAYS C=1. compute_chunk (16 or 32) changes only
    execution scheduling. No approximate CG or delayed dictionary compression.
    FP32 normalized geometry/Gram solve and state by default; optional float64
    projection for diagnostics. Tensor-core tf32x3 products. First-order gradients
    only. CUDA, M<=64, K/V<=128. Chunk boundary state and low-rank update tape;
    no per-token M x V states or M x M geometry tape.
    """
    if projection_dtype not in ('float32','float64'):
        raise ValueError('projection_dtype must be float32 or float64')
    if not q.is_cuda:
        raise ValueError('exact_komp requires a CUDA device')
    if q.ndim!=4 or k.shape!=q.shape or v.shape[:3]!=q.shape[:3]:
        raise ValueError('q/k/v must have compatible [B,T,H,D] shapes')
    b,t,h,d=q.shape;dv=v.shape[-1];k0,s0=initial_state;m=k0.shape[-2]
    if t<1 or not 1<=m<=64 or not 1<=d<=128 or not 1<=dv<=128:
        raise ValueError('supported sizes: T>=1, 1<=M<=64, 1<=K,V<=128')
    if compute_chunk not in (16,32) or eviction not in ('fifo','learned'):
        raise ValueError('compute_chunk must be 16/32; eviction must be fifo/learned')
    if temperature<=0 or ridge<=0:
        raise ValueError('temperature and ridge must be positive')
    if q.dtype not in (torch.float32,torch.bfloat16) or k.dtype!=q.dtype or v.dtype not in (torch.float32,torch.bfloat16):
        raise ValueError('q/k/v support FP32 and BF16')
    if k0.shape!=(b,h,m,d) or s0.shape!=(b,h,m,dv) or beta.shape!=(b,t,h):
        raise ValueError('initial_state or beta has the wrong shape')
    scalar=log_decay.shape==(b,t,h)
    if scalar:
        log_decay=log_decay[...,None].expand(b,t,h,m)
    if log_decay.shape!=(b,t,h,m):
        raise ValueError('log_decay must be [B,T,H] or [B,T,H,M]')
    learned=eviction=='learned'
    gate=learned if score_write_gate is None else bool(score_write_gate)
    if scores is None:
        if learned or gate:raise ValueError('learned selection or write gate requires scores')
        scores=torch.zeros_like(beta)
    if scores.shape!=(b,t,h):raise ValueError('scores must be [B,T,H]')
    if initial_scores is None:initial_scores=torch.zeros((b,h,m),device=q.device)
    if initial_births is None:
        initial_births=torch.arange(start_position-m,start_position,device=q.device,dtype=torch.int64)[None,None].expand(b,h,m)
    if initial_scores.shape!=(b,h,m) or initial_births.shape!=(b,h,m):
        raise ValueError('initial_scores/births must be [B,H,M]')
    tensors=(q,k,v,log_decay,beta,k0,s0,scores,initial_scores,initial_births)
    if any(x.device!=q.device for x in tensors):raise ValueError('all inputs must share one CUDA device')
    def time_flat(x):
        return x.transpose(1,2).contiguous().reshape(b*h,t,*x.shape[3:])
    out,kf,sf,sc,born=_Exact.apply(time_flat(q),time_flat(k),time_flat(v),time_flat(log_decay),time_flat(beta),
        k0.contiguous().reshape(b*h,m,d),s0.contiguous().reshape(b*h,m,dv),time_flat(scores),
        initial_scores.detach().float().contiguous().reshape(b*h,m),initial_births.detach().long().contiguous().reshape(b*h,m),
        learned,gate,start_position,compute_chunk,float(temperature),float(ridge),projection_dtype=="float64",scalar)
    return (out.reshape(b,h,t,dv).transpose(1,2),
            (kf.reshape(b,h,m,d),sf.reshape(b,h,m,dv)),
            {'scores':sc.reshape(b,h,m),'births':born.reshape(b,h,m)})
