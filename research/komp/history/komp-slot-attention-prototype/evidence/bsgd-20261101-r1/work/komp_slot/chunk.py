"""Rank-two chunk recurrence: parallel UT construction, Triton chunk scan and backward."""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def inverse_kernel(L, Out, N: tl.constexpr, STEPS: tl.constexpr):
    batch = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, N).to(tl.int64)
    lower = tl.load(L+batch*N*N+i[:,None]*N+i[None,:])
    inv = (i[:,None] == i[None,:]).to(tl.float32)
    # Merge diagonal inverses: [[A,0],[-B,D]]^-1 has D^-1 B A^-1
    # in its lower-left block. Avoid powers of L: their large intermediate
    # values can catastrophically cancel for repeated slot replacement.
    for step in tl.static_range(STEPS):
        width = 1 << step
        group = 2*width
        cross = tl.where((i[:,None]//group == i[None,:]//group)
                         & (i[:,None]%group >= width)
                         & (i[None,:]%group < width),lower,0.)
        update = tl.dot(tl.dot(inv,cross,input_precision='ieee'),inv,input_precision='ieee')
        inv = inv+update
    tl.store(Out+batch*N*N+i[:,None]*N+i[None,:],inv)


class _UnitLowerInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lower):
        lower = lower.contiguous()
        n = lower.shape[-1]
        result = torch.empty_like(lower)
        with torch.cuda.device(lower.device):
            inverse_kernel[(lower.numel()//(n*n),)](lower,result,n,int(math.log2(n)),num_warps=4)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        inv, = ctx.saved_tensors
        return (inv.transpose(-1,-2) @ grad @ inv.transpose(-1,-2)).tril(-1)


@triton.jit
def chunk_fwd_kernel(OS, OV, FS, FV, Value, Init, Output, Final, Tape,
                     NC: tl.constexpr, C: tl.constexpr, M: tl.constexpr, V: tl.constexpr,
                     BM: tl.constexpr, BV: tl.constexpr, SAVE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V,BV)
    bh, tile = pid//nv, pid%nv
    m = tl.arange(0,BM).to(tl.int64)
    c = tl.arange(0,C).to(tl.int64)
    v = tile*BV+tl.arange(0,BV).to(tl.int64)
    smask=(m[:,None]<M)&(v[None,:]<V)
    state=tl.load(Init+bh*M*V+m[:,None]*V+v[None,:],smask,other=0)
    if SAVE:
        tl.store(Tape+bh*(NC+1)*M*V+m[:,None]*V+v[None,:],state,smask)
    for block in range(NC):
        idx=bh*NC+block
        os=tl.load(OS+idx*C*M+c[:,None]*M+m[None,:],m[None,:]<M,other=0)
        ov=tl.load(OV+idx*C*C+c[:,None]*C+c[None,:])
        fs=tl.load(FS+idx*M*M+m[:,None]*M+m[None,:],(m[:,None]<M)&(m[None,:]<M),other=0)
        fv=tl.load(FV+idx*M*C+m[:,None]*C+c[None,:],m[:,None]<M,other=0)
        value=tl.load(Value+idx*C*V+c[:,None]*V+v[None,:],v[None,:]<V,other=0)
        out=tl.dot(os,state,input_precision='ieee')+tl.dot(ov,value,input_precision='ieee')
        state=tl.dot(fs,state,input_precision='ieee')+tl.dot(fv,value,input_precision='ieee')
        tl.store(Output+idx*C*V+c[:,None]*V+v[None,:],out,v[None,:]<V)
        if SAVE:
            tl.store(Tape+(bh*(NC+1)+block+1)*M*V+m[:,None]*V+v[None,:],state,smask)
    tl.store(Final+bh*M*V+m[:,None]*V+v[None,:],state,smask)


@triton.jit
def chunk_bwd_kernel(OS, OV, FS, FV, Value, Tape, DO, DF, DOS, DOV, DFS, DFV, DV, DI,
                     BH: tl.constexpr, NC: tl.constexpr, C: tl.constexpr, M: tl.constexpr,
                     V: tl.constexpr, BM: tl.constexpr, BV: tl.constexpr):
    pid=tl.program_id(0).to(tl.int64)
    nv=tl.cdiv(V,BV)
    bh,tile=pid//nv,pid%nv
    m=tl.arange(0,BM).to(tl.int64)
    c=tl.arange(0,C).to(tl.int64)
    v=tile*BV+tl.arange(0,BV).to(tl.int64)
    smask=(m[:,None]<M)&(v[None,:]<V)
    ds=tl.load(DF+bh*M*V+m[:,None]*V+v[None,:],smask,other=0)
    for step in range(NC):
        block=NC-1-step
        idx=bh*NC+block
        partial=tile*BH*NC+idx
        prev=tl.load(Tape+(bh*(NC+1)+block)*M*V+m[:,None]*V+v[None,:],smask,other=0)
        value=tl.load(Value+idx*C*V+c[:,None]*V+v[None,:],v[None,:]<V,other=0)
        do=tl.load(DO+idx*C*V+c[:,None]*V+v[None,:],v[None,:]<V,other=0)
        dos=tl.dot(do,tl.trans(prev),input_precision='ieee')
        dov=tl.dot(do,tl.trans(value),input_precision='ieee')
        dfs=tl.dot(ds,tl.trans(prev),input_precision='ieee')
        dfv=tl.dot(ds,tl.trans(value),input_precision='ieee')
        tl.store(DOS+partial*C*M+c[:,None]*M+m[None,:],dos,m[None,:]<M)
        tl.store(DOV+partial*C*C+c[:,None]*C+c[None,:],dov)
        tl.store(DFS+partial*M*M+m[:,None]*M+m[None,:],dfs,(m[:,None]<M)&(m[None,:]<M))
        tl.store(DFV+partial*M*C+m[:,None]*C+c[None,:],dfv,m[:,None]<M)
        os=tl.load(OS+idx*C*M+c[:,None]*M+m[None,:],m[None,:]<M,other=0)
        ov=tl.load(OV+idx*C*C+c[:,None]*C+c[None,:])
        fs=tl.load(FS+idx*M*M+m[:,None]*M+m[None,:],(m[:,None]<M)&(m[None,:]<M),other=0)
        fv=tl.load(FV+idx*M*C+m[:,None]*C+c[None,:],m[:,None]<M,other=0)
        dv=tl.dot(tl.trans(ov),do,input_precision='ieee')+tl.dot(tl.trans(fv),ds,input_precision='ieee')
        ds=tl.dot(tl.trans(os),do,input_precision='ieee')+tl.dot(tl.trans(fs),ds,input_precision='ieee')
        tl.store(DV+idx*C*V+c[:,None]*V+v[None,:],dv,v[None,:]<V)
    tl.store(DI+bh*M*V+m[:,None]*V+v[None,:],ds,smask)


class _ChunkScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, os, ov, fs, fv, value, initial, save):
        os,ov,fs,fv,value,initial=[x.contiguous() for x in (os,ov,fs,fv,value,initial)]
        batch,heads,nc,c,m=os.shape
        width=value.shape[-1]
        output=torch.empty_like(value)
        final=torch.empty_like(initial)
        tape=value.new_empty((batch,heads,nc+1,m,width) if save else (1,))
        with torch.cuda.device(value.device):
            chunk_fwd_kernel[(batch*heads*triton.cdiv(width,32),)](
                os,ov,fs,fv,value,initial,output,final,tape,nc,c,m,width,
                max(16,triton.next_power_of_2(m)),32,save,num_warps=4,
            )
        ctx.save_for_backward(os,ov,fs,fv,value,tape)
        ctx.set_materialize_grads(False)
        return output,final

    @staticmethod
    @once_differentiable
    def backward(ctx, do, df):
        os,ov,fs,fv,value,tape=ctx.saved_tensors
        batch,heads,nc,c,m=os.shape
        width=value.shape[-1]
        nv=triton.cdiv(width,32)
        do=torch.zeros_like(value) if do is None else do.contiguous()
        df=value.new_zeros(batch,heads,m,width) if df is None else df.contiguous()
        grads=[x.new_empty(nv,*x.shape) for x in (os,ov,fs,fv)]
        dv=torch.empty_like(value)
        di=torch.empty_like(df)
        with torch.cuda.device(value.device):
            chunk_bwd_kernel[(batch*heads*nv,)](
                os,ov,fs,fv,value,tape,do,df,*grads,dv,di,
                batch*heads,nc,c,m,width,max(16,triton.next_power_of_2(m)),32,num_warps=4,
            )
        return *(g.sum(0) for g in grads),dv,di,None


def chunk_coefficients(h,p,r,alpha,beta,slots,chunk_size=16,inverse_backend='triton'):
    """Build all chunk transfer maps independently, with division-free zero-safe decay."""
    if chunk_size not in (16,32):
        raise ValueError('chunk_size must be 16 or 32')
    batch,t,heads,m=h.shape
    c=chunk_size
    nc=triton.cdiv(t,c)
    pad=nc*c-t
    def blocks(x,value=0):
        if x.ndim==3:
            return F.pad(x.transpose(1,2),(0,pad),value=value).reshape(batch,heads,nc,c)
        return F.pad(x.transpose(1,2),(0,0,0,pad),value=value).reshape(batch,heads,nc,c,x.shape[-1])
    h,p,r=map(blocks,(h,p,r))
    a=blocks(alpha,1)
    beta=blocks(beta)
    e=F.one_hot(blocks(slots).long(),m).to(h.dtype)
    valid=(torch.arange(nc*c,device=h.device)<t).reshape(1,1,nc,c,1)
    p=torch.where(valid,p,e)
    x=torch.stack((p-e,-beta[...,None]*e),dim=-2).flatten(-3,-2)
    y=torch.stack((e,h),dim=-2).flatten(-3,-2)
    write=beta[...,None]*e
    i=torch.arange(c,device=h.device)
    strict=i[:,None]>i[None,:]
    causal=i[:,None]>=i[None,:]
    decay=torch.where(strict,a[..., :,None],torch.ones_like(a[..., :,None])).cumprod(-2)
    prefix=a.cumprod(-1)
    lower=(y@x.transpose(-1,-2))*(decay*strict).repeat_interleave(2,-2).repeat_interleave(2,-1)
    if inverse_backend=='triton':
        inverse=_UnitLowerInverse.apply(lower)
    elif inverse_backend=='torch':
        eye=torch.eye(2*c,device=h.device,dtype=h.dtype).expand_as(lower)
        inverse=torch.linalg.solve_triangular(eye-lower,eye,upper=False,unitriangular=True)
    else:
        raise ValueError('Unknown inverse backend')
    ys=prefix.repeat_interleave(2,-1)[...,None]*y
    yv=(y@write.transpose(-1,-2))*(decay*strict).repeat_interleave(2,-2)
    zs,zv=inverse@ys,inverse@yv
    rx=(r@x.transpose(-1,-2))*(decay*causal).repeat_interleave(2,-1)
    rv=(r@write.transpose(-1,-2))*(decay*causal)
    os=prefix[...,None]*r+rx@zs
    ov=rv+rx@zv
    endx=(x*decay[..., -1,:].repeat_interleave(2,-1)[...,None]).transpose(-1,-2)
    eye=torch.eye(m,device=h.device,dtype=h.dtype)
    fs=prefix[..., -1,None,None]*eye+endx@zs
    fv=(write*decay[..., -1,:,None]).transpose(-1,-2)+endx@zv
    return os,ov,fs,fv


def chunk_slot(h,p,r,v,alpha,beta,slots,initial_values,chunk_size=16,backend='triton'):
    """Chunk recurrence with FP32 CUDA Triton inverse/scan, or float64 torch reference.

    Only chunk boundary states are saved by the scan. Transfer-map construction
    uses differentiable PyTorch GEMMs. No per-token value-state loop is executed.
    """
    if backend not in ('triton','torch'):
        raise ValueError('backend must be triton or torch')
    if backend=='triton' and (h.device.type!='cuda' or any(x.dtype!=torch.float32 for x in
                                                       (h,p,r,v,alpha,beta,initial_values))):
        raise ValueError('Triton chunk core requires FP32 CUDA inputs')
    os,ov,fs,fv=chunk_coefficients(h,p,r,alpha,beta,slots,chunk_size,backend)
    batch,t,heads,width=v.shape
    nc=os.shape[2]
    values=F.pad(v.transpose(1,2),(0,0,0,nc*chunk_size-t)).reshape(batch,heads,nc,chunk_size,width)
    if backend=='triton':
        save=torch.is_grad_enabled() and any(x.requires_grad for x in (os,ov,fs,fv,values,initial_values))
        output,state=_ChunkScan.apply(os,ov,fs,fv,values,initial_values,save)
    else:
        state=initial_values
        outputs=[]
        for block in range(nc):
            outputs.append(os[:,:,block]@state+ov[:,:,block]@values[:,:,block])
            state=fs[:,:,block]@state+fv[:,:,block]@values[:,:,block]
        output=torch.stack(outputs,2)
    return output.reshape(batch,heads,nc*chunk_size,width)[:,:,:t].transpose(1,2).contiguous(),state
