"""Value-column decay with learned top-M compression and a bounded stable backend.

The native coordinate transform commutes with key-only selection and projection.
Hard ranking has no gradient. Scores learn via the mixer's beta write gate.
State is in physical slot order, as in block_komp_fast. This pretraining API does
not expose carried scores for streaming across separate calls.
"""
import math
import torch
from torch.nn import functional as F
from fla.ops.komp.block_fast import block_komp_fast
from fla.ops.komp.block_flex import _assign, _gather_rows
from fla.ops.komp.value_decay_hybrid import _Scan


def _check(q,k,v,g,beta,state,scores,initial_scores,chunk_size,temperature,ridge,evict):
    if evict != 'learned': raise ValueError('This operator implements learned eviction only')
    if q.ndim != 4 or k.shape != q.shape or v.shape[:3] != q.shape[:3] or g.shape != v.shape:
        raise ValueError('q/k [B,T,H,K], v/g [B,T,H,V] required')
    b,t,h,d=q.shape; keys,coef=state; m=keys.shape[-2]
    if keys.shape!=(b,h,m,d) or coef.shape!=(b,h,m,v.shape[-1]) or beta.shape!=(b,t,h):
        raise ValueError('Invalid state or beta shape')
    if scores is None or initial_scores is None or scores.shape!=(b,t,h) or initial_scores.shape!=(b,h,m):
        raise ValueError('Learned selection requires token and initial scores')
    if min(b,t,h,d,m,v.shape[-1])<1 or chunk_size not in (16,32,64,128) or m>64:
        raise ValueError('Positive native dimensions, M<=64 and C=16/32/64/128 required')
    if not math.isfinite(temperature) or temperature<=0 or not math.isfinite(ridge) or ridge<=0:
        raise ValueError('Positive finite temperature and ridge required')
    for x in (q,k,v,g,beta,keys,coef,scores,initial_scores):
        if not x.is_floating_point() or x.device!=q.device or not bool(torch.isfinite(x).all()):
            raise ValueError('All inputs must be finite floating tensors on one device')
    if bool((g>0).any()) or bool(((beta<0)|(beta>2)).any()):
        raise ValueError('g<=0 and learned write strength beta in [0,2] required')


def _kern(a,b,tau): return ((a@b.transpose(-1,-2)-1)/tau).exp()


@torch.no_grad()
def _routing(scores, initial_scores, m, chunk_size):
    """Build key-only physical dictionaries; no value computation or gradients."""
    b,t,h=scores.shape;dev=scores.device
    old=torch.arange(m,device=dev).expand(b,h,m)
    old_scores=initial_scores.detach()
    tables=[]
    for start in range(0,t,chunk_size):
        n=min(chunk_size,t-start)
        new=torch.arange(m+start,m+start+n,device=dev).expand(b,h,n)
        pool=torch.cat((old,new),-1)
        rank=torch.cat((old_scores,scores[:,start:start+n].transpose(1,2).detach()),-1)
        recent=pool.argsort(-1,descending=True)
        order=recent.gather(-1,rank.gather(-1,recent).argsort(-1,descending=True,stable=True))
        keep=order[...,:m].sort(-1).values;drop=order[...,m:].sort(-1).values
        slot=_assign(keep,m,(start+torch.arange(n,device=dev))%m,n)
        keep=keep.gather(-1,slot.argsort(-1))
        old=pool.gather(-1,keep);old_scores=rank.gather(-1,keep)
        tables.append((start,n,pool,keep,drop,old))
    return tables


def recurrent_value_learned(q,k,v,g,beta,initial_state,*,chunk_size=128,temperature=.5,ridge=1e-4,
                            evict='learned',scores=None,initial_scores=None):
    """Independent token recurrence; FP64 oracle. No coordinate rescaling."""
    _check(q,k,v,g,beta,initial_state,scores,initial_scores,chunk_size,temperature,ridge,evict)
    dtype=torch.float64 if any(x.dtype==torch.float64 for x in (q,k,v,g,beta,*initial_state)) else torch.float32
    with torch.autocast(q.device.type,enabled=False):
        qn,kn=[F.normalize(x.to(dtype),dim=-1,eps=1e-6) for x in (q,k)]
        keys=F.normalize(initial_state[0].to(dtype),dim=-1,eps=1e-6);coef=initial_state[1].to(dtype)
        m=keys.shape[-2];eye=torch.eye(m,device=q.device,dtype=dtype);out=[]
        for start,n,_,keep,drop,_ in _routing(scores,initial_scores,m,chunk_size):
            dictionary,state=keys,coef
            for t in range(start,start+n):
                state=state*g[:,t].to(dtype).exp()[...,None,:]
                kt=kn[:,t,:,None,:]
                err=v[:,t].to(dtype)-(state*_kern(dictionary,kt,temperature)).sum(-2)
                dictionary=torch.cat((dictionary,kt),-2)
                state=torch.cat((state,(beta[:,t].to(dtype)[...,None]*err)[...,None,:]),-2)
                out.append((state*_kern(dictionary,qn[:,t,:,None,:],temperature)).sum(-2))
            kept,evicted=_gather_rows(dictionary,keep),_gather_rows(dictionary,drop)
            projection=torch.linalg.solve(_kern(kept,kept,temperature)+ridge*eye,_kern(kept,evicted,temperature))
            keys,coef=kept,_gather_rows(state,keep)+projection@_gather_rows(state,drop)
        return torch.stack(out,1).to(v.dtype),(keys,coef)


def _stable(q,k,v,g,beta,initial_state,scores,initial_scores,chunk_size,temperature,ridge):
    """Inverse-free value scan; learned dictionaries/projections built in PyTorch.

    All full chunks use one scan call; the final short chunk uses a second call.
    Unlike coordinate scaling this stays bounded for large channel rate spreads.
    """
    b,t,h,d=q.shape;dv=v.shape[-1];m=initial_state[0].shape[-2]
    qn,kn=[F.normalize(x.float(),dim=-1,eps=1e-6).transpose(1,2) for x in (q,k)]
    pool=torch.cat((F.normalize(initial_state[0].float(),dim=-1,eps=1e-6),kn),-2)
    coef=initial_state[1].float();tables=_routing(scores,initial_scores,m,chunk_size)
    groups=[tables] if t%chunk_size==0 else [tables[:-1],tables[-1:]]
    outs=[]
    for group in groups:
        if not group: continue
        start,c=group[0][:2];nc=len(group);length=nc*c
        indices=torch.stack([r[2] for r in group],2)
        dictionary=_gather_rows(pool[:,:,None].expand(b,h,nc,m+t,d),indices)
        keep=torch.stack([r[3] for r in group],2);drop=torch.stack([r[4] for r in group],2)
        kept,evicted=_gather_rows(dictionary,keep),_gather_rows(dictionary,drop)
        projection=torch.linalg.solve(_kern(kept,kept,temperature)+ridge*torch.eye(m,device=q.device),_kern(kept,evicted,temperature))
        # Scatter the projection and identity into the original [old,new] order.
        p=torch.zeros(b,h,nc,m,m+c,device=q.device)
        p=p.scatter(-1,drop[...,None,:].expand(b,h,nc,m,c),projection)
        p=p.scatter(-1,keep[...,None,:].expand(b,h,nc,m,m),torch.eye(m,device=q.device).expand(b,h,nc,m,m))
        qc=qn[:,:,start:start+length].reshape(b,h,nc,c,d)
        kc=kn[:,:,start:start+length].reshape(b,h,nc,c,d)
        ak,aq=_kern(kc,dictionary,temperature),_kern(qc,dictionary,temperature)
        def packed(x,width):return x[:,start:start+length].float().transpose(1,2).reshape(b*h,nc,c,width).contiguous()
        o,coef=_Scan.apply(ak.reshape(b*h,nc,c,m+c).contiguous(),aq.reshape(b*h,nc,c,m+c).contiguous(),
            p.reshape(b*h,nc,m,m+c).contiguous(),packed(v,dv),packed(g,dv),
            beta[:,start:start+length].float().transpose(1,2).reshape(b*h,nc,c).contiguous(),coef.reshape(b*h,m,dv).contiguous())
        coef=coef.reshape(b,h,m,dv);outs.append(o.reshape(b,h,length,dv))
    keys=_gather_rows(pool,tables[-1][-1])
    return torch.cat(outs,2).transpose(1,2).to(v.dtype),(keys,coef)


def block_komp_value_learned(q,k,v,g,beta,initial_state,*,chunk_size=128,temperature=.5,ridge=1e-4,
                           evict='learned',scores=None,initial_scores=None,backend='triton'):
    _check(q,k,v,g,beta,initial_state,scores,initial_scores,chunk_size,temperature,ridge,evict)
    if backend=='recurrent':
        return recurrent_value_learned(q,k,v,g,beta,initial_state,chunk_size=chunk_size,temperature=temperature,ridge=ridge,
                                      scores=scores,initial_scores=initial_scores)
    if backend not in ('triton','hybrid'):raise ValueError('Unsupported backend')
    if not q.is_cuda or any(x.dtype==torch.float64 for x in (q,k,v,g,beta,*initial_state)):
        raise ValueError('CUDA FP32/BF16 required; use recurrent for FP64')
    with torch.autocast('cuda',enabled=False):
        scalar=g.float().mean(-1)
        cumulative=(g.float()-scalar[...,None]).cumsum(1)
        lo=torch.minimum(cumulative.detach().amin(1),torch.zeros_like(cumulative[:,0]))
        hi=torch.maximum(cumulative.detach().amax(1),torch.zeros_like(cumulative[:,0]))
        if backend=='hybrid' or bool(((hi-lo)>24).any()):
            return _stable(q,k,v,g,beta,initial_state,scores,initial_scores,chunk_size,temperature,ridge)
        origin=(lo+hi)*.5;scale=(cumulative-origin[:,None]).exp()
        keys,coef=initial_state
        out,(keys,coef)=block_komp_fast(q,k,v.float()/scale,scalar,beta,
            (keys,coef.float()*origin.exp()[:,:,None,:]),chunk_size=chunk_size,temperature=temperature,ridge=ridge,
            evict='learned',scores=scores,initial_scores=initial_scores)
        return (out*scale).to(v.dtype),(keys,coef*scale[:,-1,:,None,:])
