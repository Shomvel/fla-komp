"""FP64-capable fixed-dictionary KAPA references and differentiable Gaussian adapter.

The protected rule is a sequence-memory adaptation of KAPA-2, not the paper's
full growing-dictionary algorithm. Geometry here is intentionally PyTorch.
"""
import torch


def gaussian_features(q, k, centers, *, temperature=.5, gram_ridge=.01):
    """Return h(q), h(k), (G+gram_ridge I)^-1 using ordinary PyTorch autograd.

    q,k [B,T,H,D], centers [B,H,M,D]. No normalization; full Gaussian RBF.
    gram_ridge regularizes the coefficient metric, not the KAPA constraints.
    """
    dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
    q,k,centers=q.to(dtype),k.to(dtype),centers.to(dtype)
    def feature(x):
        return torch.exp(-(x[...,None,:]-centers[:,None]).square().sum(-1)/(2*temperature))
    gram=torch.exp(-(centers[..., :,None,:]-centers[...,None,:,:]).square().sum(-1)/(2*temperature))
    eye=torch.eye(centers.shape[-2],device=q.device,dtype=dtype)
    inverse=torch.linalg.solve(gram+gram_ridge*eye,eye.expand_as(gram))
    return feature(q),feature(k),inverse


def protected_direction(h, inverse_metric, anchors=2, regularization=.05):
    """w=P B(B^T P B+lambda I)^-1 e0, B=[current,last,...] features."""
    dtype=torch.float64 if h.dtype==torch.float64 else torch.float32
    h=h.to(dtype)
    p=inverse_metric.to(dtype)
    p=(p+p.transpose(-1,-2))*.5
    rows=[]
    for t in range(h.shape[1]):
        history=[h[:,t-r] if t>=r else torch.zeros_like(h[:,t]) for r in range(anchors+1)]
        basis=torch.stack(history,-1)
        x=p@basis
        small=basis.transpose(-1,-2)@x+regularization*torch.eye(anchors+1,device=h.device,dtype=dtype)
        rhs=torch.zeros((*small.shape[:-1],1),device=h.device,dtype=dtype)
        rhs[...,0,0]=1
        rows.append((x@torch.linalg.solve(small,rhs)).squeeze(-1))
    return torch.stack(rows,1)


def protected_kapa_reference(q,h,v,g,beta,inverse_metric,initial_state,*,anchors=2,regularization=.05):
    dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
    q,h,v,g,beta,initial_state=[x.to(dtype) for x in (q,h,v,g,beta,initial_state)]
    w=protected_direction(h,inverse_metric.to(dtype),anchors,regularization)
    state=initial_state
    outs=[]
    for t in range(q.shape[1]):
        decay=g[:,t] if g.ndim==4 else g[:,t,:,None]
        pre=decay.exp()[...,None]*state
        err=v[:,t]-(pre*h[:,t,...,None]).sum(-2)
        state=pre+beta[:,t,:,None,None]*w[:,t,...,None]*err[...,None,:]
        outs.append((state*q[:,t,...,None]).sum(-2))
    return torch.stack(outs,1),state


def replay_kapa_reference(q,h,v,g,beta,inverse_metric,initial_state,*,window=3,regularization=.05):
    """Fixed-dictionary KAPA-2: re-fit current and past target values jointly.

    This rank-r baseline differs from protected KAPA, which preserves predictions
    rather than replaying old labels. Only the reference is implemented here.
    """
    dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
    q,h,v,g,beta,initial_state=[x.to(dtype) for x in (q,h,v,g,beta,initial_state)]
    p=inverse_metric.to(dtype);p=(p+p.transpose(-1,-2))*.5
    state=initial_state;outs=[]
    for t in range(q.shape[1]):
        n=min(window,t+1)
        basis=torch.stack([h[:,t-r] for r in range(n)],-1)
        targets=torch.stack([v[:,t-r] for r in range(n)],-2)
        decay=g[:,t] if g.ndim==4 else g[:,t,:,None]
        pre=decay.exp()[...,None]*state
        x=p@basis
        small=basis.transpose(-1,-2)@x+regularization*torch.eye(n,device=q.device,dtype=dtype)
        state=pre+beta[:,t,:,None,None]*(x@torch.linalg.solve(small,targets-basis.transpose(-1,-2)@pre))
        outs.append((state*q[:,t,...,None]).sum(-2))
    return torch.stack(outs,1),state
