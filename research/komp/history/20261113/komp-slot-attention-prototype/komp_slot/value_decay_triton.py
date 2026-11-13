"""Value-coordinate forgetting through the optimized native KOMP kernels.

Write g[t,v] = s[t] + dc[t,v]. Right diagonal scaling commutes with
all dictionary-side linear operations (including the delta prediction and
FIFO projection). In coordinates C'=C exp(-c), c=cumsum(dc), the recurrence
has scalar forgetting s and values v'=v exp(-c). This is an exact algebraic
change of coordinates, NOT an approximation of the value gate by its mean.

Keep the native kernels' numerical policies. Use the bounded, inverse-free
hybrid scan if the coordinate transform would have excessive dynamic range.
"""
import torch

from .exact_komp import exact_komp
from .block_fast import block_komp_fast
from .value_decay_hybrid import block_komp_value_decay_triton as hybrid


def block_komp_value_decay_triton(q,k,v,g,beta,initial_state,*,chunk_size=128,
                                  temperature=.5,ridge=1e-4):
    from .value_decay import _check
    _check(q,k,v,g,beta,initial_state,chunk_size,temperature,ridge)
    if not q.is_cuda or any(x.dtype==torch.float64 for x in (q,k,v,g,beta,*initial_state)):
        raise ValueError('Triton requires CUDA and FP32/BF16/FP16 inputs; use recurrent for FP64')
    m=initial_state[0].shape[-2]
    if chunk_size>128 or m>64:
        raise ValueError('Triton supports chunk_size<=128 and M<=64')
    # Arbitrary chunk sizes remain valid through the original bounded backend.
    native=(chunk_size in (1,16,32,64,128) and q.dtype in (torch.float32,torch.bfloat16)
            and k.dtype==q.dtype and q.shape[-1]<=128 and v.shape[-1]<=128)
    if not native:
        return hybrid(q,k,v,g,beta,initial_state,chunk_size=chunk_size,temperature=temperature,ridge=ridge)
    with torch.autocast('cuda',enabled=False):
        scalar=g.float().mean(-1)
        cumulative=(g.float()-scalar[...,None]).cumsum(1)
        # Detached, per-sequence/channel origin. The output is invariant to this
        # origin; detaching avoids cancellation through a needless min/max VJP.
        lo=torch.minimum(cumulative.detach().amin(1),torch.zeros_like(cumulative[:,0]))
        hi=torch.maximum(cumulative.detach().amax(1),torch.zeros_like(cumulative[:,0]))
        origin=(lo+hi)*.5
        # Limit forward AND backward intermediates, not merely the exp itself.
        # Strong heterogeneous forgetting uses the inverse-free implementation.
        if bool(((hi-lo)>24).any()):
            return hybrid(q,k,v,g,beta,initial_state,chunk_size=chunk_size,temperature=temperature,ridge=ridge)
        scale=(cumulative-origin[:,None]).exp()
        values=v.float()/scale
        keys,coef=initial_state
        state=(keys,coef.float()*origin.exp()[:,:,None,:])
        if chunk_size==1:
            out,(keys,coef),_=exact_komp(q,k,values,scalar,beta,state,
                                        temperature=temperature,ridge=ridge)
            # Native Exact returns physical ring slots. Public value-decay API
            # is oldest-first, including T<M and non-M-aligned streaming calls.
            shift=q.shape[1]%m
            keys=torch.roll(keys,-shift,dims=-2)
            coef=torch.roll(coef,-shift,dims=-2)
        else:
            out,(keys,coef)=block_komp_fast(q,k,values,scalar,beta,state,
                chunk_size=chunk_size,temperature=temperature,ridge=ridge)
        return (out*scale).to(v.dtype),(keys,coef*scale[:,-1,:,None,:])
