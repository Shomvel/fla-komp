"""Per-token learned eviction with the same rank-two computational chunking.

The hard-selection prepass is sequential over time, vectorized over batch and
heads, and independent of coefficients. This is NOT a fully parallel top-M
implementation. Retained score and birth metadata preserve physical slots and
tie-breaking across streaming calls. Projection and coefficient computation
reuse the batched geometry/rank-two formulation.
"""
import torch
from torch.nn import functional as F
from c1_komp import kernel, prepare, solve_updates


@torch.no_grad()
def selection_schedule(scores, slots, initial_scores=None, initial_births=None, start_position=0):
    """Evict smallest score, breaking ties by oldest birth, as frozen reference."""
    b, length, h = scores.shape
    scores = scores.detach().transpose(1, 2)
    current_scores = scores.new_zeros(b,h,slots) if initial_scores is None else initial_scores.detach().clone()
    ids = torch.arange(slots,device=scores.device)[None,None].expand(b,h,-1).clone()
    births = (ids-slots if initial_births is None else initial_births.clone())
    old_ids, kept_ids, dropped_ids, drop_slots = [], [], [], []
    physical_slots = torch.arange(slots,device=scores.device)[None,None]
    for t in range(length):
        old_ids.append(ids.clone())
        minimum = current_scores.min(dim=-1).values
        oldest = births.masked_fill(current_scores != minimum[...,None], torch.iinfo(torch.long).max).argmin(-1)
        accept = scores[:,:,t] >= minimum  # the incoming atom is newest in any tie
        replace = (physical_slots == oldest[...,None]) & accept[...,None]
        old_drop = ids.gather(-1,oldest[...,None]).squeeze(-1)
        dropped_ids.append(torch.where(accept,old_drop,slots+t))
        drop_slots.append(torch.where(accept,oldest,-1))
        ids = torch.where(replace,slots+t,ids)
        births = torch.where(replace,start_position+t,births)
        current_scores = torch.where(replace,scores[:,:,t,None],current_scores)
        kept_ids.append(ids.clone())
    return (torch.stack(old_ids,2),torch.stack(kept_ids,2),torch.stack(dropped_ids,2),
            torch.stack(drop_slots,2),current_scores,births)


def gather_keys(bank, indices):
    b,h,_,d=bank.shape
    gathered=bank.gather(2,indices.flatten(2)[...,None].expand(b,h,-1,d))
    return gathered.reshape(*indices.shape,d)


def parallel_learned(q,k,v,g,beta,initial_state,scores,*,initial_scores=None,initial_births=None,
                     start_position=0,compute_chunk=16,score_write_gate=True,temperature=.5,ridge=1e-4,
                     projection_dtype=torch.float64):
    """beta is the base write strength; optionally apply the experiment's score gate.

    Returns output, (final_keys, final_coefficients), and selection metadata.
    For streaming, pass returned scores/births and advance start_position.
    Hard selection is detached, while the optional 2*sigmoid(score) gate remains
    differentiable, exactly as in the existing learned experiments.

    This research implementation defaults to FP64 normalization/Gram/solve for
    projection geometry, even with FP32 coefficients. Near-duplicate keys made
    both the frozen reference and the original all-FP32 prototype sensitive to
    rounding. This is a correctness baseline, NOT an efficient GPU precision
    policy. Set projection_dtype=torch.float32 to study the original arithmetic.
    """
    if compute_chunk < 1:
        raise ValueError('compute_chunk must be positive')
    if score_write_gate:
        beta=beta*(2*scores.sigmoid())
    raw_keys,raw_initial_keys=k,initial_state[0]
    q,k,v,g,beta,keys,state=prepare(q,k,v,g,beta,initial_state)
    m=keys.shape[-2]
    old_ids,keep_ids,drop_ids,drop_slots,final_scores,final_births=selection_schedule(
        scores,m,initial_scores,initial_births,start_position)
    bank=torch.cat((keys,k),dim=2)
    old,kept=gather_keys(bank,old_ids),gather_keys(bank,keep_ids)
    # Higher precision must include normalization and Gram construction; casting
    # an already rounded FP32 Gram to FP64 cannot recover the lost information.
    projection_bank=torch.cat((F.normalize(raw_initial_keys.to(projection_dtype),dim=-1,eps=1e-6),
        F.normalize(raw_keys.to(projection_dtype),dim=-1,eps=1e-6).transpose(1,2)),dim=2)
    projection_kept=gather_keys(projection_bank,keep_ids)
    projection_dropped=gather_keys(projection_bank,drop_ids[...,None])
    gram=kernel(projection_kept,projection_kept,temperature)+ridge*torch.eye(m,device=k.device,dtype=projection_dtype)
    p=torch.linalg.solve(gram,kernel(projection_kept,projection_dropped,temperature)).squeeze(-1).to(k.dtype)
    accept=(drop_slots>=0)[...,None].to(k.dtype)
    e=F.one_hot(drop_slots.clamp_min(0),m).to(k.dtype)*accept
    # If an old atom is deleted: rank two, with new key taking its physical slot.
    # If new atom is deleted: rank one, projecting its residual onto the old dictionary.
    u=torch.stack((accept*p-e,torch.where(accept.bool(),e,p)),dim=-1)
    hkey=kernel(old,k[...,None,:],temperature).squeeze(-1)
    qkey=kernel(old,q[...,None,:],temperature).squeeze(-1)
    qnew=((q*k).sum(-1).sub(1)/temperature).exp()
    vf=torch.stack((e,-beta[...,None]*hkey),dim=-1)
    read=qkey-(beta*qnew)[...,None]*hkey
    outputs,state=solve_updates(v,g,beta,state,u,vf,read,beta*qnew,compute_chunk)
    return outputs,(kept[:,:,-1],state),dict(scores=final_scores,births=final_births,dropped_slots=drop_slots)
