"""Block-KOMP with eviction and projectron flags: batched PyTorch chunk loop, autograd gradients.

Matrix form per chunk (dictionary held by physical slot, see block_reference.block_komp_token_reference):
  Gam=exp(cumsum g) by slot, D[t,s]=exp(Gc[t,slot_s]-Gc[s,slot_s]) for the chunk's temporaries,
  W = (I + diag(b)(D o Hn))^-1 diag(b) (V - (Gam o Hk) S0), O = (Gam o Rk) S0 + (D' o Rn) W,
  b = beta, or beta * novelty (projectron: the Cholesky pivots of the union Gram, i.e. each new key's
  ridge-regularized projection residual on everything before it; no gradient through novelty).
At the chunk end the M+n candidates are ranked (fifo / residual / coef_norm / learned), the lowest n are
jointly projected onto the kept M, and kept temporaries are placed into freed slots.
The serial dependency is per chunk, not per token. This path is for model studies and as a numerical
reference; block_fast.py is the Triton kernel for fifo eviction without projectron.
"""
import torch
import torch.nn.functional as F
from .block_reference import EVICTION


def _kern(x, y, tau):
    return torch.exp((x @ y.transpose(-1, -2) - 1) / tau)


def _gather_rows(x, idx):
    return torch.gather(x, -2, idx[..., None].expand(*idx.shape, x.shape[-1]))


def _assign(keep, m, tslot, n):
    """Physical slot of every kept atom. keep: ascending candidate indices [B,H,M] (slots < m, temps >= m).

    Kept slot atoms stay put; a kept temporary takes its positional slot when that slot was freed and no
    earlier kept temporary claimed it; the remaining temporaries fill the remaining freed slots in order.
    """
    is_temp = keep >= m
    zeros = torch.zeros(*keep.shape[:-1], m, dtype=torch.long, device=keep.device)
    free = zeros.scatter_add(-1, torch.where(is_temp, 0, keep), (~is_temp).long()) == 0
    s = (keep-m).clamp(min=0)
    pos = torch.where(is_temp, tslot[s], 0)
    claim = is_temp & torch.gather(free, -1, pos)
    if n > m:
        earlier = (s[..., :, None] > s[..., None, :]) & is_temp[..., None, :] & (pos[..., :, None] == pos[..., None, :])
        claim &= ~earlier.any(-1)
    claimed = zeros.scatter_add(-1, pos, claim.long()) > 0
    rest_slot = free & ~claimed
    rest_temp = is_temp & ~claim
    slot_rank = torch.cumsum(rest_slot.long(), -1)-1
    temp_rank = torch.cumsum(rest_temp.long(), -1)-1
    ar = torch.arange(m, device=keep.device).expand_as(slot_rank)
    by_rank = torch.cat((zeros, zeros[..., :1]), -1).scatter(-1, torch.where(rest_slot, slot_rank, m), ar)[..., :m]
    fill = torch.gather(by_rank, -1, temp_rank.clamp(min=0))
    return torch.where(~is_temp, keep, torch.where(claim, pos, fill))


def block_komp_flex(q, k, v, g, beta, initial_state, chunk_size=64, temperature=0.5, ridge=1e-4,
                    evict='fifo', projectron=False, scores=None, initial_scores=None):
    """q,k:[B,T,H,K], v:[B,T,H,V], g:[B,T,H] or [B,T,H,M] log decay, beta:[B,T,H].

    scores [B,T,H] and initial_scores [B,H,M] are required for evict='learned' (ranking only, no gradient).
    Returns output [B,T,H,V] and the final (keys, values) state in physical slot order.
    """
    if evict not in EVICTION:
        raise ValueError(f'evict must be one of {EVICTION}')
    # Kernel Grams and solves must stay FP32 even inside a BF16 autocast region.
    with torch.autocast(device_type=q.device.type, enabled=False):
        return _flex(q, k, v, g, beta, initial_state, chunk_size, temperature, ridge, evict, projectron,
                     scores, initial_scores)


def _flex(q, k, v, g, beta, initial_state, chunk_size, temperature, ridge, evict, projectron, scores, initial_scores):
    keys0, values0 = initial_state
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    tr = lambda x: x.to(dtype).transpose(1, 2)
    qn, kn = [F.normalize(tr(x), dim=-1, eps=1e-6) for x in (q, k)]
    vv, bb = tr(v), beta.to(dtype).transpose(1, 2)
    b_, h_, t_len, _ = qn.shape
    m = keys0.shape[-2]
    gg = tr(g) if g.ndim == 4 else g.to(dtype).transpose(1, 2)[..., None].expand(b_, h_, t_len, m)
    K = F.normalize(keys0.to(dtype), dim=-1, eps=1e-6)
    S = values0.to(dtype)
    dev = q.device
    born = torch.arange(-m, 0, device=dev, dtype=dtype).expand(b_, h_, m)
    rank_state = initial_scores.to(dtype) if evict == 'learned' else None
    sc = scores.to(dtype).transpose(1, 2) if evict == 'learned' else None
    outs = []
    for c0 in range(0, t_len, chunk_size):
        n = min(chunk_size, t_len-c0)
        Kn, Q, Vc, bc = kn[:, :, c0:c0+n], qn[:, :, c0:c0+n], vv[:, :, c0:c0+n], bb[:, :, c0:c0+n]
        Gc = torch.cumsum(gg[:, :, c0:c0+n], 2)
        Gam = torch.exp(Gc)
        tslot = (c0+torch.arange(n, device=dev)) % m
        GcS = Gc[..., tslot]
        diag = GcS.diagonal(dim1=-2, dim2=-1)
        le = torch.ones(n, n, dtype=torch.bool, device=dev).tril()
        Ex = torch.exp(torch.where(le, GcS-diag[..., None, :], 0.))
        Dl, Dp = Ex*le.tril(-1), Ex*le
        dend = torch.exp(GcS[..., n-1, :]-diag)
        gend = Gam[..., n-1, :]
        Hk, Hn = _kern(Kn, K, temperature), _kern(Kn, Kn, temperature)
        Rk, Rn = _kern(Q, K, temperature), _kern(Q, Kn, temperature)
        U = torch.cat((K, Kn), 2)
        eye = torch.eye(m+n, dtype=dtype, device=dev)
        GU = None
        if projectron or evict == 'residual':
            GU = _kern(U, U, temperature)+ridge*eye
        if projectron:
            novelty = torch.linalg.cholesky(GU.detach()).diagonal(dim1=-2, dim2=-1)[..., m:].square()
            bc = bc*novelty
        A = torch.eye(n, dtype=dtype, device=dev)+bc[..., None]*(Dl*Hn)
        W = torch.linalg.solve_triangular(A, bc[..., None]*(Vc-(Gam*Hk)@S), upper=False, unitriangular=True)
        outs.append((Gam*Rk)@S+(Dp*Rn)@W)
        coef = torch.cat((gend[..., None]*S, dend[..., None]*W), 2)
        if evict == 'fifo':
            rank = torch.cat((born, (c0+torch.arange(n, device=dev, dtype=dtype)).expand(b_, h_, n)), -1)
        elif evict == 'residual':
            rank = 1./torch.linalg.inv(GU.detach()).diagonal(dim1=-2, dim2=-1)
        elif evict == 'coef_norm':
            rank = coef.detach().norm(dim=-1)
        else:
            rank = torch.cat((rank_state, sc[:, :, c0:c0+n]), -1)
        # Ties are broken by recency: newer atoms rank first.
        allborn = torch.cat((born, (c0+torch.arange(n, device=dev, dtype=dtype)).expand(b_, h_, n)), -1)
        recent = torch.sort(allborn, dim=-1, descending=True).indices
        order = torch.gather(recent, -1, torch.sort(torch.gather(rank, -1, recent), dim=-1, descending=True, stable=True).indices)
        keep = torch.sort(order[..., :m], -1).values
        drop = torch.sort(order[..., m:], -1).values
        Kk, Ke = _gather_rows(U, keep), _gather_rows(U, drop)
        Sk, Se = _gather_rows(coef, keep), _gather_rows(coef, drop)
        P = torch.linalg.solve(_kern(Kk, Kk, temperature)+ridge*torch.eye(m, dtype=dtype, device=dev), _kern(Kk, Ke, temperature))
        Snew = Sk+P@Se
        slot = _assign(keep, m, tslot, n)
        inv = torch.argsort(slot, -1)
        K, S = _gather_rows(Kk, inv), _gather_rows(Snew, inv)
        born = torch.gather(torch.gather(allborn, -1, keep), -1, inv)
        if evict == 'learned':
            rank_state = torch.gather(torch.gather(rank, -1, keep), -1, inv)
    out = torch.cat(outs, 2).transpose(1, 2)
    return out, (K, S)
