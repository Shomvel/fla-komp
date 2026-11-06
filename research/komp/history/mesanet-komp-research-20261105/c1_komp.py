"""Exact-per-token FIFO KOMP, with independent computational chunking.

Research PyTorch implementation, not a fused production kernel. Public tensors
match Block-KOMP: q/k/v [B,T,H,D], log_decay [B,T,H,M] (or [B,T,H]),
beta [B,T,H], state (keys [B,H,M,K], coefficients [B,H,M,V]).
State uses physical ring slots, so streaming callers must pass start_position.
Every token is read BEFORE compression, matching block_reference with C=1.
"""
import torch
from torch.nn import functional as F


def kernel(a, b, temperature):
    return ((a @ b.transpose(-1, -2) - 1) / temperature).exp()


def prepare(q, k, v, g, beta, initial_state):
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k = [F.normalize(x.to(dtype), dim=-1, eps=1e-6).transpose(1, 2) for x in (q, k)]
    v = v.to(dtype).transpose(1, 2)
    keys, state = initial_state
    keys = F.normalize(keys.to(dtype), dim=-1, eps=1e-6)
    state = state.to(dtype)
    g = g.to(dtype).transpose(1, 2)
    if g.ndim == 3:
        g = g[..., None].expand(*g.shape, keys.shape[-2])
    return q, k, v, g, beta.to(dtype).transpose(1, 2), keys, state


def sequential(q, k, v, g, beta, initial_state, *, start_position=0, temperature=.5, ridge=1e-4):
    """Literal recurrence: decay, predict, append, read, project, reorder ring."""
    q, k, v, g, beta, keys, state = prepare(q, k, v, g, beta, initial_state)
    m = keys.shape[-2]
    eye = torch.eye(m, device=q.device, dtype=q.dtype)
    outputs = []
    for t in range(q.shape[2]):
        slot = (start_position + t) % m
        decayed = g[:, :, t].exp()[..., None] * state
        h = kernel(keys, k[:, :, t:t+1], temperature)
        residual = v[:, :, t] - (h * decayed).sum(-2)
        added = beta[:, :, t, None] * residual
        extended_keys = torch.cat((keys, k[:, :, t:t+1]), dim=-2)
        extended_state = torch.cat((decayed, added[..., None, :]), dim=-2)
        outputs.append((kernel(extended_keys, q[:, :, t:t+1], temperature) * extended_state).sum(-2))
        # Position the new key in the just-freed ring slot; other slots stay put.
        order = list(range(m))
        order[slot] = m
        kept_keys, kept_state = extended_keys[..., order, :], extended_state[..., order, :]
        p = torch.linalg.solve(kernel(kept_keys, kept_keys, temperature) + ridge * eye,
                               kernel(kept_keys, keys[..., slot:slot+1, :], temperature))
        state = kept_state + p * decayed[..., slot:slot+1, :]
        keys = kept_keys
    return torch.stack(outputs, dim=2).transpose(1, 2), (keys, state)


def fifo_geometry(q, k, beta, keys, start_position, temperature, ridge):
    """All projections depend on input keys, not on recurrent coefficients.

    Return U,V such that S_t=(I+U_t V_t^T)D_t S_(t-1)+beta e_r v_t^T,
    and R such that output_t=R_t^T D_t S_(t-1)+beta*kappa(q,k)*v_t.
    """
    b, h, length, d = k.shape
    m = keys.shape[-2]
    t = torch.arange(length, device=k.device) + start_position
    slots = torch.arange(m, device=k.device)
    positions = t[:, None] - 1 - (t[:, None] - 1 - slots[None]) % m
    indices = torch.where(positions < start_position, slots[None], m + positions - start_position)
    bank = torch.cat((keys, k), dim=-2)
    old = bank[:, :, indices, :]
    e = F.one_hot(t % m, m).to(k.dtype)[None, None].expand(b, h, -1, -1)
    kept = old * (1-e[..., None]) + k[..., None, :] * e[..., None]
    dropped = (old * e[..., None]).sum(-2, keepdim=True)
    gram = kernel(kept, kept, temperature)
    gram = gram + ridge * torch.eye(m, device=k.device, dtype=k.dtype)
    p = torch.linalg.solve(gram, kernel(kept, dropped, temperature)).squeeze(-1)
    hkey = kernel(old, k[..., None, :], temperature).squeeze(-1)
    qkey = kernel(old, q[..., None, :], temperature).squeeze(-1)
    qnew = ((q*k).sum(-1).sub(1)/temperature).exp()
    u = torch.stack((p-e, e), dim=-1)
    vf = torch.stack((e, -beta[..., None]*hkey), dim=-1)
    read = qkey - (beta*qnew)[..., None]*hkey
    return u, vf, read, beta*qnew, kept[:, :, -1]


def parallel(q, k, v, g, beta, initial_state, *, compute_chunk=16, start_position=0,
             temperature=.5, ridge=1e-4):
    """Rank-two block solve. compute_chunk changes scheduling, NOT compression.

    Chunk state recurrence remains, as in MesaNet's FLA implementation. Inside
    each chunk, a 2C x 2C unit-lower-triangular solve replaces C sequential
    state updates. Stable decay ratios avoid exp(-prefix) overflow.
    """
    if compute_chunk < 1:
        raise ValueError('compute_chunk must be positive')
    q, k, v, g, beta, keys, state = prepare(q, k, v, g, beta, initial_state)
    u, vf, read, direct, final_keys = fifo_geometry(q, k, beta, keys, start_position, temperature, ridge)
    outputs, state = solve_updates(v, g, beta, state, u, vf, read, direct, compute_chunk)
    return outputs, (final_keys, state)


def solve_updates(v, g, beta, state, u, vf, read, direct, compute_chunk):
    """Shared rank-two solve after dictionary geometry has been determined."""
    outputs = []
    for start in range(0, v.shape[2], compute_chunk):
        end = min(start+compute_chunk, v.shape[2])
        uc, vc, rc = u[:, :, start:end], vf[:, :, start:end], read[:, :, start:end]
        values, bc = v[:, :, start:end], beta[:, :, start:end]
        logs = g[:, :, start:end].cumsum(dim=2)
        a = logs.exp()
        c = end-start
        past = torch.arange(c, device=v.device)[:, None] > torch.arange(c, device=v.device)[None, :]
        # Only causal ratios are used; don't exponentiate positive future gaps.
        gaps = logs[:, :, :, None, :] - logs[:, :, None, :, :]
        weights = gaps.masked_fill(~past[None, None, :, :, None], 0).exp()
        weights = weights * past[None, None, :, :, None]
        coupling = torch.einsum('bhtmr,bhtsm,bhsmu->bhtrsu', vc, weights, uc)
        coupling = coupling.reshape(*coupling.shape[:2], 2*c, 2*c)
        rhs = torch.einsum('bhtmr,bhtm,bhmv->bhtrv', vc, a, state)
        writes = torch.stack((torch.zeros_like(values), bc[..., None]*values), dim=-2)
        rhs = (rhs+writes).flatten(2, 3)
        system = torch.eye(2*c, dtype=v.dtype, device=v.device)-coupling
        z = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
        z = z.reshape(*z.shape[:2], c, 2, z.shape[-1])
        read_coupling = torch.einsum('bhtm,bhtsm,bhsmr->bhtsr', rc, weights, uc)
        out = torch.einsum('bhtm,bhtm,bhmv->bhtv', rc, a, state)
        out = out + torch.einsum('bhtsr,bhsrv->bhtv', read_coupling, z)
        out = out + direct[:, :, start:end, None]*values
        outputs.append(out)
        last_decay = (logs[:, :, -1:, :]-logs).exp()
        state = a[:, :, -1, :, None]*state + torch.einsum('bhtm,bhtmr,bhtrv->bhmv', last_decay, uc, z)
    return torch.cat(outputs, dim=2).transpose(1, 2), state
