"""Causal normalized exponential-kernel geometry and independent projection oracle."""

import torch
import torch.nn.functional as F


def kernel(x, y, temperature):
    return torch.exp((x @ y.transpose(-1, -2) - 1) / temperature)


def prepare_geometry(q, k, slots, initial_keys, temperature=0.5, ridge=1e-4):
    """Return old-key similarities, relocation coefficients, and post-write query weights.

    Inputs have FLA's [B,T,H,D] layout; initial_keys is [B,H,M,D].
    Keys are normalized before replacement. A positive ridge regularizes the
    removed atom's projection; it is not an exact unregularized KOMP solve.
    """
    compute_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q = F.normalize(q.to(compute_dtype), dim=-1, eps=1e-6)
    k = F.normalize(k.to(compute_dtype), dim=-1, eps=1e-6)
    keys = F.normalize(initial_keys.to(compute_dtype), dim=-1, eps=1e-6)
    m = keys.shape[-2]
    eye = torch.eye(m, device=q.device, dtype=compute_dtype)
    hs, ps, rs = [], [], []
    for t in range(q.shape[1]):
        e = F.one_hot(slots[:, t].long(), m).to(compute_dtype)
        old = (keys * e[..., None]).sum(-2, keepdim=True)
        h = kernel(keys, k[:, t, :, None, :], temperature).squeeze(-1)
        keys = keys * (1-e[..., None]) + e[..., None] * k[:, t, :, None, :]
        gram = kernel(keys, keys, temperature)
        p = torch.linalg.solve(gram + ridge*eye, kernel(keys, old, temperature)).squeeze(-1)
        r = kernel(keys, q[:, t, :, None, :], temperature).squeeze(-1)
        hs.append(h)
        ps.append(p)
        rs.append(r)
    return torch.stack(hs, 1), torch.stack(ps, 1), torch.stack(rs, 1), keys


def recurrent_reference(h, p, r, v, alpha, beta, slots, initial_values):
    """Dense transition oracle, including gradients of the initial and final states."""
    state = initial_values
    m = state.shape[-2]
    eye = torch.eye(m, dtype=state.dtype, device=state.device)
    outputs = []
    for t in range(v.shape[1]):
        e = F.one_hot(slots[:, t].long(), m).to(state.dtype)
        transition = eye + (p[:, t]-e)[..., :, None]*e[..., None, :]
        transition = transition-beta[:, t, :, None, None]*e[..., :, None]*h[:, t, :, None, :]
        decay=alpha[:,t,...,None] if alpha.ndim==4 else alpha[:,t,:,None,None]
        state = transition @ (decay*state)
        state = state + beta[:, t, :, None, None]*e[..., :, None]*v[:, t, :, None, :]
        outputs.append((r[:, t, :, :, None]*state).sum(-2))
    return torch.stack(outputs, 1), state


def projection_oracle(q, k, v, alpha, beta, slots, initial_keys, initial_values, temperature=0.5):
    """Full function projection, independently checking the rank-two rewrite at ridge=0."""
    q, k, keys = [F.normalize(x, dim=-1, eps=1e-6) for x in (q, k, initial_keys)]
    state = initial_values
    m = keys.shape[-2]
    outputs = []
    for t in range(q.shape[1]):
        e = F.one_hot(slots[:, t].long(), m).to(q.dtype)
        decay=alpha[:,t,...,None] if alpha.ndim==4 else alpha[:,t,:,None,None]
        pre = decay*state
        ht = kernel(keys, k[:, t, :, None, :], temperature)
        residual = v[:, t]-(pre*ht).sum(-2)
        new_keys = (1-e[..., None])*keys + e[..., None]*k[:, t, :, None, :]
        rhs = kernel(new_keys, keys, temperature) @ pre
        rhs = rhs + kernel(new_keys, k[:, t, :, None, :], temperature)*(beta[:, t, :, None]*residual)[..., None, :]
        state = torch.linalg.solve(kernel(new_keys, new_keys, temperature), rhs)
        keys = new_keys
        outputs.append((kernel(keys, q[:, t, :, None, :], temperature)*state).sum(-2))
    return torch.stack(outputs, 1), keys, state
