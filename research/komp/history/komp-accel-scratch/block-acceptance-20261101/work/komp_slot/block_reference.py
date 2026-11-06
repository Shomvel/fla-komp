"""Block-KOMP reference: per-token kernel delta writes into fresh slots, FIFO compression at chunk ends.

Within a chunk of C tokens the dictionary is the M carried atoms plus the chunk's new keys. Token t
decays all coefficients by exp(g_t), predicts with the current dictionary, and writes beta_t * err_t
into its own new slot. After the chunk the newest M atoms are kept and every evicted atom is jointly
projected onto them with the ridge-regularized kernel Gram (kept coefficients are not shrunk).
"""
import torch
import torch.nn.functional as F


def kernel(x, y, temperature):
    return torch.exp((x @ y.transpose(-1, -2) - 1) / temperature)


def block_komp_reference(q, k, v, g, beta, initial_state, chunk_size=64, temperature=0.5, ridge=1e-4):
    """q,k:[B,T,H,K], v:[B,T,H,V], g (log decay <= 0), beta:[B,T,H]; state ([B,H,M,K], [B,H,M,V])."""
    initial_keys, initial_values = initial_state
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k = [F.normalize(x.to(dtype), dim=-1, eps=1e-6) for x in (q, k)]
    keys = F.normalize(initial_keys.to(dtype), dim=-1, eps=1e-6)
    coef = initial_values.to(dtype)
    v, g, beta = v.to(dtype), g.to(dtype), beta.to(dtype)
    t_len, m = q.shape[1], keys.shape[-2]
    eye = torch.eye(m, dtype=dtype, device=q.device)
    outs = []
    for c0 in range(0, t_len, chunk_size):
        n = min(chunk_size, t_len - c0)
        dictionary, state = keys, coef
        for s in range(n):
            t = c0 + s
            state = torch.exp(g[:, t])[..., None, None] * state
            kt = k[:, t, :, None, :]
            err = v[:, t] - (state * kernel(dictionary, kt, temperature)).sum(-2)
            dictionary = torch.cat((dictionary, kt), -2)
            state = torch.cat((state, (beta[:, t, :, None] * err)[:, :, None]), -2)
            outs.append((state * kernel(dictionary, q[:, t, :, None, :], temperature)).sum(-2))
        kept, evicted = dictionary[..., n:n+m, :], dictionary[..., :n, :]
        proj = torch.linalg.solve(kernel(kept, kept, temperature) + ridge * eye, kernel(kept, evicted, temperature))
        keys, coef = kept, state[..., n:n+m, :] + proj @ state[..., :n, :]
    return torch.stack(outs, 1), (keys, coef)
