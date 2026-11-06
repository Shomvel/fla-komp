"""Independent autograd reference for BSGD-inspired local merging transitions.

This is an associative-memory adaptation, not the original SVM training algorithm.
The grid mode minimizes vector-valued RKHS degradation on a finite line-segment grid.
Only pairs containing the incoming atom are considered. Hard choices are detached.
"""
import torch


def rbf(x, y, temperature):
    return torch.exp(-(x-y).square().sum(-1)/(2*temperature))


def bsgd_reference(q, k, v, g, beta, merge, initial_state, *, temperature=.5,
                   mode='gated', routing='cyclic', grid_size=17, offset=0):
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v, g, beta, merge = [x.to(dtype) for x in (q, k, v, g, beta, merge)]
    keys, coef = [x.to(dtype) for x in initial_state]
    batch, length, heads, _ = q.shape
    slots = keys.shape[-2]
    mask = torch.arange(slots, device=q.device)[None, None, :]
    outputs = []
    for t in range(length):
        decay = g[:, t] if g.ndim == 4 else g[:, t, :, None]
        old = coef * decay.exp()[..., None]
        kt, qt = k[:, t], q[:, t]
        h = rbf(keys, kt[..., None, :], temperature)
        residual = v[:, t] - (h[..., None]*old).sum(-2)
        write = beta[:, t, :, None]*residual
        index = torch.full((batch, heads), (offset+t) % slots, device=q.device, dtype=torch.long)
        gamma = merge[:, t]
        if mode == 'grid':
            with torch.no_grad():
                d2 = (keys-kt[..., None, :]).square().sum(-1)
                aa = old.square().sum(-1)
                bb = write.square().sum(-1)[..., None]
                ab = (old*write[..., None, :]).sum(-1)
                candidates = torch.linspace(0, 1, grid_size, device=q.device, dtype=dtype)
                a = torch.exp(-candidates.square()*d2[..., None]/(2*temperature))
                s = torch.exp(-(1-candidates).square()*d2[..., None]/(2*temperature))
                energy = a.square()*aa[..., None]+s.square()*bb[..., None]+2*a*s*ab[..., None]
                error = (aa+bb+2*h*ab)[..., None]-energy
                if routing == 'cyclic':
                    error = error.masked_fill(mask[..., None] != index[..., None, None], float('inf'))
                chosen = error.flatten(-2).argmin(-1)
                index = chosen // grid_size
                gamma = (chosen % grid_size).to(dtype)/(grid_size-1)
        select = (mask == index[..., None]).to(dtype)
        u = (keys*select[..., None]).sum(-2)
        cj = (old*select[..., None]).sum(-2)
        z = (1-gamma[..., None])*u+gamma[..., None]*kt
        a, s = rbf(z, u, temperature), rbf(z, kt, temperature)
        merged = a[..., None]*cj+s[..., None]*write
        keys = keys*(1-select[..., None])+select[..., None]*z[..., None, :]
        coef = old*(1-select[..., None])+select[..., None]*merged[..., None, :]
        outputs.append((rbf(keys, qt[..., None, :], temperature)[..., None]*coef).sum(-2))
    return torch.stack(outputs, 1), (keys, coef)
