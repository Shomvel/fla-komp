"""FIFO KOMP with value-coordinate forgetting; differentiable PyTorch prototype.

This is a distinct operator, not a dispatch to the existing slot-decay Triton
kernels. State is (keys, coefficients), both oldest-first. Exact compresses
after every read; Block compresses after each chunk, including a final tail.
The chunk implementation retains O(B H V C^2) temporary values and is intended
for correctness/algorithm experiments, not production throughput.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .block_reference import block_komp_reference, kernel


class ValueDecayGate(nn.Module):
    """Low-rank, FP32 gate logits, initially alpha_init on every value channel.

    The existing logsigmoid/8 parameterization is retained to isolate the
    decay-axis change. Random down + zero up avoids a doubly zero factorization.
    No slot decay is applied by this module or the operators below.
    """

    def __init__(self, hidden_size, num_heads, value_dim, rank=64, alpha_init=.995):
        super().__init__()
        if min(hidden_size, num_heads, value_dim, rank) <= 0:
            raise ValueError('gate dimensions must be positive')
        if not 0 < alpha_init < 1:
            raise ValueError('alpha_init must be between zero and one')
        self.num_heads, self.value_dim = num_heads, value_dim
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, num_heads * value_dim, bias=True)
        nn.init.zeros_(self.up.weight)
        log_p = 8 * math.log(alpha_init)
        nn.init.constant_(self.up.bias, log_p - math.log(-math.expm1(log_p)))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = F.linear(x.float(), self.down.weight.float())
            z = F.linear(z, self.up.weight.float(), self.up.bias.float())
            return (F.logsigmoid(z) / 8).reshape(*x.shape[:-1], self.num_heads, self.value_dim)


def _check(q, k, v, g, beta, initial_state, chunk_size, temperature, ridge):
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError('q,k:[B,T,H,K], v:[B,T,H,V] must have matching B,T,H')
    b, t, h, dk = q.shape
    keys, coef = initial_state
    if keys.ndim != 4 or keys.shape[:2] != (b, h) or keys.shape[-1] != dk:
        raise ValueError('keys must have shape [B,H,M,K]')
    m, dv = keys.shape[-2], v.shape[-1]
    if coef.shape != (b, h, m, dv) or g.shape != v.shape or beta.shape != (b, t, h):
        raise ValueError('coef:[B,H,M,V], g:[B,T,H,V], beta:[B,T,H] required')
    if min(b, t, h, dk, m, dv) < 1 or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('dimensions and integer chunk_size must be positive')
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError('temperature and ridge must be finite and positive')
    for x in (q, k, v, g, beta, keys, coef):
        if not x.is_floating_point() or x.device != q.device:
            raise ValueError('all inputs must be floating tensors on the same device')
        if not bool(torch.isfinite(x).all()):
            raise ValueError('inputs must be finite')
    if bool((g > 0).any()) or bool(((beta < 0) | (beta > 1)).any()):
        raise ValueError('log decay must be <= 0 and beta must be in [0,1]')


def block_komp_value_decay(q, k, v, g, beta, initial_state, *, chunk_size=128,
                          temperature=.5, ridge=1e-4, backend='chunk'):
    """Value decay replaces slot decay; outputs retain v dtype, states FP32/64.

    backend='triton' uses CUDA Triton state forward/backward with PyTorch
    geometry (first-order gradients only, FP64 unsupported).
    backend='recurrent' is the independent token reference; 'chunk' solves
    the causal delta writes in parallel with a triangular system per channel.
    Float64 is used if any input is Float64. Otherwise state math is FP32,
    including under autocast. This function does not implement generation
    caches or preserve an unfinished block between calls: streaming splits
    must occur on compression boundaries to reproduce a monolithic call.
    """
    if backend == 'triton':
        from .value_decay_triton import block_komp_value_decay_triton
        return block_komp_value_decay_triton(q, k, v, g, beta, initial_state,
                                            chunk_size=chunk_size, temperature=temperature, ridge=ridge)
    if backend not in ('recurrent', 'chunk'):
        raise ValueError('backend must be recurrent, chunk or triton')
    _check(q, k, v, g, beta, initial_state, chunk_size, temperature, ridge)
    output_dtype = v.dtype
    inputs = (q, k, v, g, beta, *initial_state)
    dtype = torch.float64 if any(x.dtype == torch.float64 for x in inputs) else torch.float32
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, g, beta, keys, coef = [x.to(dtype) for x in inputs]
        if backend == 'recurrent':
            out, state = block_komp_reference(q, k, v, g, beta, (keys, coef),
                                             chunk_size, temperature, ridge, decay_axis='value')
            return out.to(output_dtype), state
        q, k = [F.normalize(x, dim=-1, eps=1e-6).transpose(1, 2) for x in (q, k)]
        keys = F.normalize(keys, dim=-1, eps=1e-6)
        v, g, beta = [x.transpose(1, 2) for x in (v, g, beta)]
        m = keys.shape[-2]
        eye_m = torch.eye(m, device=q.device, dtype=dtype)
        outputs = []
        for start in range(0, q.shape[2], chunk_size):
            end = min(start + chunk_size, q.shape[2])
            qc, kc, vc, bc = q[:, :, start:end], k[:, :, start:end], v[:, :, start:end], beta[:, :, start:end]
            gc = g[:, :, start:end].cumsum(-2)
            n = end - start
            causal = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
            # Never exponentiate future/past inverse decay: strong gates can
            # underflow forward decay safely, without exp(+large) or 0*inf.
            delta = gc.unsqueeze(-2) - gc.unsqueeze(-3)
            decay = torch.exp(torch.where(causal[..., None], delta, 0)) * causal[..., None]
            kk = kernel(kc, kc, temperature)
            lower = bc[..., :, None, None] * kk[..., None] * decay
            lower = lower * causal.tril(-1)[..., None]
            system = lower.permute(0, 1, 4, 2, 3) + torch.eye(n, device=q.device, dtype=dtype)
            predicted = (kernel(kc, keys, temperature) @ coef) * gc.exp()
            rhs = bc[..., None] * (vc - predicted)
            writes = torch.linalg.solve_triangular(system, rhs.transpose(-1, -2).unsqueeze(-1),
                                                   upper=False, unitriangular=True)
            writes = writes.squeeze(-1).transpose(-1, -2)
            past = (kernel(qc, keys, temperature) @ coef) * gc.exp()
            reads = (kernel(qc, kc, temperature)[..., None] * decay * writes.unsqueeze(-3)).sum(-2)
            outputs.append(past + reads)
            old_coef = coef * gc[..., -1:, :].exp()
            new_coef = writes * (gc[..., -1:, :] - gc).exp()
            dictionary = torch.cat((keys, kc), -2)
            coefficients = torch.cat((old_coef, new_coef), -2)
            kept, evicted = dictionary[..., n:, :], dictionary[..., :n, :]
            projection = torch.linalg.solve(kernel(kept, kept, temperature) + ridge * eye_m,
                                            kernel(kept, evicted, temperature))
            coef = coefficients[..., n:, :] + projection @ coefficients[..., :n, :]
            keys = kept
        return torch.cat(outputs, -2).transpose(1, 2).to(output_dtype), (keys, coef)


def exact_komp_value_decay(q, k, v, g, beta, initial_state, *, temperature=.5,
                          ridge=1e-4, backend='chunk'):
    """Compress every token; read before compression, matching current Exact.

    Unlike the production ring layout, this prototype returns oldest-first
    states. Do not pass them directly to a physical-slot streaming API.
    """
    return block_komp_value_decay(q, k, v, g, beta, initial_state, chunk_size=1,
                                  temperature=temperature, ridge=ridge, backend=backend)
