# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Value-coordinate FIFO KOMP using native Triton kernels and a stable hybrid fallback."""

import math

import torch

from fla.ops.komp.block_fast import block_komp_fast
from fla.ops.komp.exact_komp import exact_komp
from fla.ops.komp.value_decay_hybrid import block_komp_value_decay_triton as hybrid


def _native(q, k, v, g, beta, initial_state, *, chunk_size=128, temperature=0.5, ridge=1e-4):
    _check(q, k, v, g, beta, initial_state, chunk_size, temperature, ridge)
    if not q.is_cuda or any(x.dtype == torch.float64 for x in (q, k, v, g, beta, *initial_state)):
        raise ValueError("Triton requires CUDA and FP32/BF16/FP16 inputs; use recurrent for FP64")
    m = initial_state[0].shape[-2]
    if chunk_size > 128 or m > 64:
        raise ValueError("Triton supports chunk_size<=128 and M<=64")
    # non-native chunk sizes use the bounded scan.
    native = (
        chunk_size in (1, 16, 32, 64, 128)
        and q.dtype in (torch.float32, torch.bfloat16)
        and k.dtype == q.dtype
        and q.shape[-1] <= 128
        and v.shape[-1] <= 128
    )
    if not native:
        return hybrid(q, k, v, g, beta, initial_state, chunk_size=chunk_size, temperature=temperature, ridge=ridge)
    with torch.autocast("cuda", enabled=False):
        scalar = g.float().mean(-1)
        cumulative = (g.float() - scalar[..., None]).cumsum(1)
        # a detached origin centers the range without adding an invariant coordinate VJP.
        lo = torch.minimum(cumulative.detach().amin(1), torch.zeros_like(cumulative[:, 0]))
        hi = torch.maximum(cumulative.detach().amax(1), torch.zeros_like(cumulative[:, 0]))
        origin = (lo + hi) * 0.5
        # large channel differences use the inverse-free implementation.
        if bool(((hi - lo) > 24).any()):
            return hybrid(q, k, v, g, beta, initial_state, chunk_size=chunk_size, temperature=temperature, ridge=ridge)
        scale = (cumulative - origin[:, None]).exp()
        values = v.float() / scale
        keys, coef = initial_state
        state = (keys, coef.float() * origin.exp()[:, :, None, :])
        if chunk_size == 1:
            out, (keys, coef), _ = exact_komp(q, k, values, scalar, beta, state, temperature=temperature, ridge=ridge)
            # convert the native physical ring to the public oldest-first state.
            shift = q.shape[1] % m
            keys = torch.roll(keys, -shift, dims=-2)
            coef = torch.roll(coef, -shift, dims=-2)
        else:
            out, (keys, coef) = block_komp_fast(
                q, k, values, scalar, beta, state, chunk_size=chunk_size, temperature=temperature, ridge=ridge
            )
        return (out * scale).to(v.dtype), (keys, coef * scale[:, -1, :, None, :])


def _check(q, k, v, g, beta, initial_state, chunk_size, temperature, ridge):
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("q,k:[B,T,H,K], v:[B,T,H,V] must have matching B,T,H")
    b, t, h, dk = q.shape
    keys, coef = initial_state
    if keys.ndim != 4 or keys.shape[:2] != (b, h) or keys.shape[-1] != dk:
        raise ValueError("keys must have shape [B,H,M,K]")
    m, dv = keys.shape[-2], v.shape[-1]
    if coef.shape != (b, h, m, dv) or g.shape != v.shape or beta.shape != (b, t, h):
        raise ValueError("coef:[B,H,M,V], g:[B,T,H,V], beta:[B,T,H] required")
    if min(b, t, h, dk, m, dv) < 1 or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("dimensions and integer chunk_size must be positive")
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("temperature and ridge must be finite and positive")
    for x in (q, k, v, g, beta, keys, coef):
        if not x.is_floating_point() or x.device != q.device:
            raise ValueError("all inputs must be floating tensors on the same device")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("inputs must be finite")
    if bool((g > 0).any()) or bool(((beta < 0) | (beta > 1)).any()):
        raise ValueError("log decay must be <= 0 and beta must be in [0,1]")


def block_komp_value_decay(q, k, v, g, beta, initial_state, *, chunk_size=128, temperature=0.5, ridge=1e-4, backend="triton"):
    """FIFO Block-KOMP with per-value log decay, read before compression.

    Inputs q/k are [B,T,H,K], v/g [B,T,H,V], beta [B,T,H]. Initial and
    final states are (keys [B,H,M,K], coefficients [B,H,M,V]), oldest-first.
    Output has v dtype; state math is FP32 (FP64 for the recurrent oracle).
    Triton supports CUDA, M<=64, C<=128 and first-order gradients for all
    seven inputs. Streaming splits must be on compression boundaries.

    Args:
        backend (str, Optional):
            'triton' selects native kernels with stable hybrid fallback;
            'hybrid' forces that fallback; 'recurrent' is the PyTorch oracle.
            Default: 'triton'.
    """
    if backend == "triton":
        return _native(q, k, v, g, beta, initial_state, chunk_size=chunk_size, temperature=temperature, ridge=ridge)
    if backend == "hybrid":
        return hybrid(q, k, v, g, beta, initial_state, chunk_size=chunk_size, temperature=temperature, ridge=ridge)
    if backend == "recurrent":
        from fla.ops.komp.value_decay_naive import naive_value_decay_komp

        _check(q, k, v, g, beta, initial_state, chunk_size, temperature, ridge)
        return naive_value_decay_komp(
            q, k, v, g, beta, initial_state, chunk_size=chunk_size, temperature=temperature, ridge=ridge
        )
    raise ValueError("backend must be triton, hybrid or recurrent")


def exact_komp_value_decay(q, k, v, g, beta, initial_state, *, temperature=0.5, ridge=1e-4, backend="triton"):
    """Value-decay KOMP with per-token compression and oldest-first final state."""
    return block_komp_value_decay(
        q, k, v, g, beta, initial_state, chunk_size=1, temperature=temperature, ridge=ridge, backend=backend
    )
