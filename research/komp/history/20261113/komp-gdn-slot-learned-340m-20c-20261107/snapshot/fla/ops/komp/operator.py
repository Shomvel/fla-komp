# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Training-capable projected slot attention with a full Triton default backend."""

import math

import torch

from .reference import prepare_geometry, recurrent_reference


def projected_slot_attention(q, k, v, alpha, beta, slots, initial_state,
                             temperature=0.5, ridge=1e-4, backend='all_triton', chunk_size=16):
    """Apply routed kernel projection and delta updates, using write-before-read causality.

    q,k: [B,T,H,K], v: [B,T,H,V], beta,slots: [B,T,H].
    alpha: [B,T,H] scalar decay, or [B,T,H,M] old-slot pre-decay.
    Slot decay is available with all_triton(_v1) and torch backends.
    initial_state: (key slots [B,H,M,K], value coefficients [B,H,M,V]).
    Temperature is positive and fixed per call; ridge is a nonnegative constant.
    FP32, BF16 and FP16 input features are promoted to FP32; the torch backend
    additionally supports float64 for independent numerical validation.
    Both returned states retain their gradient paths across calls.
    """
    if backend not in ('torch', 'triton', 'chunk', 'chunk_torch', 'all_triton', 'all_triton_v1'):
        raise ValueError('backend must be torch, triton, chunk, chunk_torch, all_triton or all_triton_v1')
    if backend in ('chunk', 'chunk_torch', 'all_triton', 'all_triton_v1') and chunk_size not in (16, 32):
        raise ValueError('chunk_size must be 16 or 32')
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(ridge) or ridge < 0:
        raise ValueError('temperature must be positive and ridge nonnegative, both finite')
    keys, values = initial_state
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError('q,k,v must share [B,T,H] dimensions and q,k must share width')
    batch, t, heads, dk = q.shape
    if keys.ndim != 4 or keys.shape[:2] != (batch, heads) or keys.shape[-1] != dk:
        raise ValueError('key state must have shape [B,H,M,K]')
    m = keys.shape[-2]
    if min(batch, t, heads, dk, m, v.shape[-1]) <= 0 or m > 64:
        raise ValueError('Positive dimensions and at most 64 slots are supported')
    if values.shape != (batch, heads, m, v.shape[-1]):
        raise ValueError('value state must have shape [B,H,M,V]')
    if alpha.shape not in (q.shape[:3], (*q.shape[:3], m)):
        raise ValueError('alpha must have shape [B,T,H] or [B,T,H,M]')
    if any(x.shape != q.shape[:3] for x in (beta, slots)):
        raise ValueError('beta,slots must have shape [B,T,H]')
    if alpha.ndim == 4 and backend not in ('all_triton', 'all_triton_v1', 'torch'):
        raise ValueError('slot decay requires all_triton or torch backend')
    floats = (q, k, v, alpha, beta, keys, values)
    if any(not x.is_floating_point() or x.device != q.device for x in floats) or slots.device != q.device:
        raise ValueError('All tensors must share a device and features/states must be floating point')
    if slots.dtype not in (torch.int32, torch.int64):
        raise ValueError('slots must be int32 or int64')
    if backend in ('triton', 'chunk', 'all_triton', 'all_triton_v1') and (q.device.type != 'cuda' or any(x.dtype == torch.float64 for x in floats)):
        raise ValueError('Triton requires CUDA and FP32/BF16/FP16 inputs')
    if backend in ('all_triton', 'all_triton_v1'):
        from .utils import validate_inputs
        q, k, v, alpha, beta, slots, keys, values = [x.contiguous() for x in (q, k, v, alpha, beta, slots, keys, values)]
        validate_inputs(q, k, v, alpha, beta, keys, values, slots)
        if backend == 'all_triton':
            # Inverse-carrying chunk geometry and two-level-scan recurrence.
            from .fast_chunk import fast_chunk
            from .geometry_chunk import triton_geometry_chunked
            h, p, r, final_keys = triton_geometry_chunked(q, k, slots, keys, temperature, ridge)
            output, final_values = fast_chunk(h, p, r, v, alpha, beta, slots, values, chunk_size)
        else:
            # Frozen previous all-Triton path, kept for comparisons.
            from .geometry import triton_geometry
            from .triton_chunk import triton_chunk
            h, p, r, final_keys = triton_geometry(q, k, slots, keys, temperature, ridge)
            output, final_values = triton_chunk(h, p, r, v, alpha, beta, slots, values, chunk_size)
        return output, (final_keys, final_values)
    # this research API validates indices and numerical inputs once before launching kernels
    if bool(((slots < 0) | (slots >= m)).any()):
        raise ValueError('slot index out of range')
    if any(not bool(torch.isfinite(x).all()) for x in floats):
        raise ValueError('Inputs must be finite')
    if any(bool(((x < 0) | (x > 1)).any()) for x in (alpha, beta)):
        raise ValueError('alpha and beta must lie in [0,1]')
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    h, p, r, final_keys = prepare_geometry(q, k, slots, keys, temperature, ridge)
    args = (h, p, r, v.to(dtype), alpha.to(dtype), beta.to(dtype), slots, values.to(dtype))
    if backend in ('chunk', 'chunk_torch'):
        from .chunk import chunk_slot
        output, final_values = chunk_slot(*args, chunk_size=chunk_size, backend='triton' if backend == 'chunk' else 'torch')
    elif backend == 'triton':
        from .kernels import fused_recurrent_slot
        output, final_values = fused_recurrent_slot(*args)
    else:
        output, final_values = recurrent_reference(*args)
    return output, (final_keys, final_values)
