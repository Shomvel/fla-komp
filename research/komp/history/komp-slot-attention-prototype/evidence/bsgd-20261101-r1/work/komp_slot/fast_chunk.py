"""Chunk recurrence with two-level affine scans; same maps, local backward and contract as triton_chunk."""
import torch
import triton as tr
from torch.autograd.function import once_differentiable
from .triton_chunk import prepare, local_bwd_kernel, reduce_kernel
from .recurrence import output_kernel
from .scan import affine_scan, value_offset_kernel, output_offset_kernel

# Single-warp local backward avoids cross-warp reduction barriers (H100 sweep).
WARPS = {'prepare': 2, 'offset': 4, 'output': 4, 'local': 1}


class _FastChunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, p, r, v, a, b, slots, initial, c, local_bv, group):
        batch, t, heads, m = h.shape
        width = v.shape[-1]
        nc = tr.cdiv(t, c)
        bm = max(16, tr.next_power_of_2(m))
        maps = prepare(h, p, r, a, b, slots, c, WARPS['prepare'])
        os, ov, fs, fv = maps
        out = torch.empty(v.shape, device=v.device, dtype=torch.float32)
        final = torch.empty(initial.shape, device=v.device, dtype=torch.float32)
        tape = torch.empty((batch, heads, nc, m, width), device=v.device, dtype=torch.float32)
        off = torch.empty_like(tape)
        bv = 32
        nv = tr.cdiv(width, bv)
        value_offset_kernel[(nc, batch*heads, nv)](fv, v, off, t, heads, m, width, c, nc, bm, bv, num_warps=WARPS['offset'])
        affine_scan(fs, off, initial, tape, final, False, group, bv)
        output_kernel[(nc, batch*heads, nv)](os, ov, v, tape, out, t, heads, m, width, c, nc, bm, bv, num_warps=WARPS['output'])
        ctx.save_for_backward(h, p, r, v, a, b, slots, tape, os, fs)
        ctx.cfg = (c, initial.dtype, local_bv, group)
        return out, final

    @staticmethod
    @once_differentiable
    def backward(ctx, do, df):
        h, p, r, v, a, b, slots, tape, os, fs = ctx.saved_tensors
        c, initial_dtype, local_bv, group = ctx.cfg
        batch, t, heads, m = h.shape
        width = v.shape[-1]
        nc = tr.cdiv(t, c)
        bm = max(16, tr.next_power_of_2(m))
        do, df = do.contiguous(), df.contiguous()
        ds = torch.empty_like(tape)
        off = torch.empty_like(tape)
        di = torch.empty(df.shape, device=v.device, dtype=initial_dtype)
        nv = tr.cdiv(width, 32)
        output_offset_kernel[(nc, batch*heads, nv)](os, do, off, t, heads, m, width, c, nc, bm, 32, num_warps=WARPS['offset'])
        affine_scan(fs, off, df, ds, di, True, group, 32)
        nv = tr.cdiv(width, local_bv)
        work = torch.empty((batch, heads, nc, c, m, width), device=v.device, dtype=torch.float32)
        dv = torch.empty_like(v)
        if nv == 1:
            result = [torch.empty_like(x) for x in (h, p, r, a, b)]
            partial = result
        else:
            partial = [torch.empty((nv, *x.shape), device=v.device, dtype=torch.float32) for x in (h, p, r, a, b)]
        local_bwd_kernel[(nc, batch*heads, nv)](h, p, r, v, a, b, slots, tape, ds, do, work, *partial[:3], dv, *partial[3:],
                                                 batch, t, heads, m, width, c, nc, bm, local_bv, a.ndim == 4, num_warps=WARPS['local'])
        if nv > 1:
            result = [torch.empty_like(x) for x in (h, p, r, a, b)]
            for x, y in zip(partial, result):
                reduce_kernel[(tr.cdiv(y.numel(), 256),)](x, y, y.numel(), nv, tr.next_power_of_2(nv), 256)
        return *result[:3], dv, *result[3:], None, di, None, None, None


def fast_chunk(h, p, r, v, alpha, beta, slots, initial_values, chunk_size=16, local_bv=64, group=None):
    """FP32 geometry; alpha is [B,T,H] or [B,T,H,M], beta is [B,T,H]."""
    if alpha.shape not in (h.shape[:3], h.shape):
        raise ValueError('alpha must have shape [B,T,H] or [B,T,H,M]')
    if beta.shape != h.shape[:3] or slots.shape != h.shape[:3]:
        raise ValueError('beta and slots must have shape [B,T,H]')
    if chunk_size not in (16, 32):
        raise ValueError('chunk_size must be 16 or 32')
    inputs = (h, p, r, v, alpha, beta, slots, initial_values)
    with torch.cuda.device(h.device):
        return _FastChunk.apply(*[x.contiguous() for x in inputs], chunk_size, local_bv, group)
