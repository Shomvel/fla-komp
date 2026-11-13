# Copyright (c) 2026. MIT license, same as this source tree.
"""Fused SwiGLU retaining the model's BF16 SiLU-before-multiply rounding."""
import torch
import triton as tr
import triton.language as tl
from torch.nn import functional as F


@tr.jit
def _forward(A, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    a = tl.load(A+i, i < N, 0).to(tl.float32)
    b = tl.load(B+i, i < N, 0).to(tl.float32)
    silu = (a/(1.+tl.exp(-a))).to(A.dtype.element_ty).to(tl.float32)
    tl.store(Y+i, silu*b, i < N)


@tr.jit
def _backward(A, B, DY, DA, DB, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    a = tl.load(A+i, i < N, 0).to(tl.float32)
    b = tl.load(B+i, i < N, 0).to(tl.float32)
    dy = tl.load(DY+i, i < N, 0).to(tl.float32)
    denom = 1.+tl.exp(-a)
    sig = 1./denom
    silu = (a/denom).to(A.dtype.element_ty).to(tl.float32)
    # These BF16 roundings match the separate eager MultiplyBackward and SiLUBackward.
    dsilu = (dy*b).to(A.dtype.element_ty).to(tl.float32)
    da = dsilu*sig*(1.+a*(1.-sig))
    tl.store(DA+i, da, i < N)
    tl.store(DB+i, dy*silu, i < N)


class _RoundedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        a, b = a.contiguous(), b.contiguous()
        y = torch.empty_like(a)
        _forward[(tr.cdiv(a.numel(),1024),)](a,b,y,a.numel(),1024,enable_fp_fusion=False)
        ctx.save_for_backward(a,b)
        return y

    @staticmethod
    def backward(ctx, dy):
        a,b = ctx.saved_tensors
        da,db = torch.empty_like(a),torch.empty_like(b)
        _backward[(tr.cdiv(a.numel(),1024),)](a,b,dy.contiguous(),da,db,a.numel(),1024,enable_fp_fusion=False)
        return da,db


def rounded_swiglu(a,b):
    if a.is_cuda and a.dtype in (torch.bfloat16,torch.float16) and a.dtype==b.dtype:
        return _RoundedSwiGLU.apply(a,b)
    return F.silu(a)*b
