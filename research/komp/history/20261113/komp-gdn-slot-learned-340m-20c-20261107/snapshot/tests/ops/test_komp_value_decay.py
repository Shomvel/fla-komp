# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Value-decay KOMP outputs, state, all VJPs, fallback and streaming."""

import pytest
import torch

from fla.ops.komp import block_komp_value_decay, exact_komp_value_decay
from fla.utils import IS_NVIDIA, assert_close, device


def inputs(t, k=64, v=64, m=32, dtype=torch.float32, target=device):
    torch.manual_seed(42)
    tensors = [torch.randn(1, t, 2, k, device=target, dtype=dtype) for _ in range(2)]
    tensors += [
        torch.randn(1, t, 2, v, device=target, dtype=dtype),
        -torch.rand(1, t, 2, v, device=target) * 0.1,
        torch.rand(1, t, 2, device=target),
        torch.randn(1, 2, m, k, device=target),
        torch.randn(1, 2, m, v, device=target) * 0.1,
    ]
    return tensors


def call(xs, chunk, backend="triton"):
    o, (k, s) = block_komp_value_decay(*xs[:5], (xs[5], xs[6]), chunk_size=chunk, backend=backend)
    return o, k, s


def compare(xs, chunk, backend="triton"):
    actual = [x.detach().requires_grad_() for x in xs]
    reference = [x.detach().double().requires_grad_() for x in xs]
    a, r = call(actual, chunk, backend), call(reference, chunk, "recurrent")
    weights = [torch.randn_like(x) for x in a]
    da = torch.autograd.grad(a, actual, weights)
    dr = torch.autograd.grad(r, reference, [x.double() for x in weights])
    tolerance = 0.025 if xs[0].dtype == torch.bfloat16 else 0.003
    for name, x, y in zip(("out", "keys", "state", "dq", "dk", "dv", "dg", "dbeta", "dk0", "ds0"), (*a, *da), (*r, *dr)):
        assert torch.isfinite(x).all()
        assert_close(name, y, x.double(), tolerance, err_atol=3e-4)


@pytest.mark.skipif(not IS_NVIDIA, reason="KOMP Triton is currently validated on NVIDIA CUDA")
@pytest.mark.parametrize(
    ("t", "k", "v", "m", "chunk", "dtype"),
    [
        pytest.param(33, 64, 64, 32, 1, torch.float32, id="exact-ring-wrap"),
        pytest.param(65, 64, 32, 32, 1, torch.bfloat16, id="exact-asymmetric"),
        pytest.param(35, 64, 64, 32, 16, torch.float32, id="block-short-chunk"),
        pytest.param(129, 64, 64, 32, 128, torch.bfloat16, id="block-tail-bf16"),
        pytest.param(257, 64, 64, 32, 128, torch.float32, id="block-long-vjp"),
        pytest.param(7, 5, 7, 3, 3, torch.float32, id="non-native-shape-fallback"),
    ],
)
def test_block_komp_value_decay(t, k, v, m, chunk, dtype):
    compare(inputs(t, k, v, m, dtype), chunk)


@pytest.mark.skipif(not IS_NVIDIA, reason="requires NVIDIA CUDA")
@pytest.mark.parametrize(
    ("chunk", "mode"),
    [
        pytest.param(1, "uniform", id="exact-zero-reference-gradient"),
        pytest.param(128, "heterogeneous", id="block-large-range-fallback"),
        pytest.param(1, "below_bound", id="exact-native-range-edge"),
        pytest.param(16, "above_bound", id="block-fallback-range-edge"),
    ],
)
def test_value_decay_strong_forgetting(chunk, mode):
    xs = inputs(17)
    xs[3].fill_(-1000)
    if mode == "heterogeneous":
        xs[3][..., 1::2] = 0
    elif mode in ("below_bound", "above_bound"):
        span = 23.9 if mode == "below_bound" else 24.1
        xs[3].fill_(0)
        xs[3][..., ::2] = -2 * span / 17
    compare(xs, chunk)


@pytest.mark.skipif(not IS_NVIDIA, reason="requires NVIDIA CUDA")
@pytest.mark.parametrize(
    ("chunk", "backend"),
    [
        pytest.param(1, "triton", id="exact-native"),
        pytest.param(16, "triton", id="block-native"),
        pytest.param(16, "hybrid", id="block-strict-hybrid"),
    ],
)
def test_value_decay_streaming(chunk, backend):
    xs = inputs(33)
    base = call(xs, chunk, backend)
    part = call([x[:, :16] for x in xs[:5]] + xs[5:], chunk, backend)
    last = call([x[:, 16:] for x in xs[:5]] + list(part[1:]), chunk, backend)
    pairs = list(zip((torch.cat((part[0], last[0]), 1), *last[1:]), base))
    edited = [x.clone() for x in xs]
    for x in edited[:3]:
        x[:, 16:] += 4
    edited[3][:, 16:] = -0.2
    pairs.append((call(edited, chunk, backend)[0][:, :16], base[0][:, :16]))
    for x, y in pairs:
        if backend == "hybrid":
            torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-4)
        else:
            # native Block retains the validated TF32 numerical budget across coordinate origins.
            assert_close("stream/prefix", y, x, 0.003, err_atol=3e-4)
    if chunk == 1:
        direct, _ = exact_komp_value_decay(*xs[:5], (xs[5], xs[6]))
        torch.testing.assert_close(direct, base[0])


def test_value_decay_reference_gradcheck():
    xs = [x.double().requires_grad_() for x in inputs(3, 3, 2, 2, target="cpu")]
    assert torch.autograd.gradcheck(lambda *a: call(a, 1, "recurrent"), tuple(xs))
