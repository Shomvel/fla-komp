# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import json
import math
import traceback
from pathlib import Path

import torch

from fla.layers.komp import ALPHA, BETA, D, K, M
from fla.layers.komp import KOMPMixerBase as Mixer
from fla.ops.komp import block_komp_fast, exact_komp
from fla.ops.komp.block_reference import block_komp_reference, block_komp_token_reference

ROOT = Path.cwd()
(ROOT / "evidence").mkdir(exist_ok=True)
checks = []


def check(name, a, b, low=False):
    a, b = a.detach().double(), b.detach().double()
    delta = a - b
    row = dict(
        name=name,
        rms=float(delta.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-6)),
        maximum=float(delta.abs().max() / b.abs().max().clamp_min(1)),
        rms_tolerance=0.035 if low else 0.005,
        max_tolerance=0.04 if low else 0.005,
    )
    checks.append(row)
    assert torch.isfinite(a).all() and row["rms"] <= row["rms_tolerance"] and row["maximum"] <= row["max_tolerance"], row


def call(kind, x):
    return (
        exact_komp(*x[:5], tuple(x[5:]), compute_chunk=16)[:2]
        if kind == "exact"
        else block_komp_fast(*x[:5], tuple(x[5:]), chunk_size=128)
    )


def differential(kind, x, label):
    x = [a.detach().requires_grad_() for a in x]
    gold = [a.detach().double().requires_grad_() for a in x]
    a, ss = call(kind, x)
    b, tt = (
        block_komp_token_reference(*gold[:5], tuple(gold[5:]), chunk_size=1, evict="fifo")
        if kind == "exact"
        else block_komp_reference(*gold[:5], tuple(gold[5:]), chunk_size=128)
    )
    weights = [torch.randn_like(z.float()) for z in (a, *ss)]
    da = torch.autograd.grad((a, *ss), x, weights)
    db = torch.autograd.grad((b, *tt), gold, [z.double() for z in weights])
    for name, aa, bb in zip(
        ["o", "keys", "state", "dq", "dk", "dv", "dg", "dbeta", "dk0", "ds0"], (a, *ss, *da), (b, *tt, *db)
    ):
        check(label + "/" + name, aa, bb, x[0].dtype == torch.bfloat16)
    print("PASS", label, flush=True)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    for kind in ("exact", "block"):
        mix = Mixer(kind).cuda()
        for dtype in (torch.float32, torch.bfloat16):
            n = 129
            q, k, v = [torch.randn(1, n, 2, K, device="cuda", dtype=dtype) for _ in range(3)]
            keys = mix.keys.detach()[None, :2].contiguous()
            s = torch.randn(1, 2, M, K, device="cuda") * 0.1
            differential(
                kind,
                [
                    q,
                    k,
                    v,
                    torch.full((1, n, 2, M), math.log(ALPHA), device="cuda"),
                    torch.full((1, n, 2), BETA, device="cuda"),
                    keys,
                    s,
                ],
                kind + "/" + str(dtype),
            )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = torch.randn(1, 2048, D, device="cuda")
            args = mix.operands(x)
        # New full training operator shape, and causal prefix perturbation.
        with torch.no_grad():
            a, _ = call(kind, args)
            changed = [z.clone() for z in args]
            for z in changed[:5]:
                z[:, 1024:] += 0.01
            b, _ = call(kind, changed)
            check(kind + "/T2048H16_causality", b[:, :1024], a[:, :1024], True)
        # Actual projected activations, two heads and one compression boundary.
        subset = [z[:, :129, :2].contiguous() for z in args[:5]] + [z[:, :2].contiguous() for z in args[5:]]
        differential(kind, subset, kind + "/real_activations")
        del mix
    report = dict(
        status="passed",
        checks=checks,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        scope="new H16 T2048 causality; 129-token random and actual activation forward/state/all-gradient differential vs double reference; no claim of universal ill-conditioned bound",
    )
    (ROOT / "evidence/numerical.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        (ROOT / "evidence/numerical-failure.json").write_text(
            json.dumps(dict(status="failed", checks=checks, traceback=traceback.format_exc()), indent=2)
        )
        raise
