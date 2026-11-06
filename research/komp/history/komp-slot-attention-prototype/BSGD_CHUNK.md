# Chunk gated BSGD merging

`komp_slot/bsgd_chunk.py` is a chunk-parallel implementation of the **same gated,
cyclic, per-token** rule as `bsgd_merging(mode='gated')`. Unlike block KOMP,
changing `chunk_size` here does not change when dictionary maintenance happens.
All arithmetic in forward and first backward is Triton; Python allocates tensors,
validates metadata and launches kernels. Higher derivatives are unsupported.

## API

```python
from komp_slot.bsgd_chunk import chunk_bsgd_merging

out, (centers, coefficients) = chunk_bsgd_merging(
    q, k, v, log_decay, beta, merge, initial_state,
    temperature=0.5, chunk_size=32, offset=processed_tokens,
)
```

Shapes, Gaussian RBF, raw-center convention and gate domains match
[BSGD_MERGING.md](BSGD_MERGING.md): q/k `[B,T,H,D]`, v `[B,T,H,V]`, beta/merge
`[B,T,H]`, log_decay `[B,T,H]` or `[B,T,H,M]`, initial centers `[B,H,M,D]`, initial
coefficients `[B,H,M,V]`. FP32/BF16 tensors on one CUDA device; M<=64 and D,V<=128.
Output dtype is v.dtype; both final states are FP32. Merge/beta are expected in
[0,1], log_decay<=0; no hidden clipping or activation is applied.

Supported chunks: 16, 32, 64. Arbitrary token cuts are supported with the original
cyclic phase passed as `offset`. No value-dependent grid search is implemented by
this path: that policy cannot precompute centers independently of value state.

## Geometry stage

For physical slot j, its updates form an independent affine recurrence
`u_new = (1-gamma_t) u_old + gamma_t k_t` at tokens `(offset+t) mod M = j`.
One CTA handles each batch/head/slot, with serial depth approximately T/M rather
than T. This first implementation does not parallel-scan within each slot chain.
Only initial centers and one new center per token are stored in a pool:
`O(B*H*(M+T)*D)` instead of a full dictionary per token.

Once the pool is built, every token independently reads its pre- and post-update
dictionary and computes

```
h_t,i = kappa(u_before_i, k_t)
r_t,i = kappa(u_after_i, q_t)
a_t   = kappa(z_t, u_before_j)
s_t   = kappa(z_t, k_t)
p_t   = a_t e_j
b_t   = beta_t s_t
alpha_t = exp(log_decay_t)
```

The coefficient recurrence is therefore exactly the generic KOMP core with
`p=a e_j` and effective write `b=beta s`:

`C_t = diag(alpha_t) C_{t-1} + e_j[y_t^T diag(alpha_t) C_{t-1} + b_t v_t^T]`,
where `y_t = (a_t-1)e_j - b_t h_t`.

## Rank-1 chunk maps

Let `u_t` denote the V-wide correction at token t. Within a chunk:

`u_t = y_t^T diag(P_t) C_in + sum_{s<t} y_t,j_s D_t,s u_s + b_t v_t`,

where `P_t,i = product_{r=0..t} alpha_r,i`,
`D_t,s = product_{r=s+1..t} alpha_r,j_s`.

Build the strictly lower matrix `L_t,s = y_t,j_s D_t,s` and compute
`U = (I-L)^-1 [diag-slot-prefix(Y) C_in + diag(b) V]`.
The implementation uses the existing divide-free triangular inverse helper.
This is a C×C system, rather than the generic rank-2 core's 2C×2C system.

Using read responses `R_t,s = r_t,j_s D_t,s` for s<=t, construct the maps
`O = OS C_in + OV V`, `C_out = FS C_in + FV V`.
All chunks build their maps in parallel. Existing Triton two-level affine scans
propagate chunk-entry states; outputs are independent per chunk thereafter.
Scalar and per-slot decay use multiplicative scans, never division by products
or subtraction of cumulative log infinities. Padding uses identity decay, zero
writes, zero readout, and no selected slot.

## Backward

1. Reverse affine scan obtains the adjoint at every chunk boundary.
2. Each chunk independently recomputes its local coefficient states and applies
   the existing exact KOMP recurrence adjoint, returning dh/dp/dr/dalpha/db/dv.
   It supports independent value-width tiles and reduces partial gradients.
3. Independent token geometry kernels differentiate Gaussian reads, predictions
   and merge coefficients. They atomically accumulate center-pool gradients;
   FP32 summation order is not bitwise deterministic.
4. Each slot traverses its center chain backwards, adds the final-center
   adjoint, and emits dk/dmerge/dinitial_centers. `dg=dalpha*alpha` is a Triton
   elementwise kernel. Effective-write gradients are propagated to beta and s.

Thus gradients cover q, k, v, scalar/per-slot decay, beta, merge and both initial
states, including losses that depend only on outputs or only on final states.

## Memory and scope

Forward retains geometry O(B*T*H*M), pooled centers O(B*H*(T+M)*D), chunk maps,
and boundary coefficient states. Backward still allocates an FP32 local-state
recompute workspace O(B*T*H*M*V); it is not a constant-memory backward. Chunk
maps and scan temporaries also cost memory. The center stage remains serial
within each slot, and geometry backward uses atomics; neither is claimed fully
parallel in T or optimized to FLA performance.

`validate_bsgd_chunk.py` compares outputs, both final states and all input
gradients to the independent FP64 token reference, and tests extreme gates,
irregular widths, tails, chunk sizes, causality and arbitrary-cut streaming.
`benchmark_bsgd_chunk.py` compares the identical gated serial rule and chunk
variants on the same GPU, with block KOMP as a semantically different baseline.
Frozen sources and actual GPU results are under `evidence/bsgd-chunk-20261101-r1/`.

## H100 results (2026-11-01)

**868 checks passed**, including 216 suffix interventions and 36 future-gradient
checks (both maximum changes exactly zero on tested inputs). Against the independent
FP64 recurrence, the ordinary output/state/all-gradient comparisons had worst RMS
relative error **5.29e-7 in FP32** and **0.001751 in BF16**. Tolerances were 5e-4
and 0.015, with an absolute fallback of 2e-6 for nearly zero quantities. The suite
also covers initial/final states, separate output/state losses, T<M, irregular
widths, exact gate endpoints and arbitrary-cut streaming. No formal all-input
causality or bitwise deterministic-gradient guarantee is claimed.

Scalar-decay full forward+backward timings, BF16 q/k/v, same H100:

| B/T/H/D/V/M | serial merging | best chunk merging | speedup | block KOMP C64 |
|---|---:|---:|---:|---:|
| 2/256/4/32/32/16 | 0.844 ms | 0.090 ms (C16) | 9.36× | 0.094 ms |
| 2/2048/4/64/64/32 | 13.756 ms | 0.514 ms (C16) | 26.76× | 0.301 ms |
| 64/192/8/64/64/32 | 3.134 ms | 2.459 ms (C32) | 1.27× | 1.520 ms |

The last row's forward-only time regresses from 0.666 to 0.793 ms, despite the
forward+backward improvement. Per-slot decay at 2k/C16 takes 0.528 ms versus
0.514 ms for scalar decay. This is not uniformly faster than block KOMP, whose
semantics are different. The default C32 is configurable; C16 is best in the
tested small-batch configurations, while C32 wins for the tested large batch.

All 24 scalar/slot timing rows, memory measurements and method details are in
[the generated report](evidence/bsgd-chunk-20261101-r1/RESULTS.md). There was no
Depo/Brevo training in this acceptance run. Kernel source fingerprints, job exit
code 0, environment and cleanup receipts are preserved with the report.
