# Geometry optimization candidate

This candidate preserves the normalized exponential kernel, temperature, ridge,
source routing and analytical gradients. It does not delete the projection solve,
approximate the dictionary, or assume cyclic routing. The original serial implementation is retained as
`triton_geometry_serial`; gather candidates are explicitly selected with
`triton_geometry_fast`. The final section records the later route-only default integration.

## Changes

1. In LDL factorization and triangular substitution, extract a pivot column with
   `tl.gather` instead of a masked reduction across the entire matrix. The update
   algorithm and pivot order are the same; no matrix inverse update accumulates
   error across tokens.
2. The candidate accepts `dot_precision='ieee'` or `'tf32x3'`. IEEE isolates the
   gather change. TF32x3 uses three-component TF32 Tensor Core products for Gram
   construction and the matrix part of geometry backward. This is an explicit
   arithmetic mode change, not a claim of bitwise IEEE equivalence. All state,
   factorization and accumulation remain FP32. Defaults of the original function
   remain IEEE.

Both candidates keep each token independent after route lookup. M=1..64,
arbitrary/repeated replacement routes and non-power-of-two key dimensions retain
the existing contracts. The factor tape is unchanged, so this version does not
reduce asymptotic memory. Key-gradient accumulation still uses FP32 atomic adds.

## APIs

```python
from komp_slot.geometry import triton_geometry_serial, triton_geometry_fast
baseline = triton_geometry_serial(q, k, slots, initial_keys)
solver_only = triton_geometry_fast(q, k, slots, initial_keys, dot_precision='ieee')
tensor_core = triton_geometry_fast(q, k, slots, initial_keys, dot_precision='tf32x3')
```

Do not switch the public operator or training default until GPU numerical and
same-card timing evidence identifies a winner. The scalar and slot decay changes
use the same h/p/r geometry and are orthogonal to this experiment.

## Verification

`python validate_geometry_fast.py` checks all three modes against the independent
Torch recurrence geometry oracle, including every input gradient and final-key
gradient. It covers M=1/5/16/32/64, tails, repeat routing, noncontiguous inputs,
zero vectors, BF16 inputs and causal streaming. Normal acceptance uses the prior
FP32 atol=2e-4/rtol=1e-3 gradient tolerances (BF16 gradient atol=.003/rtol=.01).

A separate ill-conditioned near-duplicate case compares each mode against an
FP64 oracle. It requires finite results and candidate RMS error no greater than
max(.001, 1.25 times baseline RMS error). This is a conditioning regression test,
not a replacement for the normal-case tolerances. Exact duplicate atoms at
ridge=0 are outside the existing nonsingular-Gram contract.

`python benchmark_geometry.py --quick` compares T=256/2048 M32; without `--quick`
it additionally measures M16/M64 and T192. It records geometry-only and complete
geometry+chunk forward and forward/backward, median of three eager and CUDA Graph
trials, excluding compilation. It uses the same baseline API, temperature, ridge,
state dimensions and input dtype for every mode. The full measurement includes
initial/final state differentiation; no geometry gradients are omitted.

Local static checks passed. GPU runs are owned and submitted by the parent agent;
performance and correctness must be read from the generated JSON, not inferred
from this design note.

## Follow-up: exact parallel route history

The first GPU run passed 189 geometry checks, but gather/TF32x3 alone showed only
small or negative full-operator gains (T192 M32: baseline .371ms, gather IEEE
.375ms, TF32x3 .395ms; T1024 M16: baseline .900ms, gather IEEE .829ms, TF32x3
.820ms, graph forward/backward). This motivated inspection of the remaining
serial route construction, rather than claiming the matrix change solved the
bottleneck. These numbers are parent-reported partial measurements; final JSON
is the authoritative evidence.

`triton_geometry_parallel` independently enables exact integer route scans:

1. Each 128-token tile computes a per-slot inclusive maximum of matching write
   indices, and stores local sources plus a block summary.
2. A block-summary prefix scan computes the source entering each tile, including
   the initial dictionary indices.
3. Each tile combines local sources and carry, writes removed sources from its
   exclusive local prefix, and writes final dictionary sources.

The algorithm accepts arbitrary repeated routes. It does not assume a cyclic
slot schedule. The finish kernel never reads another CTA's mutable Source tile;
previous-token data comes from the same CTA's already loaded local tensor. All
source arithmetic is integer max, so output must equal the original route loop
bit for bit. Complexity is linear in the route output size plus a small block
prefix; no per-head serial loop over all T tokens remains.

The default parallel API retains original LDL and IEEE products. A separate
`gather=True,dot_precision='tf32x3'` experiment tests combined changes. Existing
baseline and gather APIs retain their old sequential route implementation for
an interpretable comparison.

Run `python validate_routes_parallel.py --benchmark` first, then
`python benchmark_geometry.py --parallel --quick`. Route tests include exact
comparison with both the sequential GPU kernel and an independent CPU oracle,
M1/5/16/32/64, boundaries 127/128/129, T4097 and all-repeated/cyclic/random routes.
All geometry outputs and input gradients are also checked with the new route
history. Route microbenchmarks include the training batch B64/T192/H2.

## Integrated default after route measurements

The route experiment passed 18 exact-history cases and 28 geometry output/gradient
checks. At B2/T2048/H4/M32, parent-measured route time fell from .38044ms to
.008476ms (44.88x). Complete operator forward/backward improved on the measured
shapes: T192 M32 .3715→.3461ms; T1024 M16 .9006→.6818ms; T1024 M32
1.8081→1.5239ms. These are CUDA Graph partial-results figures and should be cited
with the final benchmark evidence.

`triton_geometry` now selects the parallel integer history when T>=192 and M>=16,
retaining the original LDL and IEEE products. Other shapes retain serial history.
This is a conservative empirical H100 threshold, not a claim of universal shape
optimality. **The frozen baseline is now `triton_geometry_serial`.** The numerical
and benchmark harnesses explicitly reference it, so default integration cannot
silently change the comparison baseline. `triton_geometry_fast` still represents
the separate gather/TF32x3 candidate and is not the selected default.

`validate_routes_parallel.py` additionally checks T191/192, M15/16, T257, and a
T257 stream split into129/128-token pieces; values and every input gradient must
agree with the frozen serial baseline. Parent agent owns the post-integration
GPU regression, including the existing complete operator and slot decay tests.
