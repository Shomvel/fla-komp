# BSGD-inspired merging transition prototype

This prototype implements the local merging transition discussed in
`codex://threads/01a19283-b11a-7b71-a6a2-4c0012471b81`, with an independent
coefficient-dependent search baseline. Source: Wang, Crammer and Vucetic,
[Budgeted Stochastic Gradient Descent for Large-Scale SVM Training (2012), §6.3](https://www.jmlr.org/papers/v13/wang12b.html).
It is an associative-memory adaptation, not a reproduction of SVM training.

The gated policy also has a chunk implementation with Triton forward and backward:
`komp_slot/bsgd_chunk.py`, documented in [BSGD_CHUNK.md](BSGD_CHUNK.md). The original
serial implementation and grid-search modes below remain independent references.

## State and kernel

Slots store centers `U [B,H,M,D]` and vector coefficients `C [B,H,M,V]`.
The readout is `f(q) = sum_i kappa(U_i,q) C_i`. The kernel is

`kappa(x,y) = exp(-||x-y||² / (2 tau))`, default `tau=0.5`.

Its diagonal is exactly one. On unit input centers it equals block KOMP's
`exp((x·y-1)/tau)`, but merged centers generally have norm below one. They are
**not renormalized**. Using the cosine expression after merging would change
both the kernel and its diagonal, invalidating the coefficient formula.
The public operator does not normalize q/k or the initial state. A model may
normalize its input embeddings before calling it, while passing returned centers
unchanged into subsequent streaming calls.

## Per-token transition

For scalar or per-physical-slot log decay `g_t`, write strength `beta_t`, selected
slot `j`, and merge fraction `gamma_t`:

```
Cminus_i = exp(g_t,i) Cprev_i
h_i      = kappa(Uprev_i, k_t)
e        = v_t - sum_i h_i Cminus_i
b        = beta_t e
z        = (1-gamma_t) Uprev_j + gamma_t k_t
a        = kappa(z, Uprev_j)
s        = kappa(z, k_t)
Cnew_j   = a Cminus_j + s b
Unew_j   = z
```

Other centers remain unchanged and other coefficients equal `Cminus_i`.
Output reads the **post-merge** state. No full Gram matrix or inverse is required.
For a fixed center `z`, the merged coefficient is the exact RKHS least-squares
optimum for the two atoms. With `e_j` the slot basis vector, the coefficient
transition is

`Cnew = [I + e_j ((a-1)e_jᵀ - beta_t s hᵀ)] Cminus + beta_t s e_j v_tᵀ`.

The bracket is identity plus a rank-1 correction. This statement applies for a
fixed selected route and center; coefficient-dependent selection makes the full
transition nonlinear in the value state.

## Two policies

- `mode='gated', routing='cyclic'` uses caller-supplied `merge [B,T,H]` and
  `j=(offset+t) mod M`. Gradients propagate through the gate and moved center.
  This is the discussion's input-gated transition, **not** BSGD's optimized
  center search. Expected gate domain: `0<=merge,beta<=1`, `g<=0`.
- `mode='grid'` searches `grid_size` uniformly spaced fractions in `[0,1]`
  (default 17). `routing='cyclic'` searches only the designated slot;
  `routing='min_error'` searches all pairs consisting of one old slot and the
  incoming atom. For old coefficient `c` and incoming coefficient `b`, it
  minimizes the actual vector-valued RKHS degradation
  `||c||² + ||b||² + 2 kappa(u,k)<c,b> - ||a c + s b||²`.
  Endpoints are included; for a fixed pair the selected error is no greater than
  either endpoint candidate's error. It does not consider old-old merges,
  fractions outside `[0,1]`, or a continuous/global optimum. The paper's
  normalized preimage surrogate and smallest-coefficient pair heuristic are
  different. This mode is explicitly an adaptation of its maintenance objective.

Grid choices use smallest slot index and then smallest fraction on exact ties.
The backward treats these discrete choices as fixed and differentiates the
selected branch. Away from switching boundaries this is the exact derivative of
the finite-choice implementation; there is no derivative through `argmin`.
`merge` is ignored and has zero gradient in grid mode. Do not infer the paper's
optimization/regret guarantees for either memory transition.

## Implementation and API

- `komp_slot/bsgd.py`: Triton forward and manually derived first backward, including
  q/k/v, decay, beta, merge gate and both initial states.
- `komp_slot/bsgd_reference.py`: independent PyTorch/FP64 token-loop reference.
- `validate_bsgd.py`: outputs/states/all gradients, scalar and slot decay, irregular
  widths, extreme gates, identical centers, arbitrary-cut streaming, suffix
  interventions and future gradients.
- `benchmark_bsgd.py`: same-GPU timing beside block KOMP C=64.

```python
from komp_slot.bsgd import bsgd_merging

out, state = bsgd_merging(
    q, k, v, log_decay, beta, gamma, (centers, coefficients),
    mode='gated', temperature=0.5, offset=processed_tokens,
)
```

`q,k [B,T,H,D]`, `v [B,T,H,V]`, `beta,gamma [B,T,H]`,
`log_decay [B,T,H]` or `[B,T,H,M]`. Output uses `v.dtype`; final centers and
coefficients are FP32. FP32/BF16 tensors, `M<=64`, `D,V<=128`. Gates are
caller-produced and not clipped or sigmoid-transformed inside the operator.
Only shape/device/dtype/configuration validation is done in Python; all numeric
forward/backward work is Triton. Higher derivatives are unsupported.

Pass the number of preceding tokens as `offset` for cyclic streaming. Unlike
block KOMP, merging happens on every token, so arbitrary token cuts preserve
the recurrence and do not cause an extra maintenance operation.

## Parallelism and limitations

The original `bsgd.py` is a correctness-first recurrent prototype: one CTA per
batch/head, serial tokens. The gated mode's key evolution is independent of
coefficient values; `bsgd_chunk.py` uses a separate geometry pass and affine chunk
scan. The grid policy depends on the current coefficient state, so that
same precomputed-map construction cannot be applied unchanged.

Training stores pre-token centers and decayed coefficients, requiring
`O(B*T*H*M*(D+V))` FP32 tape. Inference skips this tape. No checkpoint/recompute,
chunk recurrence, Tensor Core optimization or full-model training claim is made.
Block KOMP may be faster despite doing a more expensive projection because its
implementation is parallel across chunks. Both mathematical semantics and the
intra-chunk memory budget differ, so operator timings do not establish equal
model quality.

## H100 acceptance, 2026-11-01

The frozen snapshot passed all **402** checks on H100 80GB, Torch 2.10.0a0,
Triton 3.7.1. Ordinary FP64-reference comparisons over outputs, final states and
all eight input gradients had worst relative RMS **1.97e-6 (FP32)** and
**0.002548 (BF16)**. Acceptance tolerances were 5e-4 and 0.015 respectively,
with an absolute-error fallback of 2e-6 for nearly zero gradients.

The suite also passed 60 individual suffix interventions, 12 future-gradient
checks (future gradient exactly zero), 30 streaming output/state comparisons,
separate output-only/state-only adjoints, exact duplicate-center cases, and
coefficient-optimality/endpoint-removal algebra checks. Input gates include
beta/gamma 0 and 1 and log decay -80. These are tested-condition results, not a
formal all-input causality proof. Discrete search switching surfaces remain
nondifferentiable and near-ties can select different branches at different
precisions.

CUDA Graph timings below exclude compilation; each value is the median of three
event-timed trials with three graph replays per trial. All q/k/v are BF16;
states/gates are FP32. `grid` uses 17 center fractions. The block baseline is the
**current snapshot**, which differs from the earlier block acceptance run;
its dependencies are frozen alongside the new prototype.

| B/T/H/D/V/M | mode | forward ms | forward+backward ms | extra training GiB |
|---|---|---:|---:|---:|
| 2/256/4/32/32/16 | gated cyclic | 0.282 | 0.850 | 0.0084 |
| 2/256/4/32/32/16 | grid cyclic | 0.340 | 0.958 | 0.0084 |
| 2/256/4/32/32/16 | grid all incoming-old pairs | 0.398 | 1.008 | 0.0084 |
| 2/256/4/32/32/16 | block KOMP C64 | 0.055 | 0.109 | 0.0074 |
| 2/2048/4/64/64/32 | gated cyclic | 3.183 | 13.784 | 0.2584 |
| 2/2048/4/64/64/32 | grid cyclic | 7.653 | 19.285 | 0.2584 |
| 2/2048/4/64/64/32 | grid all incoming-old pairs | 7.704 | 19.823 | 0.2584 |
| 2/2048/4/64/64/32 | block KOMP C64 | 0.114 | 0.272 | 0.0973 |

The serial prototype is much slower than block KOMP. Avoiding Gram inverses
alone does not yield higher throughput: this implementation has no token/chunk
parallelism. No Depo/Brevo training was run and no quality claim follows from
these operator checks. The subsequently added gated chunk implementation separates
value-independent center evolution from rank-1 value transitions and builds parallel
chunk maps; the coefficient-dependent grid policy cannot reuse that algorithm unchanged.

GPU results, environment, logs and frozen source fingerprints are saved under
`evidence/bsgd-20261101-r1/`. The job completed with exit code 0; its exact alias
was removed after collecting evidence, while GPFS files are retained.
