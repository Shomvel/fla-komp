# Fixed-dictionary soft-protected KAPA prototype

This prototype adapts the normalized KAPA-2 mechanism from Liu & Príncipe,
*Kernel Affine Projection Algorithms* (2008), §3.2 Eq. (30),
[doi:10.1155/2008/784292](https://doi.org/10.1155/2008/784292).
The research derivation is in
`/Users/shom/Downloads/rcdl-research/kernel-transition-literature-20261031/REPORT.md`.
The protected rank-one rule below is a sequence-memory adaptation, not an
implementation of the paper's original growing-dictionary algorithm.

## Kernel and state

For fixed dictionary U with M centers, use h(x) = κ(U,x), G = κ(U,U),
and coefficient state C ∈ R^(M×V). Prediction is h(q)ᵀC.
The convenience adapter uses the full Gaussian RBF
κ(x,y) = exp(−||x−y||²/(2τ)), default τ=0.5, without normalization.
P = (G + ρI)⁻¹, default ρ=0.01. The Gram ridge ρ regularizes the dictionary
coefficient metric; it is distinct from the constraint regularizer λ below.

The feature API accepts an explicit SPD P and symmetrizes it differentiably.
The Gaussian adapter constructs the actual regularized RKHS metric, not P=I.
Centers are trainable model parameters but stay fixed during one sequence call.

## Implemented transition

Let B_t = [h_t, h_(t−1), …, h_(t−p)] with missing history set to zero.
Default p=2; p=0…7 is supported. Let e₀=(1,0,…,0)ᵀ.

    w_t = P B_t (B_tᵀ P B_t + λI)⁻¹ e₀
    C⁻_t = diag(exp(g_t)) C_(t−1)
    r_t = v_t − h_tᵀ C⁻_t
    C_t = C⁻_t + β_t w_t r_tᵀ
    o_t = h(q_t)ᵀ C_t

Here g_t is scalar or per-slot, β_t is scalar, and default λ=0.05.
Thus the memory transition is rank one:

    C_t = (I − β_t w_t h_tᵀ) diag(exp(g_t)) C_(t−1)
          + β_t w_t v_tᵀ.

The anchors protect predictions **after decay**. They do not preserve the old
labels or undo forgetting caused by g_t. The desired constraint increment is
[e_current, 0, …, 0], not a window of supervised residuals. A separate PyTorch
`replay_kapa_reference` implements the fixed-dictionary rank-r replay rule.

The correction before β scaling is the minimizer of

    1/2 ||ΔC||²_(P⁻¹) + 1/(2λ) ||B_tᵀ ΔC − e₀ r_tᵀ||²_F.

λ>0 softens both fitting and protection; neither is exact. Independent feasible
constraints admit a hard λ→0 limit. Identical current/anchor keys with conflicting
requirements cannot fit the new value and preserve the old prediction exactly.
The validation includes this conflict rather than declaring protection successful.
For p=0, w=P h/(hᵀP h+λ), giving normalized kernel delta learning.

## Implementation boundary

- `komp_slot/kapa.py`: all feature-core direction solves, recurrence, and first
  backward are Triton. Small systems use a symmetric sweep inverse with FP32
  accumulation and TF32x3 matrix products. Shared metric and history gradients
  accumulate with FP32 atomics; bitwise deterministic backward is not promised.
- `komp_slot/kapa_geometry.py`: Triton Gaussian q/k features, Gram construction,
  symmetric sweep inverse, and analytical first backward. Distances use direct
  differences; Gram adjoint is −Pᵀ(dP)Pᵀ. Center gradients include both arguments
  of the Gram and both feature paths, accumulated in FP32 before a final cast.
- `komp_slot/kapa_reference.py`: independent FP64 PyTorch references.
  Production `gaussian_protected_kapa` uses **all-Triton numeric fwd/bwd**;
  PyTorch handles allocation, layout and autograd dispatch.
- Gradients cover q features, h features (including their historical anchor uses),
  v, g, β, P and initial C. The adapter also differentiates raw q/k and centers,
  including center influence through G and its inverse.
- Supported core tensors: one CUDA device, FP32/BF16; P is FP32; M≤64, V≤128.
  SPD P is a mathematical precondition, not checked by a GPU eigendecomposition.
  Raw geometry additionally requires D≤128 and emits FP32 features/metric.
  FP32 inversion requires a sufficiently resolved ridge; tiny ridge values can
  make duplicate dictionaries singular numerically. Only first-order backward
  is supported; λ, ρ and τ are scalar hyperparameters.
- The baseline in `kapa.py` is serial in T within a batch/head CTA.
  `kapa_chunk.py` now provides a chunk-parallel counterpart for scalar and slot
  decay, including backward; see [KAPA_CHUNK.md](KAPA_CHUNK.md).
- The returned state is coefficient C only. There is **no streaming anchor cache**;
  passing this state into another call loses the prior p features. Full streaming
  equivalence requires an anchor-history interface in future work.
- The dictionary is fixed: no KOMP replacement, center transport or BSGD merging.

## Acceptance and performance

`validate_kapa.py` checks GPU forward/final states and all trainable gradients
against independent FP64, p=0 normalization, independent primal proximal systems,
hard-constraint feasibility, duplicated/conflicting and nearly duplicated keys,
scalar/per-slot decay, β=0/1, exp(g)=0, future-input interventions and exact-zero
future gradients from prefix losses, output-only and state-only losses, and the
end-to-end Gaussian adapter. Both FP32 and BF16 are included, with odd sizes and
short histories.

`benchmark_kapa.py` uses CUDA graphs on H100. It times the feature core including
small solves and full first backward, excluding Gaussian/Gram geometry.
GPU evidence is in `evidence/kapa-protected-20261101-r1/`: frozen inputs and SHA256 manifest, pinned job YAML, environment, raw logs, terminal state and JSON results.

H100 80GB / Torch 2.10.0a0 / Triton 3.7.1: **441 checks passed**.
Worst ordinary FP32 RMS relative error: 1.46e−6; BF16: 2.39e−3.
Duplicate and near-duplicate histories with λ=0.01: worst 6.44e−5.
Gaussian adapter including all raw-key and center gradients: worst 7.13e−7.
Every future intervention and prefix-loss future-gradient check was exactly zero.
These results validate the tested configurations, not universal conditioning.

| B,T,H,M,V | Anchors p | Forward ms | Forward+backward ms | Extra training GiB |
|---|---:|---:|---:|---:|
| 2,256,4,16,32 | 0 | 0.173 | 0.405 | 0.0048 |
| 2,256,4,16,32 | 2 | 0.174 | 0.407 | 0.0049 |
| 2,256,4,16,32 | 4 | 0.175 | 0.408 | 0.0050 |
| 2,2048,4,32,64 | 0 | 1.629 | 6.014 | 0.1391 |
| 2,2048,4,32,64 | 2 | 1.638 | 6.035 | 0.1397 |
| 2,2048,4,32,64 | 4 | 1.647 | 6.048 | 0.1412 |

The serial recurrence dominates these timings. This experiment does not measure
training quality, capacity gains, or relative speed against GSA/GDN. The chunk-parallel implementation is now in `kapa_chunk.py`; its separate
acceptance and same-GPU comparisons are documented in [KAPA_CHUNK.md](KAPA_CHUNK.md).


## Raw-key all-Triton geometry acceptance (2026-11-01)

`gaussian_protected_kapa` now calls `komp_slot/kapa_geometry.py` instead of the
PyTorch reference adapter. q/k/center inputs may be FP32 or BF16; q/k features,
Gram, inverse and gradient accumulation are FP32. The center gradient includes
Gaussian query features, Gaussian key features, and the inverse Gram path.
No input normalization or detached metric is introduced.

On H100, the updated adapter passed all **441 existing checks** plus **690 new
geometry/end-to-end checks** against independent FP64 PyTorch autograd. New
coverage includes M=1…64, D=1…128, odd sizes, noncontiguous inputs, scalar/slot
decay, individual feature/inverse loss paths, nonzero starting state, duplicate
and nearly duplicate centers, and Gram ridges 0.01 and 0.001.

- Ordinary FP32 worst RMS relative error: **1.06e−6**.
- FP32 with exact duplicates, ridge 0.01: **2.72e−4**.
- FP32 with nearly duplicate centers, ridge 0.001: **1.30e−3**.
- BF16 worst RMS relative error: **2.68e−3**.
- All **80** new future-input intervention/future-gradient checks: exactly zero.

The test dictionary is fixed independently of the token suffix. These causality
results do not authorize constructing centers from future sequence tokens.
Very small ridges and arbitrary ill-conditioned dictionaries remain outside the
validated numerical range. FP32 atomic center accumulation is not bitwise
reproducible. The measurements in this section use the serial adapter; the separate chunk
entry point retains the same missing streaming anchor-cache limitation.

Evidence: `evidence/kapa-geometry-20261101-r1/results/`. The numerical suites
passed; that job then aborted because the old PyTorch `torch.linalg.solve`
adapter cannot be captured by the CUDA Graph benchmark. The benchmark harness
was audited and changed to identical eager timing for both adapters; new Triton
Graph timings are reported separately in `evidence/kapa-geometry-20261101-r2/`.
No numerical kernel correction or repeated numerical run was needed.


Both adapters below use the same BF16 raw inputs, p=2, FP32 gates/state,
and synchronized eager wall-clock timing (median of five batches of ten calls).
The measurements include raw features, inverse Gram, direction and recurrence;
training includes every input gradient. The old adapter is the unchanged PyTorch
reference geometry feeding the same Triton feature core.

| B,T,H,D,M,V | Geometry | Forward ms | Forward+backward ms | Extra training GiB |
|---|---|---:|---:|---:|
| 2,256,4,32,16,32 | PyTorch | 0.607 | 1.753 | 0.0205 |
| 2,256,4,32,16,32 | Triton | 0.207 | 0.618 | 0.0053 |
| 2,2048,4,64,32,64 | PyTorch | 2.814 | 8.014 | 0.6371 |
| 2,2048,4,64,32,64 | Triton | 1.640 | 6.171 | 0.1456 |

Eager full training improves **2.84×** at T=256 and **1.30×** at T=2048.
Triton-only CUDA Graph forward/training: **0.176/0.453 ms** at T=256,
**1.625/6.242 ms** at T=2048. Graph and eager numbers have separate timing
semantics and normal run-to-run variation. The long-sequence bottleneck remains
the serial recurrence; this is not a chunked or GSA-speed claim.

Both isolated jobs were deleted by exact alias after saving logs, terminal status,
source manifests and results; their cleanup queries returned empty lists.
