# Chunk-parallel soft-protected KAPA

`komp_slot/kapa_chunk.py` implements the same fixed-dictionary transition as
`protected_kapa`, with parallel chunks and all-Triton numeric forward/backward.
The raw-key entry point includes the existing Triton Gaussian/Gram geometry.

```python
from komp_slot.kapa_chunk import chunk_gaussian_protected_kapa

out, final_coefficients = chunk_gaussian_protected_kapa(
    q, k, v, log_decay, beta, centers, initial_coefficients,
    temperature=0.5, gram_ridge=0.01,
    anchors=2, regularization=0.05, chunk_size=32,
)
```

For precomputed features and inverse metric, use `chunk_protected_kapa` with the
same arguments as `protected_kapa`, plus `chunk_size=16`, `32` or `64`.
Both scalar and per-slot log decay are supported. Inputs use FP32/BF16 on one
CUDA device; the explicit inverse metric is FP32. M≤64, V≤128, and raw D≤128.
Output has v's dtype; final coefficients and intermediate arithmetic are FP32.
Only first derivatives are implemented.

## Exact chunk reformulation

Compute the protected write direction for every token independently:

    B_t = [h_t, h_(t−1), …, h_(t−p)]
    w_t = P B_t (B_tᵀ P B_t + λI)⁻¹ e₀

These solves use global token indices, so past-key anchors cross every internal
chunk boundary. Within each chunk, define D_i=diag(exp(g_i)) and
A_(i:j)=D_i D_(i−1)…D_j. Empty products are identity. Scalar decay is D_i=α_i I.
Writing u_i=β_i(v_i−h_iᵀD_i C_(i−1)) gives:

    C_i = A_(i:0) C_in + Σ_(j≤i) A_(i:j+1) w_j u_j
    L_ij = −β_i h_iᵀ A_(i:j+1) w_j       (j<i, otherwise 0)
    S_i  = −β_i h_iᵀ A_(i:0)
    T    = (I−L)⁻¹
    U    = T S C_in + T diag(β) V

The system is unit-lower-triangular. Its inverse is built with the existing
FLA-style block doubling routine in Triton, with explicit causal masks. Let:

    R_ij = q_iᵀ A_(i:j+1) w_j             (j≤i, otherwise 0)
    Q_i  = q_iᵀ A_(i:0)
    E_j  = A_(last:j+1) w_j
    O_S  = Q + R T S
    O_V  = R T diag(β)
    F_S  = A_(last:0) + E T S
    F_V  = E T diag(β)

Then every chunk provides two matrix equations:

    O     = O_S C_in + O_V V
    C_out = F_S C_in + F_V V

Scalar preparation fuses products, the triangular solve and all four maps into
one kernel per chunk/head. Slot decay first computes each causal matrix row in
parallel using C×M work, then composes the maps. This avoids materializing a
C×C×M tensor. All decay products use multiplication scans/reductions, never
prefix division; zero/underflowed gates remain supported. Padding tokens have
identity decay and zero writes/reads.

## Parallel forward and backward

Forward computes F_V V for every chunk, scans the affine chunk maps for all
boundary states, then evaluates outputs in parallel. The shared two-level scan
composes groups in parallel, scans the group boundaries, and expands each group
in parallel. Its serial depth is O(G+N_chunks/G), rather than O(T).

Backward similarly scans

    dC_in = F_Sᵀ dC_out + O_Sᵀ dO

in reverse chunk order. Each chunk then independently replays at most C forward
steps and C reverse steps to obtain gradients for q, h, w, v, log decay and β.
Value tiles reduce into FP32 coefficient gradients before the final dtype cast.
The existing protected-direction adjoint adds historical h and metric gradients;
the raw-key adapter additionally differentiates Gaussian features and inverse
Gram, including centers. No numerical computation falls back to PyTorch.

This is chunk parallelism, with a short serial replay inside each chunk's
backward; it is not a fully token-parallel backward or an O(log T) tree scan.
The local replay workspace is O(B H T M V), padded to whole chunks. Atomics in
shared history/metric/center gradients mean backward is not bitwise deterministic.

The dictionary remains fixed during the call. Final coefficients do not include
an anchor-history cache: splitting a sequence into separate calls still requires
an additional streaming cache interface. Internal chunking does not have that
restriction. Centers must not be derived from future tokens if causal sequence
modeling is required. Gram-conditioning limits are unchanged from
[KAPA_PROTOTYPE.md](KAPA_PROTOTYPE.md).

## Verification

`validate_kapa_chunk.py` compares feature and raw-key paths with independent FP64
PyTorch autograd. It covers all three chunk sizes; p=0/2/7 and raw p=4; scalar and
slot decay; FP32/BF16; M=1…64, D up to 128, V up to 128; short and partial chunks;
multiple scan groups; duplicate history/centers; zero decay and β=0/1; nonzero
initial state; noncontiguous inputs; output-only/state-only/prefix losses;
future interventions at both sides of chunk boundaries; and prefix truncation.

`benchmark_kapa_chunk.py` compares raw-key serial and chunk implementations on
the same H100, inputs and CUDA Graph timing. It includes Gaussian features,
Gram inverse, protected-direction solves, recurrence, and every input gradient.

## H100 results (2026-11-01)
All **1,608 checks passed**. Worst RMS relative errors against FP64:
- Feature path FP32: **2.16e−5**, including near-duplicate histories.
- Feature path BF16: **2.91e−3**.
- Raw path FP32: **7.75e−4**, including duplicate centers and ridge 0.001.
- Raw path BF16: **1.85e−3**.
- **210** future-input interventions and **30** future-gradient checks: exactly zero.

All **24 benchmark rows** completed. H100 80GB, Torch 2.10.0a0, Triton 3.7.1; pinned image and source hashes are in the evidence directory. CUDA Graph times below include raw-key geometry and all gradients, p=2. Chunk16 is the fastest training chunk size in each measured case.

| B,T,H,D,M,V | Decay | Serial fwd ms | Chunk16 fwd ms | Serial fwd+bwd ms | Chunk16 fwd+bwd ms | Training speedup |
|---|---|---:|---:|---:|---:|---:|
| 2,256,4,32,16,32 | scalar | 0.176 | 0.041 | 0.453 | 0.113 | 4.02× |
| 2,256,4,32,16,32 | slot | 0.174 | 0.044 | 0.413 | 0.116 | 3.56× |
| 2,2048,4,64,32,64 | scalar | 1.626 | 0.192 | 6.163 | 0.727 | 8.48× |
| 2,2048,4,64,32,64 | slot | 1.748 | 0.227 | 6.425 | 0.761 | 8.44× |
| 64,192,8,64,32,64 | scalar | 0.914 | 0.998 | 3.357 | 3.539 | 0.95× |
| 64,192,8,64,32,64 | slot | 0.940 | 1.190 | 3.325 | 3.763 | 0.88× |

Use `chunk_size=16` for the measured small-batch cases. At B=64,T=192, the serial kernel already has ample batch/head parallelism, and chunk preparation/scan overhead makes this implementation slower (about 5–13%). No automatic dispatch or universal speedup is claimed. Chunk16 also uses more peak incremental training memory: for B=2,T=2048, scalar 0.146→0.175 GiB and slot 0.147→0.177 GiB.

Evidence: `evidence/kapa-chunk-20261101-r1/`, including frozen source, SHA256 manifest, environment, logs and structured results. The job completed with exit code 0, was deleted by exact alias, and the cleanup query confirmed absence.
