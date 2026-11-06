# Exact per-token KOMP in Triton

This implementation inserts, reads and compresses the dictionary on **every
token**. It supports FIFO and the existing input-score learned eviction rule.
`compute_chunk=16/32` changes execution scheduling, not dictionary semantics.
"Exact" means the C=1 update rule, not bitwise real arithmetic or an approximate
CG solve. Numerical and training evidence is generated on an isolated H100.

## Interface

```python
from exact_komp import exact_komp

output, (final_keys, final_coefficients), metadata = exact_komp(
    q, k, v, log_decay, beta, (initial_keys, initial_coefficients),
    eviction="learned", scores=scores,
    compute_chunk=16,
)
```

The identical implementation is also installed in the existing prototype project
as `komp_slot/exact_komp/`, where the import is
`from komp_slot.exact_komp import exact_komp`. The original operator and kernels
were not modified. `evidence/integration-files.json` records matching hashes.

- `q/k`: `[B,T,H,K]`, FP32 or BF16, same dtype.
- `v`: `[B,T,H,V]`, FP32 or BF16. Output has the v dtype.
- `log_decay`: `[B,T,H,M]` or scalar-per-head `[B,T,H]`; normally FP32 and nonpositive.
- `beta`: `[B,T,H]`, the base write strength, normally FP32.
- `scores`: `[B,T,H]`, required for learned selection; normally FP32.
- Initial keys/coefficients: `[B,H,M,K]` and `[B,H,M,V]`.
- `1<=M<=64`, `1<=K,V<=128`, `T>=1`, CUDA only.
- `temperature=.5`, `ridge=1e-4`; both must be positive.
- Final keys and coefficients are FP32; selection metadata has no gradient.
- First-order differentiation only.

The learned default applies `beta_eff=beta*2*sigmoid(scores)`, matching the
existing learned experiment. Set `score_write_gate=False` if beta already
contains that gate, or when explicitly testing selection without the gate.
Hard selection has no gradient; the gate supplies the score gradient.

For streaming, carry both returned tensors, `metadata['scores']` and
`metadata['births']`; supply these as `initial_scores` and `initial_births`, and
increment `start_position` by the number of tokens already processed. FIFO also
needs the correct start_position to preserve its physical ring slots.

## Algorithm

After per-slot decay, write `beta*(v-prediction)` to a new key, read the current
query from the M+1 dictionary, then compress back to M keys. FIFO removes the
oldest key. Learned keeps the highest fixed insertion scores, preferring newer
atoms on ties. When the new key is rejected, its residual is still projected
onto the retained dictionary.

With physical slot e, projection p and key read vector h, keeping the new key
gives `S_new = D S + (p-e)e^T D S + e beta(v^T-h^T D S)` (rank two).
Rejecting it gives `S_new = D S + p beta(v^T-h^T D S)` (rank one).
The dictionary trajectory depends on keys/scores, not recurrent coefficients.
The update coefficients within a computational chunk therefore satisfy one
unit-lower-triangular system of size 2C. Its inverse is evaluated with a finite
nilpotent series, not an approximate iterative stopping criterion.

## Implementation

All numerical forward and backward stages are Triton kernels. PyTorch performs
allocation, layout conversion, API validation and autograd dispatch; it does not
solve the projection or recurrence in a fallback path.

1. FP64 q/key normalization and physical dictionary selection. Selection is a
   lightweight time loop per batch/head, independent of coefficients.
2. Parallel per-token Gram construction and FP64 symmetric-sweep solve. Retains
   the inverse for the analytical projection backward.
3. Parallel per-chunk rank-two maps using FP32 `tf32x3` tensor-core products.
4. Chunk-boundary state scan, followed by parallel intra-chunk state expansion
   and current-token output. No delayed dictionary compression.
5. Analytical reverse recurrence for coefficient, decay, beta and geometry
   derivatives; analytical Gram-solve and normalization backward. This first
   implementation's coefficient adjoint still scans tokens serially per
   batch/head/value tile.

FP64 geometry follows the validated precision baseline from the research
prototype. Near-duplicate keys made all-FP32 Gram arithmetic sensitive to
rounding; FP64 includes normalization and Gram construction, not just casting
an already rounded matrix before the solve. State arithmetic remains FP32.

The design references MesaNet's separation of parallel linear-system work from
state propagation, and its analytical solve backward. MesaNet maintains KK/KV
sufficient statistics; this kernel retains the KOMP dictionary and C=1
compression rule. The state/geometry identities are derived in the research
directory below.

## Verification and reproducibility

- `validate_gpu.py`: GPU differential outputs, final state/dictionary, all input
  gradients including learned scores, scalar and per-slot decay, BF16, duplicate
  keys, ties, accept/reject cases, stream continuation and causality.
- `train_smoke.py`: optimizer-level comparison against the frozen FP64 reference,
  using real GDN-slot decay, score gating, trainable dictionary and projections.
- `benchmark.py`: same-device CUDA-event forward/forward+backward timings,
  compilation excluded; includes wrapper and allocation costs.
- `reference.py`: unmodified frozen token reference; origin/hash recorded in
  `evidence/reference-source.json`.
- `job.yaml`: isolated WeaverRSI/proj-32 job, pinned image digest, one H100.
- GPU runtime and result JSON are stored in `evidence/`; numerical failures
  stop the task queue and are retained if any arise.

## Current limits

This is the first complete correctness implementation, not a performance win
claim. It saves O(BHT M^2) FP64 geometry and O(BHT MV) FP32 states; long-sequence
multi-layer training needs checkpointing/fusion. Per-token FP64 Gram solves,
selection and the reverse recurrence are remaining optimization targets. The
small optimizer test does not establish language-model quality versus C128.

Only FIFO and fixed input-score learned eviction are supported. A choice based
on the current coefficient norm or full deletion error cannot reuse the same
precomputed selection trajectory without further work.

## Sources

- MesaNet paper: https://arxiv.org/abs/2506.05233
- FLA implementation, inspected at commit
  `9e82a80fdaefcd7c9cc3da51ea36d6a9e1587bd1`:
  https://github.com/fla-org/flash-linear-attention/tree/9e82a80fdaefcd7c9cc3da51ea36d6a9e1587bd1/fla/ops/mesa_net
- Exact C=1 rank-two derivation and PyTorch prototypes:
  `/Users/shom/Downloads/mesanet-komp-research-20261105/README.md` and `LEARNED.md`.
- The symmetric Gram sweep follows the existing local KOMP geometry solver;
  its algorithm is implemented here in FP64.
