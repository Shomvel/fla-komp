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
    compute_chunk=16, projection_dtype="float32",
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

## Second optimization pass

Default geometry and state arithmetic are FP32; matrix products use tf32x3.
`projection_dtype="float64"` enables diagnostic FP64 geometry. The final state
remains FP32 for either choice. See RESEARCH.md for the stable distance-based
Gaussian evaluation and the GDN/Mesa-inspired chunk forward/backward derivation.

No per-token M x V state or M x M Gram/inverse tape is stored. Forward contracts
readout into chunk maps. Backward scans only chunk boundaries, computes local
adjoints in parallel across chunks, reconstructs tile states in registers, and
recomputes geometry. Learned sparse windows share LDL factors and aggregate dictionary
gradients. Dense/FIFO tokens directly solve RHS with symmetric sweep. Long
sequences use a hierarchical boundary scan; local states use stable affine
prefix/suffix scans. Scalar-decay chunk interactions use matrix products;
per-slot decay retains stable relative exponentials.

All numerical forward/backward stages run in Triton. PyTorch allocates tensors,
converts layouts and dispatches autograd; there is no numerical reference fallback.

## Verification and reproducibility

- validate_gpu.py: frozen FP64 token-reference differential checks, including near
  duplicate keys, all gradients, streaming and prefix causality.
- edge_acceptance.py: zero beta, zero/tiny norms, scalar strong decay, optional
  FP64 mode, and complete 8K gradients against the original validated kernel.
- train_smoke.py: 12-step AdamW trajectory comparisons for FIFO and learned.
- compare.py: same-H100 8K old/new/GDN comparison, both FP32 and BF16 inputs,
  eager and CUDA Graph timings, incremental peak memory.
- profile_kernels.py: kernel attribution after warmup, kept separate from timings.
- baseline_komp/: unchanged previous FP32 optimized implementation for comparison.
- validate_scan.py / validate_versions.py: hierarchy, tails, exact prefix causality,
  version reuse and full gradients.
- ablations.py: reproduce independent changes using the frozen ablation_base_komp/.
  These candidate timings precede final dense-solver tuning.
- version_counts.py: actual factorization count on the seeded 8K inputs.
- reference.py: unchanged frozen token reference.
- evidence/: environment, hashes, numerical results, timings, task receipts.

## Current limits

The learned selector and coarse group-boundary scans still retain serial dependencies.
The per-token Gram solve remains O(M^3); this patch removes its tape and improves
precision/performance without replacing it by approximate CG. FP32 atomic adds
can have run-to-run rounding differences. Low batch/head counts also limit state
scan occupancy. Small optimizer tests do not establish language-model quality.

Only FIFO and fixed insertion-score learned eviction are supported. A choice
based on coefficient norms or deletion error needs a different trajectory.

## Sources

- MesaNet paper: https://arxiv.org/abs/2506.05233
- FLA implementation, inspected at commit
  `9e82a80fdaefcd7c9cc3da51ea36d6a9e1587bd1`:
  https://github.com/fla-org/flash-linear-attention/tree/9e82a80fdaefcd7c9cc3da51ea36d6a9e1587bd1/fla/ops/mesa_net
- Exact C=1 rank-two derivation and PyTorch prototypes:
  `/Users/shom/Downloads/mesanet-komp-research-20261105/README.md` and `LEARNED.md`.
- The symmetric Gram sweep follows the existing local KOMP geometry solver;
  its algorithm solves the RHS directly in the selected geometry precision.
