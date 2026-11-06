## Round 5 (2026-11-01): hotspot study

Each idea was first measured on its own, with an op-level bitwise comparison where the change should
be exact, and then kept or rejected. Register and spill counts come from the compiled Triton kernels
(`probes-regs-warps.txt`). All probe logs are in `evidence/accel-20261101e/`.

### Kept

| change | effect at 8k | numerics |
|---|---|---|
| `QN_BF16`: normalized queries stored in BF16 (only BF16 products read them) | -0.14 ms | bitwise identical |
| `SKIP_ZERO`: no writes of the all-zero `Nc`/`PEN` under SHARE | -0.05 ms | bitwise identical |
| `DOV_BF16`: `dOV` stored in BF16 (read only by BF16 products in `chunk_bwd`) | -0.06 to -0.10 ms | bitwise identical |
| `SLOT_GC_CACHE`: `ut` stores the rotated cumulative log decay; the other four kernels load it, and read its diagonal, last row and `gend` directly instead of masked reductions | per-slot about -0.4 ms | one scan result shared by all kernels (they had ulp-level differences before) |
| `chunk_bwd` with 8 warps for per-slot decay (`WARPS['chunk_bwd_slot']`) | per-slot -0.45 ms; spills 306 -> 26 | bitwise identical |
| `OVERLAP_DICT`: the dictionary-free `Hn, Rn` and the UT solve run on a side stream next to `dict_scan` and `proj` (learned) | learned -0.22 ms | forward bitwise identical |

### Measured and rejected

* Two-select Gram sweep (`SWEEP_FUSED`). The values are bitwise identical, but it was slower: 0.89 vs
  0.82 ms standalone for 8192 inverses at 2 warps, and +0.1 ms in the operator.
* Destination-major key gradients (`KEYGRAD_DEST`). Bitwise identical. `pool_norm` dropped by
  0.09 ms, but `keygrad` grew by 0.12 ms, so the net was +0.03 ms. The DU round trip is not what
  limits keygrad.
* Newton-Schulz refinement of the rank-1 sweep (CPU float32 check,
  `komp-accel-scratch/proto/rank1_refine_check.py`). It made the error 3x worse at Gram condition
  numbers near 6e5: the FP32 residual is not accurate enough there. On CPU the rank-1 sweep and the
  standard sweep have the same accuracy, so the GPU near-duplicate failure of round 4 is a rounding
  pattern, not an algorithmic loss.
* Shorter live ranges in `chunk_bwd`: scalar `Gam` broadcast instead of materialized, and `Dp`, `Dl`
  re-formed at use in a CSE-distinct form. Bitwise identical, but spills changed only from 140 to 124
  and the time did not change. The edits are kept because they cost nothing.
* 8 warps elsewhere: scalar `chunk_bwd` +0.2 ms (1 CTA/SM); learned `proj` / `proj_bwd` +1.3 to
  +2.7 ms.
* The per-slot scan is not the cost: a [64, 64] axis-0 `tl.cumsum` runs at copy speed (0.094 vs
  0.095 ms for 8192 tiles).
