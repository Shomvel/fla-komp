## Round 4 (2026-11-01): C == M specialization, BF16 tape, merged key gradients

Every change is a switch at the top of `block_fast.py`. The defaults are measured on H100 and
validated against FP64.

* `SHARE_GRAM` (FIFO, C == M; this is the 8k test shape). The chunk-end selection keeps all C new
  atoms and evicts the whole old dictionary, so several projection-side quantities are fixed:
  * `Nc = 0`, `PEN = 0`, and `Jn` is a pure selection. In a full chunk, `Jn = I`.
  * `proj` skips the `Knew Kn^T` product. In full chunks `Knew = Kn`, so `proj` also writes `Hk` and
    `Hn` from its tf32x3 cosines, and `att` keeps only the query products.
  * `compose` builds `FS = Jo diag(gend) + diag(dend) WS` and `FV = diag(dend) WV` without products in
    full chunks.
  * `proj_bwd_a` replaces its four tf32x3 products with row-shifted loads (`Jn^T X` is X shifted by
    M-n rows) and drops `dJn`, which is never read. `proj_bwd_b` drops the `Zn` products, and
    keygrad skips the zero `CNC`.
  * The last partial chunk keeps the general formulas.
* `TAPE_BF16`: with BF16 inputs the chunk-entry states and their adjoints are stored in BF16. Only
  `mapgrad` reads them, and it multiplies them in BF16 anyway, so the results are bitwise unchanged
  (`max-rms-vs-off 0`).
* `MERGE_KEYGRAD` improved. `F_k` is `F_old` transposed, and the two `CNC` terms are `Y + Y^T`, so
  the factors are loaded once and transposed in registers. 8k keygrad: 1.84 -> 1.42 ms.
* Fixed while adding these: Triton also compiles statements after a constexpr-if `return`. `_mdot`
  and `_adot` now use if/else, because the BF16 tape exposed a dtype error in the dead branch.
  Runtime branches whose body only type-checks under a constexpr condition must be nested inside
  that condition (`compose_kernel`).
