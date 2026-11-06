## Round 3 (2026-11-01): learned eviction in Triton, cross-kernel fusion, BF16

### Learned-score eviction in Triton

`block_komp_fast(..., evict='learned', scores=[B,T,H], initial_scores=[B,H,M])` now covers the
best arm of round 2. The ranking depends only on the scores, not on the values. A serial
`dict_scan_kernel` (one program per batch-head, one step per chunk) therefore writes two tables:
the physical slot to pool row map `OLD[c, slot]` and the kept candidates `CAND[c, slot]`. All chunk
maps then stay parallel, as in FIFO. They read the dictionary through `OLD`, and selection matrices
and eviction masks come from `CAND`. Key gradients go into pool rows with FP32 atomics, so the
learned path is not bitwise deterministic. The FIFO path is deterministic. When the scores equal
insertion times, the learned path reproduces FIFO (`learned_as_fifo` checks, 1e-4).

### Cross-kernel fusion and BF16

Switches are at the top of `block_fast.py`. The defaults are the fastest combination measured.

* `FUSE_SCAN`: the scan kernels compute the chunk offsets (`FV V`, `OS^T dO`) themselves, and the
  forward expand pass also writes the outputs. The separate offset and output kernels and their
  buffers are gone.
* `FUSE_CHUNK_BWD`: out_bwd + ut_bwd + decay_bwd become one kernel.
* `FUSE_POOL_NORM`: the pool-row gradient sum and the q/k normalization backward become one
  row-block kernel. There is no DPool round trip, and the backward no longer launches one program
  per row. `ROW_NORMALIZE` does the same for the forward normalization.
* `BF16`: with BF16 inputs, attention-side products (`_adot`) run on BF16 tensor cores. The
  projection side (Gram, inverse, P, J and their gradients) stays tf32x3 with FP32 storage.
* Kept off because they were slower: `FUSE_CHUNK_FWD`, i.e. att+ut+compose in one kernel (255
  registers at 8 warps, 1 CTA/SM: 2.9 vs 1.33 ms staged), and `FUSE_PROJ_BWD`.

### Other changes this round

* Per-slot decay with C a multiple of M: g is loaded with rotated columns. FIFO old atom u lives in
  ring slot (o+u) mod M, and new atom s in slot (o+s) mod M, i.e. column s mod M. This replaces the
  register gathers, and in `chunk_bwd` also the 'ieee' one-hot product. The learned path rotates only
  its new-atom part back, with one gather. 8k per-slot `chunk_bwd`: 3.35 -> 1.80 ms.
* Key gradients, FIFO with C >= M: the kept atoms are the last M new keys. Every product with Knew
  is therefore a product with Kn, and the kept-atom gradient lands on new-key rows. The
  projection-side factors are shifted into the attention-side ones at load time
  (`_keygrad_merged`). This reduces about 22 to 13 MMA passes per program and removes the
  register-gather placement. `KEY_BLOCK` 32 -> 64.
* `dict_scan` loads the next chunk's scores one iteration ahead and runs with 8 warps; per-slot
  `ut` uses 4 warps. These were picked by one-at-a-time sweeps (`evidence/accel-20261101c/tuning-8k.txt`).
* Rejected: rank-1 Gram sweep (`RANK1_SWEEP`, off). It writes each pivot step as a single
  `a - x x^T/p` with `x = a[:, j] - e_j` and removes the 2 I offset at the end. This is the same
  mathematics and about 0.2 ms faster at 8k. However, it failed the near-duplicate bound: dk RMS
  8.4e-3 against a limit of 7.2e-3 (1.25 x v0). Flag isolation showed that only this change moves
  the error (`near-duplicate-flag-isolation.txt`, `block-rank1-sweep-failed-validation.json`).
