# Block KOMP — GDN-style slot decay + learned eviction, 340M

User authorized one independent 8-H100 long-training run on 2026-11-07.

This ports the original `komp_learned_gdn_slot` variant from
`physics100m-komp-variants-20261102/suite100m/variants.py` (archived under evidence).
Every token projects a per-head score. Each chunk retains the M highest-scoring
old/new keys, ties prefer newer keys. The same score modulates write strength as
`beta * 2 * sigmoid(score)`; the hard ranking itself has no gradient. Initial
scores are fixed zero buffers. Evicted coefficients are projected onto retained
keys. This is neither FIFO nor a fixed learned-slot dictionary.

Decay is per physical slot: `g = -exp(A_log) * softplus(Wx + dt_bias)`.
A_log starts at zero, dt_bias matches alpha=0.995; both have zero weight decay.
Original full D→D SiLU output gate and full D→H*M decay projection are retained.
No KDA three-option low-rank/sigmoid/multi-timescale additions.

D1024 / L24 / H16 / K64 / M32 / C128 / FF1824, 340,372,736 parameters,
0.11939% above original KOMP339,966,848. Legacy seed42 shared weights and initial
keys are reproduced layer-by-layer; only the new gate/decay parameters differ.

FineWeb-Edu253 shards, frozen flame32K, heldout whole-document hash exclusions,
identical source data order and disjoint val/test sets. 6,800,000,000 supervised
tokens; seq2048, microbatch16, accumulation2, world8, global256. AdamW lr3e-4,
betas(.8,.95), eps1e-15, WD.1 except established no-decay groups and time constants;
warmup1024 then cosine to .1. Mixer-only activation checkpointing, native FLA
block/loop/MLP projections/fused linear CE, compatibility BF16 SiLU rounding,
FP32 parameters/residual/projection/state and BF16 Block tape.

Fresh model/Adam/RNG/loader begins only after new source-bound preflight passes.
Trial updates do not count toward training budget. Full/tail DDP, finite gradients,
replica equality, model/Adam/RNG/loader restoration, tokenizer/data hashes,
random/real FP64 operator comparison, learned retention/ties and score learning
must pass. Thresholds frozen in configs/native-acceptance.json before GPU results.
Indexed gradient atomics mean no bitwise replay promise.

Complete checkpoints every256 steps; validation every512; final test only.
No changes to existing training jobs. Project WeaverRSI, queue proj-32.
Read-only monitoring entry: reports/collect.py. Runtime progress: runs/block;
log: logs/train-block.log. All prior failed attempts must be preserved.

## Launch acceptance

2026-11-07 00:26 CST: source `d0238bb7a8a390f1ee1c834aef79bec02680775dd66fb150f9f93e24e846b9e9` passed new operator and 8rank full/tail acceptance. Initial and updated causal prefix differences were zero. All score/time-constant gradients and updates passed; full recovery receipts agree across8 ranks, 531 Adam states. Job `job-79815af6f373-20261107001646` is Running; fresh formal snapshot step28 / 14,680,064 supervised tokens, loss8.96564. Warm median2.6699s (~196369 token/s); rank0 peak35,449,789,440B (~33.02GiB). No formal checkpoint/validation yet (256/512). See evidence/launch-snapshot.json and preflight-block.json.
