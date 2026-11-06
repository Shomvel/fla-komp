# 340M KOMP language models at approximately 20 tokens per parameter

User scope: two independently trained models, Exact-KOMP and Block-KOMP,
eight H100 GPUs each, on Flame's general pretraining corpus. Maximum concurrent
allocation for this experiment is 16 GPUs, including acceptance jobs. Use
Northjob project WeaverRSI / Volcano queue proj-32.

## Fixed comparison

Both models use 339,966,848 parameters: width 1024, 24 layers, 16 heads,
head dimension 64, SwiGLU intermediate width 1824, 32K vocabulary, untied
embedding/output weights, short convolution width 4, and M32 dictionary state.
Shared parameters use identical initialization. Alpha starts at .995 and beta
at .2; FIFO dictionary eviction, temperature .5, and ridge 1e-4 are fixed.

Exact-KOMP compresses after every token, with execution chunk 16. Block-KOMP
uses chunk 128. Network computation uses BF16; projection and persistent-state
computation are FP32. The optimized Block implementation uses BF16 chunk-entry
backward tape, whereas Exact uses FP32 tape. This is an implementation/precision
difference in addition to compression frequency and must remain visible in reports.

The recipe follows the existing Flame general-data configuration: FineWeb-Edu,
original 32K tokenizer, sequence length 2048, global batch 256 sequences,
AdamW peak LR 3e-4, betas (.8, .95), epsilon 1e-15, weight decay .1 for the
configured decayed parameter group, 1024 warmup steps, cosine decay to 10%,
seed 42. The common FFN is sized for the new KOMP architecture; this is not a
claim that its tensor layout is identical to a Flame Transformer or GDN.

Each run has 6,800,000,000 effective input and supervised next-token positions,
or 20.00195 positions per actual parameter. At 524,288 effective tokens per full
update, this is 12,969 full updates plus 508,928 effective tokens in the final
update (12,970 updates total). The final padded computation has 15,360 masked
positions, which do not contribute to the objective or effective token budget.
Acceptance updates and their artifacts are recorded separately from formal runs.

## Data and acceptance

See DATA.md for heldout construction, exact-text exclusions, packing, source
fingerprints, and the frozen tokenizer contract. Train and evaluation must use
the same tokenization despite Transformers 5's incompatible AutoTokenizer
normalizer conversion. Both models consume the same deterministic rank streams.

Formal release requires source-bound numerical evidence, eight-rank gradient
and update checks, causality, replica consistency, real-data finite loss and
gradients, optimizer/model/RNG restoration, exact loader cursor and buffer
restoration, and matching release/source/data receipts. Running state is not
acceptance. A new failure is diagnosed statically before resuming validation.

The existing Exact FP32 near-collinear stress limitation is retained; passing
the representative new shapes does not establish universal numerical stability.
Record validation CE/perplexity during training and evaluate the fixed test
stream only at the end. Compare runs at equal effective token budgets; a single
seed does not establish statistical significance.

The sole GPU queue operator is the existing physics_100m_training worker.
The parent owns data auditing, reporting, and the half-hour heartbeat. Keep
the former PhysicsLM experiments and their checkpoints unchanged.

## User-requested batching change

Both runs initially migrated to microbatch 16 per GPU and two accumulation steps, retaining
the global batch of 256. Exact migrated at step 2304 / 1,207,959,552 tokens;
Block migrated at step 4096 / 2,147,483,648 tokens. Both stopped at complete
checkpoints with no completed unsaved steps discarded.

The original runtime and logs remain in the package root. Continuation source,
metrics, configurations, and checkpoints are under `migration-mb16/runtime/`.
Combine the original `runs/` metrics through each cutover with continuation
metrics after it; token counters remain cumulative, not a new training budget.

Source/data/batching-bound migration receipts are in
`migration-mb16/runtime/evidence/migration-exact.json` and
`migration-mb16/runtime/evidence/migration-block.json`. Both passed a real-data
global-batch update at the new shape, discarded that trial, and restored the
checkpoint model, Adam, RNG, and loader state before formal continuation.
The batch shape change can change floating-point trajectories; bitwise
equivalence to continued microbatch-4 training is not claimed.

## Immediate restart at microbatch 32

The user subsequently requested microbatch 32 / accumulation 1 and explicitly
instructed immediate task termination followed by restart from existing complete
checkpoints. Final stopped logs were Exact step2395 and Block step4226. Available
complete checkpoints were still Exact step2304 and Block step4096, so 91 Exact
steps (47,710,208 tokens) and 130 Block steps (68,157,440 tokens) must be recomputed.
The superseded microbatch-16 logs are preserved, not added to effective progress.

The continuation at that stage was `migration-mb32/runtime/runs/{exact,block}/`.
Both source/data/batching-bound migration receipts passed real-data batch32
updates and full state restoration; trial updates are discarded. Global batch
remains 256, the allocation remains two groups of eight GPUs, and each run's
effective target remains 6.8B tokens. For the retained training trajectory, join
the original microbatch-4 logs through the cutovers with the microbatch-32 logs
after them. The microbatch-16 branch contributes only migration and cost evidence.

## Native FLA model and requested microbatch 16

The user subsequently requested a FLA-style KOMP mixer and native FLA model
definitions, followed by microbatch16/accumulation2 (global256 unchanged).
The new target runtime is `migration-fla-mb16/runtime/`. It uses registered
KOMP configuration/model/mixer classes, native FLA block and model forward
loops, separate MLP projections, and native fused linear cross entropy.
The MLP activation explicitly preserves legacy BF16 rounding. Only the mixer
is checkpointed; neither the full layer nor the output head is checkpointed.

The preceding mb32 implementation stopped at Exact2491 and Block4458 without
rollback. The intermediate FLA mb32 version passed actual-shape Exact checks,
completed one formal update, and paused at Exact2492. The new B16 run resumes
that FLA checkpoint directly; Block resumes the old-format4458 checkpoint
using explicit weight and Adam name/slice mappings. Expected counters at
these starts are 1,306,525,696 and 2,337,275,904 effective tokens, respectively.

Strict per-tensor legacy trajectory comparisons and native training acceptance
are recorded separately. Earlier short-shape/B32 Block compatibility failures
remain preserved. The new B16 acceptance policy was frozen before its GPU
results and checks actual full/tail objectives, whole-model gradient/update
differences, finite values, causality, replica identity, and full state restore.
Source, detailed migration rationale and limitations are in FLA-MIGRATION.md.
Consult source-bound receipts and advancing metrics before claiming that a
model has entered formal training; a scheduled job is not acceptance.
