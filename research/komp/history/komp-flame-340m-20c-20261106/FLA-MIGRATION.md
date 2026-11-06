# Native FLA integration contract

Status: implementation and acceptance in progress. This document describes the
required migration, not a release receipt or a claim that training has switched.

The user's requested implementation uses KOMP mixers inside native FLA model
definitions, replacing the handwritten layer loop and checkpointed 256-token
output-head loop. The existing worker owns implementation and GPU operations.
The mb32 runs saved complete checkpoints and paused for replacement testing:
Exact step 2491 / 1,306,001,408 tokens and Block step 4458 / 2,337,275,904
tokens. Each final logged step matches its checkpoint; this cutover discards
zero completed updates. Stop records are `migration-fla/stop-{kind}.json`.

## Preserved training experiment

Keep 339,966,848 parameters, 24 layers, width 1024, 16 heads of width 64,
intermediate width 1824, vocabulary 32000, untied embeddings/head, M32,
alpha .995, beta .2, and the existing Exact/Block compression rules. Network
autocast remains BF16, geometry/state FP32, and Block backward tape BF16.
Do not adopt native GDN's decay, normalization, initialization, or cache math
in place of KOMP's operators merely to reuse its model scaffolding.

The latest user instruction changes the target to microbatch16/accumulation2,
eight GPUs per model, global256, sequence2048. Preserve optimizer parameter
groups, AdamW state and schedule, tokenizer, rank-local
data order and the original cumulative 6.8B effective-token target per model.
All acceptance updates are discarded before formal continuation. Maximum
concurrent allocation, including acceptance, remains sixteen GPUs.

## Native components and precision

Use the frozen FLA model forward/block contracts, GatedMLP, and native fused
linear cross entropy. Preserve the residual stream and norm epsilon; fused
norm/residual precision changes require measured evidence. Disable blanket
whole-layer checkpointing. Selective mixer recomputation is allowed when
saved KOMP tape sizes require it. Measure memory at the actual B32/T2048 shape
before claiming that all recomputation can be disabled.

FLA fused linear CE still chunks internally. Its default eight chunks process
8192 positions each for B32/T2048, replacing the old 256 chunks of 256 positions.
It computes and retains input/weight gradients instead of retaining all logits
or checkpointing every head chunk. Actual throughput and peak memory must be
measured; fewer head calls alone do not establish a speedup.

## Objective and checkpoint mapping

The loader already returns inputs ids[:, :-1] and aligned next-token targets
ids[:, 1:]. Native causal LM normally shifts labels. The adapter must invoke
the native loss with explicit aligned shift_labels, or equivalently adapt the
input contract, so each supervised position is counted exactly once. The
frozen native FLAForCausalLM.forward does not itself forward shift_labels to
the loss helper. Preserve summed loss and existing DDP global-token scaling;
the final partial update must mask exactly the same tail positions.

The former combined MLP up matrix stores gate rows first and value rows
second. Map its first 1824 rows to native gate_proj and its remaining 1824
rows to native up_proj. Apply identical slices to Adam exp_avg and exp_avg_sq,
copy the scalar step to both, and preserve the original decay group. Map all
other tensors by explicit names. Loading old optimizer parameter IDs without
this name mapping is invalid. Require exhaustive source/target coverage and
an exact reverse mapping of model and Adam tensors before testing updates.

Preserve CPU/CUDA/Python/NumPy RNG, per-rank shard/file byte offset and token
buffer. Archive old source, logs and complete checkpoints. Record last logged
step versus loaded checkpoint; superseded updates are compute cost only,
never added to the cumulative effective-token budget.

## Release evidence

New source-bound evidence must include checkpoint-mapped real-input loss,
gradients and optimizer-update comparisons, full/tail target masking and
scaling, causal behavior, eight-rank finite updates and replica agreement,
and model/Adam/RNG/data-cursor restoration. Compare identical input shapes;
use documented tolerances and retain every failure. On the first failure of
a class, stop repeating dynamic tests and audit the relevant contracts first.
Preserve the prior near-collinear FP32 limitation; do not claim bitwise
equivalence after kernel/layout changes.

Before announcing completion, verify new formal metrics actually advance
after restoring the acceptance checkpoint. Report source and checkpoint
lineage, selected recomputation policy, loss implementation, warm-step time,
throughput and peak allocated/reserved memory, distinguishing cold compilation
from steady-state execution and before/after samples from controlled benchmarks.

## First native-fusion compatibility attempt

The first CUDA attempt stopped before formal continuation. Exact failed the
per-tensor Adam-update comparison for layer 0 MLP norm (relative RMS
0.035971 against 0.035); Block failed the layer 1 decay-weight gradient
comparison (0.043983 against 0.035). Strict weight mapping and independently
reconstructed Adam moments/steps passed. These failures are retained; they
are not release receipts and thresholds must not be silently relaxed.

Static investigation identifies distinct rounding paths requiring isolation:
native fused SwiGLU rounds the FP32 SiLU/product once, whereas the old BF16
SiLU then product rounds twice; the split MLP GEMMs can also differ. Fused CE
precomputes gradients and applies the external loss multiplier later, unlike
ordinary CE backward. With FP32 residual hidden states under BF16 autocast,
its forward linear quantizes hidden states but its FP32 `addmm(out=...)`
weight-gradient accumulation uses the original hidden states. A loss-boundary
cast and normalization placement can be tested without changing residual
precision or the mathematical training objective. None of these static
observations alone establishes the cause or magnitude of the failed checks.

The subsequent T257 checkpoint diagnostic isolates the MLP rounding change as
the main contributor. Native MLP with original / mean / mean-plus-BF16-input
CE yields 87 / 90 / 88 failed Exact comparisons and 390 / 386 / 386 failed
Block comparisons. Retaining separate native gate/up projections while
restoring the old two-rounding activation and mean-plus-cast CE produces zero
Exact failures and six Block failures. Packing projections again does not
fully resolve Block and is not the intended implementation. These are
diagnostic results, not authorization to label failed tests as passed.

The decisive continuation check must use the latest requested B16/T2048 with
two accumulation steps, eight ranks and global256, identical checkpoint and
input batch, including the final masked budget shape. Each rank has 65536
targets in a full update; the last update has 63616, split as 32768 plus
30848. Both microbatch losses must use the total local update budget for
scaling, with synchronization on the second backward. Record short-shape differences separately;
passing actual-shape continuation cannot be described as all-shape numerical
equivalence. The second candidate uses separate native MLP projections with
an explicit legacy-rounding activation adapter and native fused CE with mean
normalization/BF16 head inputs. It does not use the native fused SwiGLU
activation. Source `849003b9d20b8818353bb8a2ee7c935666e3426f419943c32cf95cad84e764fd`
was submitted for B32 validation before the latest user batching instruction.
It must not automatically start B32 formal training. A new source/batching
receipt for B16 with two accumulation steps is required before continuation.

The B32 candidate's actual-shape check passed for Exact. It completed one
formal update and then saved/paused at step 2492 / 1,306,525,696 tokens after
the user's B16 request. Continue from this FLA-format checkpoint directly;
do not apply the legacy name mapping a second time. Block did not enter
formal training: B32 full/tail comparisons retained 11/9 failed tensors,
maximum gradient relative RMS 5.226%/6.850% and update relative RMS
4.221%/4.209%, despite mean loss differences of 1.49e-7/1.79e-7. Its resume
source remains the old step 4458 checkpoint. These comparisons remain failed
compatibility evidence and must not be relabeled as universal equivalence.

## B16 continuation acceptance

The latest candidate is isolated at `migration-fla-mb16/runtime`, source
`1dde0fbb5668f1105425de6774a4d188535b278819d3af3f1c569a987359f5ea`.
Before its first GPU result, `configs/native-acceptance.json` declared two
separate outcomes. Strict trajectory compatibility retains the original
per-tensor criteria and any failed status. Native training acceptance requires
full/tail loss absolute difference below .005, whole-model gradient and update
relative L2 below .035, finite parameters and gradients, causal-prefix relative
RMS below 1e-5, identical updated replicas, complete Adam/RNG/loader restoration,
and correct aligned token accounting. Per-tensor errors and reference norms
remain in the receipt even if the native training criteria pass.

This distinguishes acceptable continuation with a new implementation from
the stronger claim of matching every tensor's numerical trajectory. The
candidate must not claim strict compatibility when that comparison fails.
Exact compares its native FLA checkpoint at B32x1 versus B16x2; Block compares
its legacy checkpoint at B16x2 versus native FLA at B16x2. Both preserve the
same 32 sequences per rank and global256 update.

Exact B16 native training acceptance and full-state restoration passed, with
formal metrics advancing from step 2493 onward. Full/tail loss difference was
zero; whole-model gradient relative L2 was .003452/.004388 and update
relative L2 was .003990/.004479. Strict trajectory compatibility remains
FAILED: two tensors in each case exceeded the per-tensor criterion. These
are different conclusions, both preserved in the receipt. Formal rank0 peak
allocated memory is 34,464,085,504 bytes (32.10 GiB). A 20-step warm sample
through step 2519 has a median of 5.3406 seconds per 524288-token update
(98,170 tokens/second across eight GPUs). This before/after observation is
not a controlled hardware benchmark.

Block subsequently passed native B16x2 training acceptance and full-state
restoration, and formal metrics advanced from4459 onward. Its strict
trajectory comparison remains FAILED (11 comparisons for each full/tail
case); whole-model gradient relative L2 is .012186/.012471 and update
relative L2 is .008411/.008757. Loss differences are approximately1e-7,
causal-prefix difference is zero and all eight updated replicas agree.
At step4492 it has2,355,101,696 effective tokens (34.63% of6.8B), loss2.83421,
20-step median2.5443s/update (206,066tokens/s), and rank0 peak33.08GiB.
It resumes checkpoint4458 with zero rollback and zero counted trial tokens.

At the same observation Exact is at2583 /1,354,235,904tokens (19.92%),
loss2.99902, with complete checkpoint2560. Its step2560 validation uses
4,194,304 supervised tokens: CE3.02045024 / perplexity20.50052. Block's
new-runtime validation is pending; the preceding step4096 validation belongs
to the older runtime and a different training budget. Both original B16 job
identities are running with sixteen GPUs total.
