# Exact / Block KOMP: KDA-inspired options 1+2+3

User authorized both architectures with all three options enabled, each fresh seed42, 6.8B supervised training tokens, 8 H100. This adds 16 GPUs to the original Exact/Block/KDA 24-GPU group, for at most 40 across these five jobs. Do not submit GDN or additional candidates. This package owns only aliases `komp340m-{exact,block}-kda123-20261106`.

## Independent options

`KOMPConfig` exposes `output_sigmoid=False`, `low_rank_gates=False`, `multi_timescale_init=False`, `gate_rank=64`, `timescale_spread=4.`. Explicit config defaults retain old behavior. `LM(kind)` loads this experiment's frozen `configs/options.json` (all three enabled); pass an explicit options dictionary for a separate future experiment. Each frozen experiment requires its own source manifest and preflight. Do not toggle flags in a running package.

1. Output gate SiLU -> sigmoid. KOMP geometry/recurrence unchanged; output normalization remains the old implementation, so no fused-norm numerical change is bundled.
2. Both gate projections use rank64: output1024->64->1024, no bias; decay1024->64->512 with final bias. No intermediate activation. Output down std.02/up std1/sqrt64 matches the old full gate's expected effective weight variance. Decay random down + zero up permits learning; both are not zero initialized.
3. Initial decay rates span head-wise factors from1/4 to4, normalized by their arithmetic mean. Thus mean(log alpha) remains log(.995). Within each head all32 slots begin at the same rate. Alpha range approximately .985825–.999108; pure-decay half-lives48.55–776.82 tokens (not a claim about actual memory after delta writes/compression). The prior beta=.2 is unchanged. New low-rank decay biases exactly encode these rates through the old logsigmoid/8 function.

Both models have307,723,136 parameters, versus baseline339,966,848. Keep D1024/L24/H16/K64/M32/FF1824. Lower gate parameter count is explicit: this experiment measures combined behavior at equal data budget and fixed backbone, not equal total parameter count. No FF widening. Initialize all shared weights exactly as the original seed42 model. Preflight compares the entire all-options-off model state with mapped legacy initialization, and checks8flag combinations.

## Shared recipe

FineWeb-Edu253shards; frozen32K tokenizer and the identical data manifest/cursor algorithm. Exclude6879heldout entire-document UTF8 hashes. Val/test each4,194,304supervised tokens. Sequence2048, microbatch16, accumulation2, world8, global256. AdamW lr3e-4 betas(.8,.95), eps1e-15, WD.1 with the original norm/bias/initial-key exemptions, clipping1; warmup1024 and100%-budget cosine to.1. BF16 network, FP32 residual/projection/state; Block BF16 backward tape, Exact FP32. FLA native block/loop, compatible BF16 MLP rounding, fused linear CE; mixer-only activation checkpoint. Tail global508928valid tokens; trial updates count zero. Only validation ranks results, test evaluated at final completion.

## Release checks and evidence

Frozen new-source operator checks include both random varying slot decays and real-text-derived initial activations at T129 across compression boundary; outputs/final state/all operand gradients vs FP64 reference. Existing FP32 near-collinear limitations are retained; passing representative cases is not a universal guarantee. All8flag combinations finite and full-off mixer output exactly matches old mixer at identical weights. Real8rank B16x2 full65536 and tail63616local budgets verify finite all-parameter gradients and updates, identical replica SHA, causal prefix (<1e-5), aligned fused CE vs explicit full/tail (<.005), peak<72GiB, exact model/Adam/RNG/loader restore. Source/data/batching-bound `evidence/preflight-{kind}.json` gates formal training. Full-off full model initialization is also checked before release.

Each formal job reconstructs fresh seed42 model and original data cursor, requiring the initial model hash to match its receipt. Independent runs/{kind}, logs/train-{kind}.log, complete checkpoints every256steps (latest2retained), validation every512, final val/test. Exact and Block shared initial model hashes must agree. Preserve original model baselines and other jobs. First dynamic error stops the launch chain; static diagnosis precedes a new source/release. Existing baseline bitwise migration failures are not erased or relabeled by these fresh runs.

Project WeaverRSI, queue proj-32; pinned same worker image digest. Sources and configs are frozen with freeze.py. Option APIs are in models.py and mixer_options.py; required checks in option_checks.py/preflight.py. No separate test-only GPU job is needed: each group's job executes its gate and then formal training using set-e.
