# Block KOMP Value Decay 340M training

Independent user-authorized 8-H100 run, alias `komp340m-block-value-decay-20261106`, identity `job-a8bbce53a014-20261106195012`, WeaverRSI/proj-32. Source SHA: `211b49fdea7eb41c878491735332303db14d8dc4f6486bbcf7dad62752e305be`.

Native FLA model, D1024/L24/H16/K64/M32/FF2272, 341,552,000 parameters. Value-column decay, rank64 low-rank decay/output gates, sigmoid output and head-wise multi-timescale initialization; alpha geometric mean .995, beta .2. Fresh native seed42 initialization differs from the legacy baseline initialization. This is not a pure decay-only ablation.

The baseline data manifest/loader and tokenizer are retained: FineWeb-Edu 253 shards, flame32K, 6879 excluded heldout document hashes, validation/test each 4,194,304 supervised tokens. AdamW lr3e-4, betas(.8,.95), eps1e-15, WD.1 (norm/bias/keys exempt); warmup1024 then cosine to .1. Seq2048, microbatch16, accumulation2, global256, exact 6.8B supervised-token budget. BF16 autocast, FP32 parameters/residual/state/gates, mixer-only checkpointing. Complete checkpoint every256, validation512, test only at completion; retain latest two complete checkpoints within this run.

New source-bound preflight passed: random and real operand FP64 forward/state/seven-gradient comparisons, 8rank full/tail finite updates and identical replicas, fused loss alignment, tokenizer validation, and full model/Adam/RNG/loader restoration. Trial updates are excluded from formal training; formal initialization SHA matches the fresh preflight model. Preflight initial prefix difference0; after updates maximum absolute prefix difference .02216798, RMSrel .00152045, passing the frozen .003 relative or .0003 absolute budget. This is numerical causality, not strict1e-5 or bitwise equivalence.

Use `python3 reports/collect.py` for read-only current progress. Raw metrics: `runs/block/metrics.jsonl`; preflight: `evidence/preflight-block.json`; startup snapshot: `evidence/launch-snapshot.json`. Remote package is `/gpfs/users/linlizhi/komp-value-decay-block-340m-20c-20261106`. Do not resubmit while this identity exists. Existing baseline and combo jobs were not changed.
