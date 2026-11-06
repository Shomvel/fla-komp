# Exact-KOMP LR screen

Two additional8GPU groups, authorized separately from the original16GPU formal experiment. Maximum concurrent allocation32GPUs. Each task runs two new learning rates serially, with10M next-token positions per rate;40M new positions total. The central LR reuses the completed alpha0=.995 run with SHA-bound checkpoint, metrics and source receipts.

| Task | New LR | Reused LR | New LR |
|---|---:|---:|---:|
| Depo2 | .00025 | .0005 | .001 |
| Brevo2 | .0005 | .001 | .002 |

All other settings are fixed: alpha0=.995, beta0=.2, temperature=.5, ridge=1e-4, FIFO/M32/C16, FP64 geometry, FP32 state, BF16 model, seed731, batch128documents, WD=.1. Data order and the independent10M warmup/cosine schedule match the original hyperparameter screen. Final full-validation masked CE selects LR; no heldout/test selection.

Remote: /gpfs/users/linlizhi/physics100m-exact-20261105/exact_lr_tuning_20261105. Project WeaverRSI, queue proj-32. Dedicated aliases physics100m-exact-lr-depo2-20261105 and physics100m-exact-lr-brevo2-20261105. Training and checkpoint paths are isolated from both original tuning and formal runs.

Numerical/DDP acceptance is inherited only for the byte-identical operator and backbone, with source checks at launch. New LR outcomes are measured afresh. Source-manifest.json pins package and inherited sources. Do not edit those files while jobs run. STOP in this directory saves LR checkpoints at an optimizer boundary without stopping formal jobs; FAILED.json blocks new LR candidates pending diagnosis. No automatic retry is configured.

Evidence: submit-*.json, jobs-first.json, reuse-*.json and environment-*.json. Running is not acceptance: check results/{task}/{rate}/metrics.jsonl for actual world_size8 optimizer steps. Once both new candidates finish, evidence/group-{task}-complete.json is written. Sync JSON/JSONL without checkpoints, run report.py, retain results and then delete only the exact completed LR aliases.

## Final status

All four new 10M-position runs and the two reused central-LR runs are complete. The final source/configuration/data-order/budget/validation audit passed (`evidence/final-audit.json`). Both dedicated 8-GPU LR jobs completed and were deleted; all checkpoints are preserved. Depo2 selects LR=0.0005 (CE 1.088958905); Brevo2 selects LR=0.002 (CE 1.239847671). These short-run choices have not been applied to the original formal Exact runs. `evidence/startup.json` records historical pending status; use `evidence/current-status.json` for completion.
