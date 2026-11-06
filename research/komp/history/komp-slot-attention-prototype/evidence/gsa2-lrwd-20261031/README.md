# Native FLA GSA2 experiment

Requested as an additional arm in the ongoing LR/WD and 10x scaling study. Native layer source is pinned to FLA commit `035f54b64e22abdb640725b3cf13db47d33692c5`, downloaded as the immutable `fla-source.tgz` archive. The layer and kernels are not edited.

First submission: `komp-gsa2-lrwd-20261031-r1` / `job-ec72d6ce8873-20261031231946`. Failed during import, before numerical preflight/training: the operator-only loader skipped the native module initializer and its `fla.modules.l2norm` alias. Static review found both Oja2 and GDN2 use that alias. The adapter now executes upstream `fla.modules.__init__` unchanged, restoring all native exports/aliases. A preflight assertion verifies alias identity. Failure evidence is retained and the failed job was deleted by exact alias.

Second submission: `komp-gsa2-lrwd-20261031-r2`, one H100, isolated remote root `/gpfs/users/linlizhi/komp-gsa2-lrwd-20261031-r2`. Dataset symlink points read-only to the main study's already-frozen 100M corpus; no data is regenerated or overwritten.

Adapter source is `../../physics_lm/gsa2_experiment.py`; protocol is `../../physics_lm/gsa2_runs/PLAN.md`. GSA2 uses native 64 slots and both native memories, 8192 recurrent elements per head versus 4096 in the preceding arms. Native q/k/v/w convolutions and initialization are retained. Weight-decay exclusions follow the layer's flags; residual learning-rate scaling applies only to its output projection and the backbone FFN down projection, not the slot-code `w_down` projection.

## Completed 2026-11-01

All 11 GSA2 runs and 3 heldout evaluations completed. Selected LR=0.001, WD=0.1 on final full validation CE. Large run consumed all 100,000,690 input tokens; confirm CE=0.4451576002, Depo=106/128 (82.8125%), Brevo=109/128 (85.15625%). Final archive `final-evidence.tgz` contains metrics, summaries, generations, selection and logs. `final-verification.json` records 286 passed initial/final prefix interventions, exact checkpoint CE readback for all 11 runs, and checkpoint/script SHA256 values.

Saved Completed job status, deleted exact job `job-0702dee35048-20261031232239`, and confirmed absence; see terminal-status.json, cleanup.json and after-cleanup.json. GPFS checkpoints and data remain. Combined report is `../../physics_lm/comparison_runs/RESULTS.md`. Native GSA2 has 103,709,872 parameters and 8192 recurrent elements/head; comparisons with the 4096-element prior arms are not equal-state ablations.
