# LR/WD and 10x scaling evidence

Job: `komp-lrwd-scale-20261031-r1` / `job-3aa5b7bc9830-20261031225931`, WeaverRSI, three H100 GPUs.
Remote root: `/gpfs/users/linlizhi/komp-lrwd-scale-20261031-r1`.

Source layers are the frozen decay training archive, the validated shortconv r2 overlay, then `lrwd-overlay.tgz`. No historical source or checkpoint was overwritten. The Triton cache is copied into the isolated root; the earlier cache stays unchanged.

Predeclared protocol is in `../../physics_lm/lrwd_runs/PLAN.md`: 27 validation-only search runs, three selected-config seed732 replications, three 88–90M-parameter/100M-token runs at seed731. The selection file is written before heldout evaluation. FLA commit remains `97874af2400916e2885c33ea30559a70ff686b66`.

Generated data: 1,420,992 complete documents, 100,000,690 nonpadding input tokens, 710,496 examples of each task. Includes the frozen 10M train corpus; fixed val/test/confirm manifests are identical to the previous experiment. `data100m-manifest.json` records generation, exclusions, source hashes and SHA256.

Final model checkpoints remain under the remote `physics_lm/lrwd_runs/results/` directory. Local evidence will contain summaries, manifests, selection, per-step metrics, heldout generations, environment, terminal status and exact-job cleanup receipts.
