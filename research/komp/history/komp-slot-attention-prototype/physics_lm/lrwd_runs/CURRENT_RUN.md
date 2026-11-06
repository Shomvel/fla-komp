# Experiment complete — 2026-11-01

User requests fulfilled: LR/WD search, roughly 10x model/data scale-up, and native FLA GSA2 arm. No active study jobs remain. Do not restart training.

Workspace: /Users/shom/Downloads/komp-slot-attention-prototype
Final report: physics_lm/comparison_runs/RESULTS.md
Plots: lrwd-search.png/svg and scale-validation-curves.png/svg in the same directory.

44 training runs (36 small search, 4 seed732 replications, 4 large runs), 12 heldout evaluations. Every large run consumed 100,000,690 input tokens. Initial/final causality checks: 858 main + 286 GSA2 = 1144, all passed; all 44 checkpoint CE reload errors zero.

Large seed731 confirm results (CE / Depo / Brevo):
- KOMP 0.8316229 / 48/128 / 95/128; 90,204,472 params; 288.72 ms/step.
- GSA 1.4064801 / 21/128 / 50/128; 89,897,152 params; 137.91 ms/step.
- GDN 0.8634244 / 52/128 / 117/128; 88,050,352 params; 128.53 ms/step.
- Native GSA2 0.4451576 / 106/128 / 109/128; 103,709,872 params; 296.68 ms/step.

All selected LR=.001; WD=.1 except GSA=.01. Selected on small final validation CE; all LR optima at grid boundary. Small results averaged over two seeds; large results one seed. Historical fixed heldout used. GSA2 native64slots/8192 state elements per head vs others4096, different FLA snapshot and native layer initialization/recipe. Not equal-state or equal-parameter ablation.

KOMP Brevo errors: 32 wrong node sets/duplicates, 1 ordering-only. On same128 Brevo cases, GDN solves all95 KOMP successes plus22 more. This does not locate the internal cause.

Archives/verification/cleanup receipts: evidence/lrwd-scale-20261031 and evidence/gsa2-lrwd-20261031. All model.pt files remain remote under corresponding GPFS roots. Main alias komp-lrwd-scale-20261031-r1 and GSA2 alias komp-gsa2-lrwd-20261031-r2 deleted after evidence collection; absent verified. Failed GSA2 r1 also cleaned earlier. Unrelated komp-accel-20261031-dev job was not touched.

Native GSA2 source unchanged, pinned FLA035f54b64e22abdb640725b3cf13db47d33692c5. Earlier main arms pin97874af2400916e2885c33ea30559a70ff686b66. Do not overwrite frozen remote sources.
