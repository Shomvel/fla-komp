# Frozen-kernel trial plan

This plan was written before training results were inspected. All 70 planned runs use seeds 731–737, the verified frozen-20261101-r3 kernel directory, one H100, and the unchanged 10,001,108-token corpus. The kernel SHA256SUMS includes Block-KOMP and KAPA. Snapshots are isolated from the actively edited main tree. All preflights run in that snapshot and precede training; the launcher stops on the first failed command.

Five Block-KOMP arms: scalar FIFO C64, slot FIFO C64, slot score-gated FIFO C64, slot learned C64, slot learned C32. The new gate-only arm has the same score head, beta modulation and dormant initial-score parameter as learned, but still evicts FIFO. Scores receive gradients via the beta modulation, not through hard selection. The slot learned C32 arm changes the memory compression frequency, so it is not just a speed comparison.

Three baselines: GDN, GSA and per-token KOMP gsa_slot. The wrappers invoke unchanged experiment/decay harness functions while redirecting result paths. All baseline seeds are rerun, including 731/732. The original gsa_slot optimizer excludes decay.bias from weight decay; other arms use ordinary AdamW weight decay on all parameters. Thus comparisons across these harnesses include a small optimizer-policy difference.

Two KAPA arms: scalar and per-slot decay, p=2, chunk16, temperature=.5, Gram ridge=.01, constraint regularization=.05, M=32. Centers are learned parameters fixed during a sequence call. The Gaussian inputs are raw q/k/centers with no L2 normalization. These are prespecified pilot settings, not an optimized KAPA recipe. Chunk16 preserves the recurrence semantics; no streaming cache is used.

All arms use width128, four layers, two heads, K/V dimension64, FFN256, vocabulary32000, batch64, fixed right padding192 and one epoch (2231 steps). No short convolution is added. The same named initialization seeds are used for shared modules and the same deterministic data order is used for every training run. LR peaks at 5e-4, 5% warmup then cosine to10%; residual-output LR multiplier transitions .1→.3. AdamW betas(.9,.95), WD .1 except the inherited gsa_slot exclusions. No checkpoint selection or hyperparameter search uses confirm metrics.

Evaluation uses full validation/test/confirm splits for teacher-forced metrics; generation uses the same first128 confirm examples per task, greedy decoding, and the original validator. Depo valid is exact match; Brevo valid accepts any valid topological ordering of the required nodes. Saved generations will be audited against summary rates.

Primary contrasts are seed-paired learned−FIFO, gate−FIFO, gate−learned, slot-block−gsa_slot and slot-block−GDN. Additional prespecified contrasts: slot−scalar block, learned-C32−learned-C64, slot−scalar KAPA, slot-KAPA−slot-block and slot-KAPA−GDN. Report each arm's mean and sample standard deviation, and paired percentile bootstrap 95% CIs (20,000 matched-seed resamples, RNG91731). These intervals measure seed variability conditional on a fixed dataset/evaluation subset. They are not population-level data uncertainty and are not adjusted for multiple comparisons.

Gate−FIFO measures adding the learned write gate. Learned−gate measures adding selection on top of the gate, including interaction and changed score gradients. There is no standalone eviction-only arm, so the results do not identify an interaction-free causal contribution of eviction. Hard score ranking itself is nondifferentiable.

E3 is skipped: both sequence length and M are hardcoded shared constants in the existing harness, with no supported per-run override. We do not edit those shared constants.
