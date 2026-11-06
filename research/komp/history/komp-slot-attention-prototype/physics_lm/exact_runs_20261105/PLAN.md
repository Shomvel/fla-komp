# Exact per-token KOMP: frozen Depo/Brevo protocol

Before observing outcomes, run four variants (scalar FIFO, per-slot FIFO, per-slot gate-only FIFO, per-slot learned) for seeds 731–737. Each uses the accepted exact KOMP opt2 source, algorithmic C=1, compute C16, FP32 geometry/state, temperature .5, ridge 1e-4. Learned applies the score write gate exactly once. Gate-only preserves the same score parameters but evicts FIFO. Hard selection and initial scores have no gradient path.

The dataset, named initialization, four-layer width128 backbone, M32, K/V64, batch64, padding192, one epoch/10,001,108 tokens/2231 steps, AdamW(.9,.95), WD .1, peak LR5e-4, 5% warmup, cosine and residual-output LR multiplier remain identical to block_runs_v4. No learning-rate selection or checkpoint selection uses confirm data. All 28 runs are retained.

Reuse the unchanged train/evaluate/generate functions. Report full val/test/confirm teacher-forced CE and the same first128 confirm generations/task. Audit saved generations with the original validators. Primary contrasts: learned minus FIFO, gate minus FIFO, learned minus gate; comparison with the matching historical Block C64 variants, GDN, GSA, gsa_slot and GSA2. Report seed-paired bootstrap intervals (20,000 draws, RNG91731), conditional on this fixed corpus/subset, without multiplicity correction.

Historical GSA2 has 64 slots, 8192 recurrent elements/head, native convolution/optimizer policy and both LR5e-4/LR1e-3 recipes. Exact KOMP and GDN have 4096 recurrent elements/head (learned metadata extra). Old gsa_slot excludes decay.bias from weight decay. These are reported qualifications, not matched architecture claims.

Run one fresh GDN seed731 control to assess reproduction. Performance from historical full training is contextual only; operator microbenchmarks are separate. Model preflight checks FP64 C=1 reference output, training decrease, causal interventions after training, nonzero score gradients and actual batch64 backward. Freeze and hash sources before launch; fail fast on the first command failure. Preserve checkpoints remotely, collect summaries/curves/generations locally, then clean up the exact owned GPU job.
