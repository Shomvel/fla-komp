# Value Decay + learned + 123 Block — 340M / 6.8B

New user-authorized independent 8-H100 experiment, WeaverRSI / proj-32.
Alias: `komp340m-block-value-learned123-r2-20261107`.

D1024/L24/H16/K64/M32/C128/FF2272; 341,970,176 parameters.
Based on learned123: same legacy seed42 shared FF2272 backbone, output sigmoid,
rank64 gates, normalized head time scales, learned top-M eviction (ties newest),
and beta * 2 sigmoid(score) write strength. GDN decay -exp(A_log)*softplus(z+dt_bias)
now acts on 64 value columns, not 32 physical slots. A_log/dt_bias remain WD0.
This adds 811,008 parameters relative to learned123 (~0.238%). It does not inherit
the earlier Value experiment's native initialization or log-sigmoid gate formula.

FineWeb-Edu frozen 253 shards / flame32K / exclusions and heldout splits unchanged.
Fresh seed42, seq2048, mb16 accum2, global256, AdamW3e-4 (.8,.95), WD.1,
eps1e-15, warmup1024 cosine to .1, exact 6.8B supervised tokens. Mixer-only
checkpointing; BF16 network, FP32 state. Full checkpoint256, validation512,
test only at final completion. Trial updates excluded from formal budget.

Value coordinate transform reuses the learned Triton kernel. For cumulative
channel residual span >24, inverse-free Triton state scan uses learned key
geometry built in PyTorch. This fallback preserves learned selection. Neither
backend differentiates hard ranking; score learning comes from write strength.

Acceptance: independent token FP64 outputs/state/seven gradients, random
beta>1, full chunks+tail, forced and automatic stable fallback, explicit learned
routing, eight gate-option combinations, real 8rank full/tail finite updates,
192 gate learning checks, replica identity, complete model/Adam/RNG/loader restore.
Initial causal absolute <1e-5; updated abs<=3e-4 or RMSrel<=.003, inherited from
value-coordinate scaling. This is numerical causality, not strict/bitwise.
New source-bound preflight must pass before fresh formal training.

`python3 reports/collect.py` gathers health and matched-token validation against
learned123, previous Value, learned, combo Block, baseline and KDA.

Current source: `0b906d5b3e39c017e862c13c6e7474a741a00ccfbde9be654f10e1265fc8f763`.
R1 failed before training on a stable argsort API call; source/logs retained in attempts/r1.
R2 uses explicit keyword dim after auditing all local sorting calls.

## Verified launch

R2 identity `job-01b39cc9c07e-20261107130332` passed new-source operator, properties, full/tail 8rank update and full restore acceptance. Formal fresh training observed at step62 / 32,505,856 supervised tokens (2026-11-07 13:15 Asia/Shanghai). Initial model SHA `ee1531e6a76dbd72ae1de740867434a4f00e41e616efac393ea6846ca1918077`.

Initial prefix difference0; after trial updates abs0.00660488 / RMSrel0.000593817, passing the frozen numerical criterion. Do not call this strict or bitwise causality. Startup median2.34922s/step, rank0 peak36.1687GiB; later performance must be read from metrics, not extrapolated from this snapshot.
