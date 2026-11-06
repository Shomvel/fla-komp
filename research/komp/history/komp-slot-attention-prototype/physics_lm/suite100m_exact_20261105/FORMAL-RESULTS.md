# PhysicsLM 100M formal comparison

Training complete: 14/14. Heldout complete: 14/14.

Test CE appears after the full heldout split is scored. Generation accuracy appears only after all 256 prompt-only probes finish; a CE result alone is not counted as a completed heldout evaluator. Original selected recipes and budgets are retained. Exact uses alpha0=.98 and FP32 projection; the separate 10M tuning winners have not replaced its formal recipe. GSA2 uses the separately bound numerical-stability runtime with fresh 8GPU distributed acceptance; its strict wholebatch-Adam witness remains failed.

## depo2

| Model | Validation CE | Heldout CE | Generation accuracy |
|---|---:|---:|---:|
| GDN | 0.511347 | 0.513791 | 6.64% |
| Full | 0.744895 | 0.748124 | 3.91% |
| SWA | 0.912000 | 0.914779 | 3.91% |
| GSA | 0.918212 | 0.919615 | 2.34% |
| GSA2 | 0.650603 | 0.651842 | 4.30% |
| Block-KOMP | 0.697053 | 0.700365 | 5.08% |
| Exact-KOMP | 0.757860 | 0.762959 | 4.30% |

## brevo2

| Model | Validation CE | Heldout CE | Generation accuracy |
|---|---:|---:|---:|
| GDN | 0.668396 | 0.671432 | 14.84% |
| Full | 0.763570 | 0.769177 | 5.86% |
| SWA | 0.911020 | 0.919156 | 5.47% |
| GSA | 0.830668 | 0.836601 | 7.03% |
| GSA2 | 0.683094 | 0.689097 | 12.89% |
| Block-KOMP | 0.700359 | 0.704322 | 14.06% |
| Exact-KOMP | 0.774105 | 0.779092 | 11.33% |

## Comparison scope

Each architecture retains its own prior validation-selected LR/WD. These are one-seed results on two adapted PhysicsLM tasks. Exact formal alpha0=.98/FP32 projection differs from the separate alpha0=.995/FP64 short-run screen; that screen does not establish the latter recipe at100Mpositions. GSA2 uses a separately source-bound stable runtime, with the original wholebatch-Adam mismatch retained as a limitation.

| Task | Model | LR | Weight decay |
|---|---|---:|---:|
| depo2 | GDN | 0.0005 | 0 |
| depo2 | Full | 0.0005 | 0 |
| depo2 | SWA | 0.001 | 0.1 |
| depo2 | GSA | 0.001 | 0 |
| depo2 | GSA2 | 0.001 | 0.01 |
| depo2 | Block-KOMP | 0.0005 | 0.1 |
| depo2 | Exact-KOMP | 0.0005 | 0.1 |
| brevo2 | GDN | 0.0005 | 0.1 |
| brevo2 | Full | 0.001 | 0.1 |
| brevo2 | SWA | 0.001 | 0 |
| brevo2 | GSA | 0.0005 | 0.01 |
| brevo2 | GSA2 | 0.001 | 0.1 |
| brevo2 | Block-KOMP | 0.001 | 0.1 |
| brevo2 | Exact-KOMP | 0.001 | 0.1 |
