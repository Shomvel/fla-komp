# Validation-selected tuned Exact formal results

Existing100Mformal checkpoints selected only by final validation CE. No duplicate retraining or additional positions. Original frozen14-model/task table remains separate.

| Task | alpha | LR | Validation CE | Test CE | Generation |
|---|---:|---:|---:|---:|---:|
| depo2 | .995 | 0.001 | 0.630805 | 0.631022 | 20/256 (7.81%) |
| brevo2 | .995 | 0.002 | 0.708866 | 0.711004 | 30/256 (11.72%) |

All candidates have the same100Mbudget, FP32projection/state, BF16network, WD.1, seed731 and originaldataorder. Originalalpha.98 and alpha-onlycenter checkpoints are bound in manifest.json. These are tuning-selected results, not identically tuned comparisons against all other architectures.
