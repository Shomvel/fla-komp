# Exact-KOMP learning-rate screen

Alpha0=.995, beta0=.2, temperature=.5, ridge=1e-4; FP64 projection, FP32 state, BF16 model. Independent10M schedules, seed731. One central LR per task is reused without additional training.

| Task | LR | Source | Status | Final validation CE |
|---|---:|---|---|---:|
| depo2 | 0.00025 | new10M run | completed | 1.098166044 |
| depo2 | 0.0005 | reused original alpha0-.995 | completed | 1.088958905 |
| depo2 | 0.001 | new10M run | completed | 1.094426539 |
| brevo2 | 0.0005 | new10M run | completed | 1.295064607 |
| brevo2 | 0.001 | reused original alpha0-.995 | completed | 1.257567771 |
| brevo2 | 0.002 | new10M run | completed | 1.239847671 |
