# Exact-KOMP alpha .995 single-factor comparison

Training 2/2; complete heldout 2/2.

Only initial alpha changed; original FP32projection/state, BF16network, per-taskLR/WD, seed731, exactdataorder and100Mschedule are retained. This is separate from the FP64short-run screen and excludes the Brevo LR=.002 winner. One seed only; original FP32stress numerical limitation remains.

| Task | alpha.98 valCE | alpha.995 valCE | delta | alpha.98 testCE | alpha.995 testCE |
|---|---:|---:|---:|---:|---:|
| depo2 | 0.757860 | 0.669557 | -0.088303 | 0.762959 | 0.671253 |
| brevo2 | 0.774105 | 0.734830 | -0.039275 | 0.779092 | 0.742074 |

Generation accuracy is reported in results.json only after all256probes are independently regraded.
