# Exact beta × M formal comparison

New training 6/6; new heldout 6/6; completed baselines reused 2/2.

| Task | beta0 | M | Parameters | Validation CE | Test CE | Generation |
|---|---:|---:|---:|---:|---:|---:|
| depo2 | 0.2 | 32 | 100,526,392 | 0.630805 | 0.631022 | 20/256 |
| depo2 | 0.1 | 32 | 100,526,392 | 0.662867 | 0.663171 | 17/256 |
| depo2 | 0.2 | 64 | 102,742,072 | 0.620201 | 0.621772 | 11/256 |
| depo2 | 0.1 | 64 | 102,742,072 | 0.571101 | 0.571875 | 16/256 |
| brevo2 | 0.2 | 32 | 100,526,392 | 0.708866 | 0.711004 | 30/256 |
| brevo2 | 0.1 | 32 | 100,526,392 | 0.744906 | 0.750315 | 32/256 |
| brevo2 | 0.2 | 64 | 102,742,072 | 0.707906 | 0.708997 | 31/256 |
| brevo2 | 0.1 | 64 | 102,742,072 | 0.727021 | 0.734299 | 35/256 |

Alpha .995; FP32 projection/state; BF16 network; seed 731; WD .1. Depo2 LR .001 and Brevo2 LR .002. Width512 / 15 layers / FF1984 remain fixed. Only final validation CE selects cells. M64 is 102.74M parameters, compared with M32 at 100.53M.

Original near-collinear FP32 stress failure remains; representative shape and DDP preflights do not imply all-stress acceptance

## Measured cost

Warm benchmark: one GPU, batch1 × 1024 tokens, full-model forward/backward with activation checkpointing, no optimizer/DDP. Training wall time includes cold shape compilation and final validation; rank0 training peak includes optimizer state. These are different measurements.

| Task | beta0 | M | Training wall minutes | Training peak GiB | Warm fwd+bwd ms | Warm peak GiB | Heldout wall minutes |
|---|---:|---:|---:|---:|---:|---:|---:|
| depo2 | 0.2 | 32 | 9.98 | 2.766 | 182.52 | 0.827 | 0.82 |
| depo2 | 0.1 | 32 | 10.00 | 2.766 | 185.01 | 0.827 | 0.82 |
| depo2 | 0.2 | 64 | 49.74 | 2.966 | 190.59 | 0.870 | 44.00 |
| depo2 | 0.1 | 64 | 20.04 | 2.966 | 181.87 | 0.870 | 43.74 |
| brevo2 | 0.2 | 32 | 13.89 | 2.686 | 187.23 | 0.827 | 2.39 |
| brevo2 | 0.1 | 32 | 14.10 | 2.686 | 190.36 | 0.827 | 1.91 |
| brevo2 | 0.2 | 64 | 42.07 | 2.865 | 186.02 | 0.870 | 110.91 |
| brevo2 | 0.1 | 64 | 22.64 | 2.865 | 189.32 | 0.870 | 111.03 |

The first M64 candidate pays new-shape compilation; the later M64 candidate reuses the shared cache. Native unpadded generation also pays per-length M64 compilation: heldout wall time was about44minutes on Depo2 and111minutes on Brevo2. These are observed cold-cache evaluation costs, not steady-state decoding throughput. Wall-time differences are not pure arithmetic comparisons. M64 doubles dictionary key/coefficient state and quadruples M×M map storage; fixed-backbone parameters rise by 2,215,680 (+2.20%).

Validation-only winners after all four cells finish: {'depo2': {'beta0': 0.1, 'slots': 64, 'validation_ce': 0.5711007879179516}, 'brevo2': {'beta0': 0.2, 'slots': 64, 'validation_ce': 0.7079059019078068}}.
