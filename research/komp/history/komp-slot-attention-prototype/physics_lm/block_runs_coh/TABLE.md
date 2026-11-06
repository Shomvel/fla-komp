Mean ± sample SD; seeds731–737. CE lower is better; generation percent higher is better. All arms use the same frozen FP32 flex kernel and10,001,108 training tokens.

| arm | n | content CE | Depo CE | Brevo CE | Depo gen % | Brevo valid % | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| coh_fifo_off | 7 | 2.0892 ± 0.0780 | 3.0426 ± 0.1526 | 1.1357 ± 0.0265 | 5.36 ± 2.57 | 35.27 ± 2.83 | 416.1 ± 7.3 |
| coh_fifo_99 | 7 | 2.1213 ± 0.0605 | 3.0580 ± 0.1035 | 1.1845 ± 0.0574 | 5.58 ± 2.23 | 32.81 ± 4.47 | 751.4 ± 10.8 |
| coh_fifo_95 | 7 | 2.4326 ± 0.0552 | 3.0669 ± 0.0773 | 1.7984 ± 0.0502 | 4.80 ± 1.31 | 11.27 ± 3.18 | 743.6 ± 9.2 |
| coh_fifo_80 | 7 | 2.3319 ± 0.1231 | 2.6895 ± 0.2806 | 1.9743 ± 0.0513 | 6.14 ± 2.27 | 7.59 ± 1.95 | 732.8 ± 19.8 |
| coh_gate_off | 7 | 2.0426 ± 0.0830 | 2.9969 ± 0.1749 | 1.0883 ± 0.0395 | 7.25 ± 3.98 | 38.95 ± 2.94 | 420.2 ± 12.8 |
| coh_gate_99 | 7 | 2.1241 ± 0.0425 | 3.0453 ± 0.1129 | 1.2029 ± 0.1263 | 6.03 ± 2.91 | 32.37 ± 5.52 | 755.7 ± 12.6 |
| coh_learned_off | 7 | 2.0158 ± 0.0888 | 2.8980 ± 0.1831 | 1.1335 ± 0.0478 | 7.37 ± 1.42 | 36.94 ± 2.29 | 421.9 ± 5.0 |
| coh_learned_99 | 7 | 2.1150 ± 0.0674 | 2.9851 ± 0.1143 | 1.2450 ± 0.1342 | 6.03 ± 1.79 | 32.37 ± 7.06 | 747.9 ± 23.2 |
| coh_fifo_995 | 1 | 2.1555 ± NA | 3.1738 ± NA | 1.1372 ± NA | 3.91 ± NA | 35.94 ± NA | 732.3 ± NA |
| coh_fifo_999 | 1 | 2.1338 ± NA | 3.1483 ± NA | 1.1193 ± NA | 3.12 ± NA | 35.94 ± NA | 773.9 ± NA |

Seed-paired percentile bootstrap:20,000 resamples, RNG91731. Generation differences are percentage points. Fixed data and evaluation cases; no multiplicity correction. Selected-threshold intervals do not account for threshold selection. Difference-in-differences are supplemental interaction diagnostics.

| contrast | metric | n | mean difference | paired bootstrap 95% CI |
|---|---|---:|---:|---|
| coh_fifo_99 − coh_fifo_off | content CE | 7 | +0.0321 | [+0.0009, +0.0692] |
| coh_fifo_99 − coh_fifo_off | Depo CE | 7 | +0.0155 | [-0.0256, +0.0578] |
| coh_fifo_99 − coh_fifo_off | Brevo CE | 7 | +0.0488 | [+0.0195, +0.0948] |
| coh_fifo_99 − coh_fifo_off | Depo gen % | 7 | +0.22 | [-1.34, +1.90] |
| coh_fifo_99 − coh_fifo_off | Brevo valid % | 7 | -2.46 | [-6.14, +0.78] |
| coh_fifo_95 − coh_fifo_off | content CE | 7 | +0.3435 | [+0.3209, +0.3654] |
| coh_fifo_95 − coh_fifo_off | Depo CE | 7 | +0.0243 | [-0.0291, +0.0846] |
| coh_fifo_95 − coh_fifo_off | Brevo CE | 7 | +0.6627 | [+0.6317, +0.6950] |
| coh_fifo_95 − coh_fifo_off | Depo gen % | 7 | -0.56 | [-2.12, +1.45] |
| coh_fifo_95 − coh_fifo_off | Brevo valid % | 7 | -24.00 | [-25.78, -22.32] |
| coh_fifo_80 − coh_fifo_off | content CE | 7 | +0.2427 | [+0.1822, +0.3001] |
| coh_fifo_80 − coh_fifo_off | Depo CE | 7 | -0.3531 | [-0.5098, -0.2140] |
| coh_fifo_80 − coh_fifo_off | Brevo CE | 7 | +0.8385 | [+0.7887, +0.8892] |
| coh_fifo_80 − coh_fifo_off | Depo gen % | 7 | +0.78 | [-1.67, +3.01] |
| coh_fifo_80 − coh_fifo_off | Brevo valid % | 7 | -27.68 | [-30.47, -24.44] |
| coh_gate_99 − coh_gate_off | content CE | 7 | +0.0815 | [+0.0348, +0.1323] |
| coh_gate_99 − coh_gate_off | Depo CE | 7 | +0.0484 | [-0.0182, +0.1408] |
| coh_gate_99 − coh_gate_off | Brevo CE | 7 | +0.1146 | [+0.0468, +0.1922] |
| coh_gate_99 − coh_gate_off | Depo gen % | 7 | -1.23 | [-3.46, +0.89] |
| coh_gate_99 − coh_gate_off | Brevo valid % | 7 | -6.58 | [-10.83, -3.68] |
| coh_learned_99 − coh_learned_off | content CE | 7 | +0.0993 | [+0.0336, +0.1664] |
| coh_learned_99 − coh_learned_off | Depo CE | 7 | +0.0871 | [-0.0245, +0.1924] |
| coh_learned_99 − coh_learned_off | Brevo CE | 7 | +0.1115 | [+0.0401, +0.1774] |
| coh_learned_99 − coh_learned_off | Depo gen % | 7 | -1.34 | [-1.90, -0.89] |
| coh_learned_99 − coh_learned_off | Brevo valid % | 7 | -4.58 | [-8.71, -0.22] |
| coh_gate_99 − coh_fifo_99 | content CE | 7 | +0.0028 | [-0.0269, +0.0355] |
| coh_gate_99 − coh_fifo_99 | Depo CE | 7 | -0.0127 | [-0.0879, +0.0561] |
| coh_gate_99 − coh_fifo_99 | Brevo CE | 7 | +0.0184 | [-0.0399, +0.0896] |
| coh_gate_99 − coh_fifo_99 | Depo gen % | 7 | +0.45 | [-1.34, +2.23] |
| coh_gate_99 − coh_fifo_99 | Brevo valid % | 7 | -0.45 | [-2.23, +1.45] |
| coh_learned_99 − coh_fifo_99 | content CE | 7 | -0.0062 | [-0.0500, +0.0373] |
| coh_learned_99 − coh_fifo_99 | Depo CE | 7 | -0.0729 | [-0.1315, -0.0185] |
| coh_learned_99 − coh_fifo_99 | Brevo CE | 7 | +0.0604 | [-0.0172, +0.1397] |
| coh_learned_99 − coh_fifo_99 | Depo gen % | 7 | +0.45 | [-0.67, +1.79] |
| coh_learned_99 − coh_fifo_99 | Brevo valid % | 7 | -0.45 | [-5.13, +3.79] |
| coh_fifo_995 − coh_fifo_off | content CE | 1 | -0.0002 | [-0.0002, -0.0002] |
| coh_fifo_995 − coh_fifo_off | Depo CE | 1 | -0.0114 | [-0.0114, -0.0114] |
| coh_fifo_995 − coh_fifo_off | Brevo CE | 1 | +0.0109 | [+0.0109, +0.0109] |
| coh_fifo_995 − coh_fifo_off | Depo gen % | 1 | +0.78 | [+0.78, +0.78] |
| coh_fifo_995 − coh_fifo_off | Brevo valid % | 1 | -1.56 | [-1.56, -1.56] |
| coh_fifo_999 − coh_fifo_off | content CE | 1 | -0.0219 | [-0.0219, -0.0219] |
| coh_fifo_999 − coh_fifo_off | Depo CE | 1 | -0.0369 | [-0.0369, -0.0369] |
| coh_fifo_999 − coh_fifo_off | Brevo CE | 1 | -0.0070 | [-0.0070, -0.0070] |
| coh_fifo_999 − coh_fifo_off | Depo gen % | 1 | +0.00 | [+0.00, +0.00] |
| coh_fifo_999 − coh_fifo_off | Brevo valid % | 1 | -1.56 | [-1.56, -1.56] |
| (coh_gate_99 − coh_gate_off) − (coh_fifo_99 − coh_fifo_off) | content CE | 7 | +0.0494 | [+0.0151, +0.0877] |
| (coh_gate_99 − coh_gate_off) − (coh_fifo_99 − coh_fifo_off) | Depo CE | 7 | +0.0330 | [-0.0372, +0.1052] |
| (coh_gate_99 − coh_gate_off) − (coh_fifo_99 − coh_fifo_off) | Brevo CE | 7 | +0.0658 | [+0.0143, +0.1269] |
| (coh_gate_99 − coh_gate_off) − (coh_fifo_99 − coh_fifo_off) | Depo gen % | 7 | -1.45 | [-3.68, +1.12] |
| (coh_gate_99 − coh_gate_off) − (coh_fifo_99 − coh_fifo_off) | Brevo valid % | 7 | -4.13 | [-6.92, -1.23] |
| (coh_learned_99 − coh_learned_off) − (coh_fifo_99 − coh_fifo_off) | content CE | 7 | +0.0671 | [+0.0046, +0.1396] |
| (coh_learned_99 − coh_learned_off) − (coh_fifo_99 − coh_fifo_off) | Depo CE | 7 | +0.0716 | [-0.0554, +0.1848] |
| (coh_learned_99 − coh_learned_off) − (coh_fifo_99 − coh_fifo_off) | Brevo CE | 7 | +0.0627 | [-0.0077, +0.1343] |
| (coh_learned_99 − coh_learned_off) − (coh_fifo_99 − coh_fifo_off) | Depo gen % | 7 | -1.56 | [-3.12, -0.11] |
| (coh_learned_99 − coh_learned_off) − (coh_fifo_99 − coh_fifo_off) | Brevo valid % | 7 | -2.12 | [-8.15, +3.12] |

Merge fractions: nonpadding token-head weighted within each run; mean ± sample SD across seeds. Only optimizer steps1–224 and2008–2231, no evaluation forwards.

| arm | layer | n | first 224 steps merge % | last 224 steps merge % |
|---|---:|---:|---:|---:|
| coh_fifo_off | 0 | 7 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| coh_fifo_off | 1 | 7 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| coh_fifo_off | 2 | 7 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| coh_fifo_off | 3 | 7 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| coh_fifo_99 | 0 | 7 | 71.63 ± 0.00 | 71.58 ± 0.02 |
| coh_fifo_99 | 1 | 7 | 28.08 ± 1.51 | 36.69 ± 2.66 |
| coh_fifo_99 | 2 | 7 | 36.96 ± 2.63 | 20.79 ± 8.04 |
| coh_fifo_99 | 3 | 7 | 52.69 ± 3.03 | 19.89 ± 4.85 |
| coh_fifo_95 | 0 | 7 | 71.63 ± 0.00 | 72.83 ± 0.51 |
| coh_fifo_95 | 1 | 7 | 51.43 ± 1.82 | 50.46 ± 4.04 |
| coh_fifo_95 | 2 | 7 | 62.32 ± 2.29 | 54.34 ± 1.56 |
| coh_fifo_95 | 3 | 7 | 71.60 ± 1.38 | 59.41 ± 5.97 |
| coh_fifo_80 | 0 | 7 | 71.63 ± 0.00 | 80.76 ± 3.88 |
| coh_fifo_80 | 1 | 7 | 71.65 ± 1.18 | 70.42 ± 4.61 |
| coh_fifo_80 | 2 | 7 | 80.58 ± 0.81 | 78.02 ± 3.84 |
| coh_fifo_80 | 3 | 7 | 83.23 ± 0.90 | 82.05 ± 2.90 |
| coh_gate_off | 0 | 7 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| coh_gate_off | 1 | 7 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| coh_gate_off | 2 | 7 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| coh_gate_off | 3 | 7 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| coh_gate_99 | 0 | 7 | 71.63 ± 0.00 | 71.57 ± 0.00 |
| coh_gate_99 | 1 | 7 | 28.11 ± 1.34 | 36.22 ± 4.36 |
| coh_gate_99 | 2 | 7 | 37.12 ± 2.81 | 26.73 ± 14.67 |
| coh_gate_99 | 3 | 7 | 53.03 ± 3.07 | 33.83 ± 13.66 |
| coh_learned_off | 0 | 7 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| coh_learned_off | 1 | 7 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| coh_learned_off | 2 | 7 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| coh_learned_off | 3 | 7 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| coh_learned_99 | 0 | 7 | 67.78 ± 0.48 | 65.21 ± 2.27 |
| coh_learned_99 | 1 | 7 | 27.28 ± 1.78 | 31.13 ± 4.88 |
| coh_learned_99 | 2 | 7 | 36.46 ± 2.51 | 27.53 ± 12.09 |
| coh_learned_99 | 3 | 7 | 52.50 ± 3.23 | 33.04 ± 12.67 |
| coh_fifo_995 | 0 | 1 | 71.63 ± NA | 71.57 ± NA |
| coh_fifo_995 | 1 | 1 | 21.49 ± NA | 34.05 ± NA |
| coh_fifo_995 | 2 | 1 | 30.44 ± NA | 13.35 ± NA |
| coh_fifo_995 | 3 | 1 | 48.43 ± NA | 13.15 ± NA |
| coh_fifo_999 | 0 | 1 | 71.63 ± NA | 71.57 ± NA |
| coh_fifo_999 | 1 | 1 | 6.18 ± NA | 18.12 ± NA |
| coh_fifo_999 | 2 | 1 | 10.77 ± NA | 4.90 ± NA |
| coh_fifo_999 | 3 | 1 | 24.10 ± NA | 3.94 ± NA |
