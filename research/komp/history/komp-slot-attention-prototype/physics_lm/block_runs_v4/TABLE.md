Values are mean ± sample standard deviation across seeds. Generation is percent; paired generation differences are percentage points. CE: lower is better. Generation: higher is better.

| arm | n | content CE | Depo CE | Brevo CE | Depo gen % | Brevo valid % | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| block_c64 | 7 | 2.4135 ± 0.0906 | 3.0830 ± 0.1099 | 1.7440 ± 0.0940 | 5.47 ± 1.28 | 12.61 ± 3.01 | 60.9 ± 0.2 |
| block_slot_c64 | 7 | 2.0917 ± 0.0793 | 3.0440 ± 0.1505 | 1.1393 ± 0.0252 | 5.13 ± 1.34 | 36.50 ± 2.54 | 62.0 ± 0.1 |
| block_slot_gate_c64 | 7 | 2.0519 ± 0.0774 | 3.0133 ± 0.1566 | 1.0905 ± 0.0423 | 6.70 ± 2.06 | 40.07 ± 3.43 | 66.5 ± 0.3 |
| block_slot_learn_c64 | 7 | 2.0067 ± 0.0772 | 2.8728 ± 0.1520 | 1.1406 ± 0.0482 | 7.03 ± 1.19 | 37.28 ± 3.49 | 67.5 ± 0.3 |
| block_slot_learn_c32 | 7 | 2.0472 ± 0.0918 | 2.8881 ± 0.1312 | 1.2063 ± 0.1296 | 6.70 ± 2.66 | 35.83 ± 6.48 | 67.4 ± 0.2 |
| gdn | 7 | 2.3759 ± 0.1088 | 3.0066 ± 0.1257 | 1.7451 ± 0.1584 | 3.35 ± 1.33 | 11.16 ± 3.49 | 71.8 ± 1.2 |
| gsa | 7 | 2.7163 ± 0.0172 | 3.2943 ± 0.0224 | 2.1382 ± 0.0160 | 2.90 ± 1.08 | 3.01 ± 1.31 | 77.2 ± 2.9 |
| gsa_slot | 7 | 2.2469 ± 0.0467 | 3.2131 ± 0.0407 | 1.2808 ± 0.0731 | 4.13 ± 1.33 | 33.93 ± 2.34 | 67.7 ± 0.3 |
| kapa_scalar_p2_c16 | 7 | 2.1423 ± 0.0945 | 2.3123 ± 0.1574 | 1.9724 ± 0.0553 | 9.71 ± 1.85 | 7.14 ± 2.45 | 60.8 ± 0.7 |
| kapa_slot_p2_c16 | 7 | 2.2365 ± 0.1849 | 2.5006 ± 0.3734 | 1.9724 ± 0.0344 | 6.25 ± 2.26 | 7.37 ± 2.95 | 62.2 ± 0.5 |

Paired A − B; percentile bootstrap resamples the matched seed pairs, 20,000 draws, fixed seed 91731. Intervals describe initialization-seed variability conditional on this fixed dataset and evaluation subset; no multiple-comparison correction. Scalar-KAPA versus Block/GDN contrasts are explicitly exploratory additions after six KAPA seeds were inspected; they were not in the original contrast plan.

| A − B | metric | paired n | mean difference | bootstrap 95% CI |
|---|---|---:|---:|---|
| block_slot_learn_c64 − block_slot_c64 | content CE | 7 | -0.0850 | [-0.1337, -0.0324] |
| block_slot_learn_c64 − block_slot_c64 | Depo CE | 7 | -0.1713 | [-0.2717, -0.0685] |
| block_slot_learn_c64 − block_slot_c64 | Brevo CE | 7 | +0.0013 | [-0.0259, +0.0285] |
| block_slot_learn_c64 − block_slot_c64 | Depo gen % | 7 | +1.90 | [+1.12, +2.68] |
| block_slot_learn_c64 − block_slot_c64 | Brevo valid % | 7 | +0.78 | [-0.89, +1.90] |
| block_slot_gate_c64 − block_slot_c64 | content CE | 7 | -0.0398 | [-0.0551, -0.0263] |
| block_slot_gate_c64 − block_slot_c64 | Depo CE | 7 | -0.0308 | [-0.0626, -0.0003] |
| block_slot_gate_c64 − block_slot_c64 | Brevo CE | 7 | -0.0488 | [-0.0648, -0.0335] |
| block_slot_gate_c64 − block_slot_c64 | Depo gen % | 7 | +1.56 | [+0.33, +2.68] |
| block_slot_gate_c64 − block_slot_c64 | Brevo valid % | 7 | +3.57 | [+1.56, +5.13] |
| block_slot_gate_c64 − block_slot_learn_c64 | content CE | 7 | +0.0452 | [-0.0118, +0.1003] |
| block_slot_gate_c64 − block_slot_learn_c64 | Depo CE | 7 | +0.1405 | [+0.0259, +0.2471] |
| block_slot_gate_c64 − block_slot_learn_c64 | Brevo CE | 7 | -0.0501 | [-0.0770, -0.0297] |
| block_slot_gate_c64 − block_slot_learn_c64 | Depo gen % | 7 | -0.33 | [-1.56, +0.78] |
| block_slot_gate_c64 − block_slot_learn_c64 | Brevo valid % | 7 | +2.79 | [+2.12, +3.57] |
| block_slot_c64 − gsa_slot | content CE | 7 | -0.1553 | [-0.1894, -0.1308] |
| block_slot_c64 − gsa_slot | Depo CE | 7 | -0.1690 | [-0.2593, -0.1002] |
| block_slot_c64 − gsa_slot | Brevo CE | 7 | -0.1415 | [-0.1807, -0.1093] |
| block_slot_c64 − gsa_slot | Depo gen % | 7 | +1.00 | [-0.22, +2.23] |
| block_slot_c64 − gsa_slot | Brevo valid % | 7 | +2.57 | [+0.67, +4.69] |
| block_slot_c64 − gdn | content CE | 7 | -0.2842 | [-0.3647, -0.1983] |
| block_slot_c64 − gdn | Depo CE | 7 | +0.0375 | [-0.1186, +0.1786] |
| block_slot_c64 − gdn | Brevo CE | 7 | -0.6059 | [-0.7089, -0.5070] |
| block_slot_c64 − gdn | Depo gen % | 7 | +1.79 | [+0.67, +2.79] |
| block_slot_c64 − gdn | Brevo valid % | 7 | +25.33 | [+23.10, +27.34] |
| block_slot_c64 − block_c64 | content CE | 7 | -0.3219 | [-0.3787, -0.2624] |
| block_slot_c64 − block_c64 | Depo CE | 7 | -0.0390 | [-0.1576, +0.0577] |
| block_slot_c64 − block_c64 | Brevo CE | 7 | -0.6047 | [-0.6612, -0.5607] |
| block_slot_c64 − block_c64 | Depo gen % | 7 | -0.33 | [-1.34, +0.67] |
| block_slot_c64 − block_c64 | Brevo valid % | 7 | +23.88 | [+21.76, +26.23] |
| block_slot_learn_c32 − block_slot_learn_c64 | content CE | 7 | +0.0405 | [+0.0131, +0.0730] |
| block_slot_learn_c32 − block_slot_learn_c64 | Depo CE | 7 | +0.0153 | [-0.0450, +0.0877] |
| block_slot_learn_c32 − block_slot_learn_c64 | Brevo CE | 7 | +0.0657 | [+0.0141, +0.1510] |
| block_slot_learn_c32 − block_slot_learn_c64 | Depo gen % | 7 | -0.33 | [-2.12, +1.23] |
| block_slot_learn_c32 − block_slot_learn_c64 | Brevo valid % | 7 | -1.45 | [-4.80, +0.89] |
| kapa_slot_p2_c16 − kapa_scalar_p2_c16 | content CE | 7 | +0.0941 | [-0.0213, +0.2150] |
| kapa_slot_p2_c16 − kapa_scalar_p2_c16 | Depo CE | 7 | +0.1883 | [-0.0544, +0.4518] |
| kapa_slot_p2_c16 − kapa_scalar_p2_c16 | Brevo CE | 7 | -0.0001 | [-0.0287, +0.0321] |
| kapa_slot_p2_c16 − kapa_scalar_p2_c16 | Depo gen % | 7 | -3.46 | [-5.36, -1.56] |
| kapa_slot_p2_c16 − kapa_scalar_p2_c16 | Brevo valid % | 7 | +0.22 | [-2.12, +2.57] |
| kapa_slot_p2_c16 − block_slot_c64 | content CE | 7 | +0.1448 | [+0.0444, +0.2616] |
| kapa_slot_p2_c16 − block_slot_c64 | Depo CE | 7 | -0.5435 | [-0.7662, -0.2945] |
| kapa_slot_p2_c16 − block_slot_c64 | Brevo CE | 7 | +0.8331 | [+0.8037, +0.8583] |
| kapa_slot_p2_c16 − block_slot_c64 | Depo gen % | 7 | +1.12 | [-0.45, +2.79] |
| kapa_slot_p2_c16 − block_slot_c64 | Brevo valid % | 7 | -29.13 | [-31.03, -26.90] |
| kapa_slot_p2_c16 − gdn | content CE | 7 | -0.1394 | [-0.2382, -0.0422] |
| kapa_slot_p2_c16 − gdn | Depo CE | 7 | -0.5060 | [-0.7422, -0.2734] |
| kapa_slot_p2_c16 − gdn | Brevo CE | 7 | +0.2272 | [+0.1309, +0.3208] |
| kapa_slot_p2_c16 − gdn | Depo gen % | 7 | +2.90 | [+1.67, +4.13] |
| kapa_slot_p2_c16 − gdn | Brevo valid % | 7 | -3.79 | [-5.58, -2.23] |
| kapa_scalar_p2_c16 − block_slot_c64 (exploratory) | content CE | 7 | +0.0507 | [-0.0013, +0.1011] |
| kapa_scalar_p2_c16 − block_slot_c64 (exploratory) | Depo CE | 7 | -0.7318 | [-0.8340, -0.6319] |
| kapa_scalar_p2_c16 − block_slot_c64 (exploratory) | Brevo CE | 7 | +0.8331 | [+0.7989, +0.8644] |
| kapa_scalar_p2_c16 − block_slot_c64 (exploratory) | Depo gen % | 7 | +4.58 | [+3.35, +5.69] |
| kapa_scalar_p2_c16 − block_slot_c64 (exploratory) | Brevo valid % | 7 | -29.35 | [-31.47, -27.23] |
| kapa_scalar_p2_c16 − gdn (exploratory) | content CE | 7 | -0.2335 | [-0.3081, -0.1223] |
| kapa_scalar_p2_c16 − gdn (exploratory) | Depo CE | 7 | -0.6943 | [-0.8445, -0.5162] |
| kapa_scalar_p2_c16 − gdn (exploratory) | Brevo CE | 7 | +0.2273 | [+0.1318, +0.3187] |
| kapa_scalar_p2_c16 − gdn (exploratory) | Depo gen % | 7 | +6.36 | [+5.02, +7.70] |
| kapa_scalar_p2_c16 − gdn (exploratory) | Brevo valid % | 7 | -4.02 | [-5.69, -2.34] |
