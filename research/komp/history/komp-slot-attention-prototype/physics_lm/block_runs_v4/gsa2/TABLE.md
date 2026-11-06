Mean ± sample SD across seeds731–737. Percent rates; paired differences in percentage points. Same data/backbone/training tokens; native GSA2 has a different mixer, at least64 slots, native WD exclusions, and conv unless labelled noconv. The 1e-3 recipe was selected in the earlier validation search, not this comparison. Recurrent elements count mixer memory only, excluding convolution caches; KAPA centers are fixed model parameters. GDN slots is the harness M setting, not a slot dictionary. Seconds include training, validation and final generation, excluding process startup and checkpoint reload; this is not a kernel benchmark.

| arm | params | slots | short conv | recurrent elements/head | peak LR |
|---|---:|---:|---|---:|---:|
| block_c64 | 8932752 | 32 | False | 4096 | 0.0005 |
| block_slot_c64 | 8964744 | 32 | False | 4096 | 0.0005 |
| block_slot_gate_c64 | 8966032 | 32 | False | 4096 | 0.0005 |
| block_slot_learn_c64 | 8966032 | 32 | False | 4096 | 0.0005 |
| block_slot_learn_c32 | 8966032 | 32 | False | 4096 | 0.0005 |
| gdn | 8916368 | 32 | False | 4096 | 0.0005 |
| gsa | 8947328 | 32 | False | 4096 | 0.0005 |
| gsa_slot | 8964744 | 32 | False | 4096 | 0.0005 |
| kapa_scalar_p2_c16 | 8932752 | 32 | False | 2048 | 0.0005 |
| kapa_slot_p2_c16 | 8964744 | 32 | False | 2048 | 0.0005 |
| gsa2_native_lr5e4 | 9382800 | 64 | True | 8192 | 0.0005 |
| gsa2_native_lr1e3 | 9382800 | 64 | True | 8192 | 0.001 |
| gsa2_noconv_lr5e4 | 9374608 | 64 | False | 8192 | 0.0005 |

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
| gsa2_native_lr5e4 | 7 | 2.6069 ± 0.1486 | 3.2139 ± 0.3478 | 1.9999 ± 0.1143 | 5.58 ± 1.83 | 11.94 ± 3.46 | 195.5 ± 1.6 |
| gsa2_native_lr1e3 | 7 | 1.9819 ± 0.0963 | 2.2753 ± 0.2214 | 1.6884 ± 0.0729 | 11.94 ± 3.24 | 20.31 ± 2.51 | 194.2 ± 0.7 |
| gsa2_noconv_lr5e4 | 7 | 2.3151 ± 0.1493 | 2.8243 ± 0.3210 | 1.8060 ± 0.0941 | 5.02 ± 2.66 | 13.28 ± 3.19 | 154.2 ± 4.0 |

Seed-paired percentile bootstrap, 20,000 resamples, RNG91731; fixed dataset and128 generation cases/task, no multiplicity correction. All new contrasts are supplemental following the completed Block/KAPA study.

| A − B | metric | n | difference | paired bootstrap 95% CI |
|---|---|---:|---:|---|
| gsa2_native_lr5e4 − block_slot_c64 | content CE | 7 | +0.5153 | [+0.3962, +0.6281] |
| gsa2_native_lr5e4 − block_slot_c64 | Depo CE | 7 | +0.1699 | [-0.1125, +0.3900] |
| gsa2_native_lr5e4 − block_slot_c64 | Brevo CE | 7 | +0.8607 | [+0.7788, +0.9542] |
| gsa2_native_lr5e4 − block_slot_c64 | Depo gen % | 7 | +0.45 | [-1.12, +1.79] |
| gsa2_native_lr5e4 − block_slot_c64 | Brevo valid % | 7 | -24.55 | [-28.01, -21.21] |
| gsa2_native_lr5e4 − block_slot_gate_c64 | content CE | 7 | +0.5550 | [+0.4370, +0.6640] |
| gsa2_native_lr5e4 − block_slot_gate_c64 | Depo CE | 7 | +0.2007 | [-0.0938, +0.4263] |
| gsa2_native_lr5e4 − block_slot_gate_c64 | Brevo CE | 7 | +0.9094 | [+0.8176, +1.0110] |
| gsa2_native_lr5e4 − block_slot_gate_c64 | Depo gen % | 7 | -1.12 | [-2.46, +0.22] |
| gsa2_native_lr5e4 − block_slot_gate_c64 | Brevo valid % | 7 | -28.12 | [-32.03, -24.44] |
| gsa2_native_lr5e4 − block_slot_learn_c64 | content CE | 7 | +0.6002 | [+0.4892, +0.7236] |
| gsa2_native_lr5e4 − block_slot_learn_c64 | Depo CE | 7 | +0.3411 | [+0.0690, +0.5748] |
| gsa2_native_lr5e4 − block_slot_learn_c64 | Brevo CE | 7 | +0.8593 | [+0.7638, +0.9708] |
| gsa2_native_lr5e4 − block_slot_learn_c64 | Depo gen % | 7 | -1.45 | [-3.12, +0.11] |
| gsa2_native_lr5e4 − block_slot_learn_c64 | Brevo valid % | 7 | -25.33 | [-29.46, -21.43] |
| gsa2_native_lr5e4 − kapa_scalar_p2_c16 | content CE | 7 | +0.4646 | [+0.3129, +0.5942] |
| gsa2_native_lr5e4 − kapa_scalar_p2_c16 | Depo CE | 7 | +0.9016 | [+0.5446, +1.1555] |
| gsa2_native_lr5e4 − kapa_scalar_p2_c16 | Brevo CE | 7 | +0.0275 | [-0.0529, +0.1114] |
| gsa2_native_lr5e4 − kapa_scalar_p2_c16 | Depo gen % | 7 | -4.13 | [-6.36, -2.12] |
| gsa2_native_lr5e4 − kapa_scalar_p2_c16 | Brevo valid % | 7 | +4.80 | [+1.90, +7.59] |
| gsa2_native_lr5e4 − gdn | content CE | 7 | +0.2311 | [+0.1566, +0.3166] |
| gsa2_native_lr5e4 − gdn | Depo CE | 7 | +0.2073 | [+0.0053, +0.3800] |
| gsa2_native_lr5e4 − gdn | Brevo CE | 7 | +0.2548 | [+0.0949, +0.4187] |
| gsa2_native_lr5e4 − gdn | Depo gen % | 7 | +2.23 | [+0.78, +3.68] |
| gsa2_native_lr5e4 − gdn | Brevo valid % | 7 | +0.78 | [-3.12, +4.69] |
| gsa2_native_lr1e3 − block_slot_c64 | content CE | 7 | -0.1098 | [-0.1743, -0.0455] |
| gsa2_native_lr1e3 − block_slot_c64 | Depo CE | 7 | -0.7687 | [-0.9131, -0.6079] |
| gsa2_native_lr1e3 − block_slot_c64 | Brevo CE | 7 | +0.5492 | [+0.4890, +0.6024] |
| gsa2_native_lr1e3 − block_slot_c64 | Depo gen % | 7 | +6.81 | [+5.02, +8.59] |
| gsa2_native_lr1e3 − block_slot_c64 | Brevo valid % | 7 | -16.18 | [-18.08, -14.29] |
| gsa2_native_lr1e3 − block_slot_gate_c64 | content CE | 7 | -0.0700 | [-0.1296, -0.0148] |
| gsa2_native_lr1e3 − block_slot_gate_c64 | Depo CE | 7 | -0.7379 | [-0.8847, -0.5767] |
| gsa2_native_lr1e3 − block_slot_gate_c64 | Brevo CE | 7 | +0.5979 | [+0.5334, +0.6534] |
| gsa2_native_lr1e3 − block_slot_gate_c64 | Depo gen % | 7 | +5.25 | [+2.90, +7.59] |
| gsa2_native_lr1e3 − block_slot_gate_c64 | Brevo valid % | 7 | -19.75 | [-22.21, -17.41] |
| gsa2_native_lr1e3 − block_slot_learn_c64 | content CE | 7 | -0.0248 | [-0.1172, +0.0715] |
| gsa2_native_lr1e3 − block_slot_learn_c64 | Depo CE | 7 | -0.5975 | [-0.8039, -0.3736] |
| gsa2_native_lr1e3 − block_slot_learn_c64 | Brevo CE | 7 | +0.5478 | [+0.4779, +0.6098] |
| gsa2_native_lr1e3 − block_slot_learn_c64 | Depo gen % | 7 | +4.91 | [+2.57, +7.37] |
| gsa2_native_lr1e3 − block_slot_learn_c64 | Brevo valid % | 7 | -16.96 | [-19.20, -14.96] |
| gsa2_native_lr1e3 − kapa_scalar_p2_c16 | content CE | 7 | -0.1605 | [-0.2317, -0.0974] |
| gsa2_native_lr1e3 − kapa_scalar_p2_c16 | Depo CE | 7 | -0.0369 | [-0.1934, +0.1089] |
| gsa2_native_lr1e3 − kapa_scalar_p2_c16 | Brevo CE | 7 | -0.2840 | [-0.3310, -0.2506] |
| gsa2_native_lr1e3 − kapa_scalar_p2_c16 | Depo gen % | 7 | +2.23 | [-0.33, +4.80] |
| gsa2_native_lr1e3 − kapa_scalar_p2_c16 | Brevo valid % | 7 | +13.17 | [+11.05, +15.07] |
| gsa2_native_lr1e3 − gdn | content CE | 7 | -0.3940 | [-0.4669, -0.3236] |
| gsa2_native_lr1e3 − gdn | Depo CE | 7 | -0.7312 | [-0.8682, -0.5767] |
| gsa2_native_lr1e3 − gdn | Brevo CE | 7 | -0.0567 | [-0.1724, +0.0459] |
| gsa2_native_lr1e3 − gdn | Depo gen % | 7 | +8.59 | [+6.36, +10.71] |
| gsa2_native_lr1e3 − gdn | Brevo valid % | 7 | +9.15 | [+6.92, +11.38] |
| gsa2_native_lr1e3 − gsa2_native_lr5e4 | content CE | 7 | -0.6250 | [-0.7153, -0.5212] |
| gsa2_native_lr1e3 − gsa2_native_lr5e4 | Depo CE | 7 | -0.9386 | [-1.1618, -0.6717] |
| gsa2_native_lr1e3 − gsa2_native_lr5e4 | Brevo CE | 7 | -0.3115 | [-0.3861, -0.2347] |
| gsa2_native_lr1e3 − gsa2_native_lr5e4 | Depo gen % | 7 | +6.36 | [+3.91, +8.48] |
| gsa2_native_lr1e3 − gsa2_native_lr5e4 | Brevo valid % | 7 | +8.37 | [+5.92, +11.05] |
| gsa2_native_lr5e4 − gsa2_noconv_lr5e4 | content CE | 7 | +0.2918 | [+0.1557, +0.4491] |
| gsa2_native_lr5e4 − gsa2_noconv_lr5e4 | Depo CE | 7 | +0.3897 | [+0.0762, +0.7340] |
| gsa2_native_lr5e4 − gsa2_noconv_lr5e4 | Brevo CE | 7 | +0.1939 | [+0.1147, +0.2699] |
| gsa2_native_lr5e4 − gsa2_noconv_lr5e4 | Depo gen % | 7 | +0.56 | [-1.67, +2.57] |
| gsa2_native_lr5e4 − gsa2_noconv_lr5e4 | Brevo valid % | 7 | -1.34 | [-4.13, +1.67] |
| gsa2_noconv_lr5e4 − block_slot_c64 | content CE | 7 | +0.2235 | [+0.0913, +0.3273] |
| gsa2_noconv_lr5e4 − block_slot_c64 | Depo CE | 7 | -0.2198 | [-0.5112, +0.0160] |
| gsa2_noconv_lr5e4 − block_slot_c64 | Brevo CE | 7 | +0.6667 | [+0.5888, +0.7323] |
| gsa2_noconv_lr5e4 − block_slot_c64 | Depo gen % | 7 | -0.11 | [-2.12, +2.12] |
| gsa2_noconv_lr5e4 − block_slot_c64 | Brevo valid % | 7 | -23.21 | [-26.00, -20.31] |
