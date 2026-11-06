Values are mean ± sample standard deviation across seeds. Generation is percent; paired generation differences are percentage points. CE: lower is better. Generation: higher is better.

| arm | n | content CE | Depo CE | Brevo CE | Depo gen % | Brevo valid % | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| fifo_proj_c64 | 7 | 2.0892 ± 0.0780 | 3.0426 ± 0.1526 | 1.1357 ± 0.0265 | 5.36 ± 2.57 | 35.27 ± 2.83 | 412.9 ± 7.5 |
| fifo_noproj_c64 | 7 | 2.3169 ± 0.0129 | 3.3463 ± 0.0310 | 1.2875 ± 0.0311 | 4.35 ± 1.96 | 36.72 ± 2.30 | 306.8 ± 1.5 |
| learned_proj_c64 | 7 | 2.0158 ± 0.0888 | 2.8980 ± 0.1831 | 1.1335 ± 0.0478 | 7.37 ± 1.42 | 36.94 ± 2.29 | 425.9 ± 12.2 |
| learned_noproj_c64 | 7 | 2.3313 ± 0.0380 | 3.3438 ± 0.0599 | 1.3187 ± 0.0328 | 4.35 ± 1.91 | 36.61 ± 2.76 | 316.4 ± 8.7 |
| swa_w32 | 7 | 2.6647 ± 0.0722 | 3.2322 ± 0.1651 | 2.0973 ± 0.0407 | 5.69 ± 3.34 | 5.92 ± 1.09 | 57.0 ± 0.2 |
| swa_w96 | 7 | 2.4239 ± 0.2165 | 2.8260 ± 0.4508 | 2.0217 ± 0.0307 | 6.47 ± 3.63 | 7.03 ± 2.07 | 56.7 ± 0.5 |
| swa_w96_nope | 7 | 2.6444 ± 0.0269 | 3.1715 ± 0.0351 | 2.1173 ± 0.0571 | 3.12 ± 1.50 | 4.69 ± 0.90 | 45.8 ± 1.2 |
| full_attn | 7 | 2.4168 ± 0.2281 | 2.8013 ± 0.4649 | 2.0322 ± 0.0250 | 6.36 ± 4.05 | 6.70 ± 1.68 | 56.0 ± 0.8 |
| fifo_proj_c128 | 7 | 1.8451 ± 0.1603 | 2.6897 ± 0.3191 | 1.0005 ± 0.0284 | 7.03 ± 2.59 | 38.62 ± 3.02 | 281.1 ± 2.8 |
| fifo_noproj_c128 | 7 | 1.8904 ± 0.1534 | 2.7541 ± 0.3068 | 1.0267 ± 0.0224 | 7.03 ± 1.63 | 37.39 ± 2.76 | 217.5 ± 3.8 |

Paired A − B; percentile bootstrap resamples the matched seed pairs, 20,000 draws, fixed seed 91731. Intervals describe initialization-seed variability conditional on this fixed dataset and evaluation subset; no multiple-comparison correction. Projection contrasts and SWA comparisons were specified before training; C128 is the prespecified optional extension.

| A − B | metric | paired n | mean difference | bootstrap 95% CI |
|---|---|---:|---:|---|
| fifo_noproj_c64 − fifo_proj_c64 | content CE | 7 | +0.2277 | [+0.1876, +0.2791] |
| fifo_noproj_c64 − fifo_proj_c64 | Depo CE | 7 | +0.3037 | [+0.2179, +0.4144] |
| fifo_noproj_c64 − fifo_proj_c64 | Brevo CE | 7 | +0.1518 | [+0.1375, +0.1667] |
| fifo_noproj_c64 − fifo_proj_c64 | Depo gen % | 7 | -1.00 | [-2.34, +0.45] |
| fifo_noproj_c64 − fifo_proj_c64 | Brevo valid % | 7 | +1.45 | [+0.45, +2.57] |
| learned_noproj_c64 − learned_proj_c64 | content CE | 7 | +0.3155 | [+0.2570, +0.3831] |
| learned_noproj_c64 − learned_proj_c64 | Depo CE | 7 | +0.4458 | [+0.3331, +0.5758] |
| learned_noproj_c64 − learned_proj_c64 | Brevo CE | 7 | +0.1852 | [+0.1406, +0.2201] |
| learned_noproj_c64 − learned_proj_c64 | Depo gen % | 7 | -3.01 | [-5.02, -0.78] |
| learned_noproj_c64 − learned_proj_c64 | Brevo valid % | 7 | -0.33 | [-3.12, +2.68] |
| fifo_proj_c64 − swa_w32 | content CE | 7 | -0.5756 | [-0.6514, -0.4951] |
| fifo_proj_c64 − swa_w32 | Depo CE | 7 | -0.1896 | [-0.3584, -0.0132] |
| fifo_proj_c64 − swa_w32 | Brevo CE | 7 | -0.9616 | [-0.9984, -0.9341] |
| fifo_proj_c64 − swa_w32 | Depo gen % | 7 | -0.33 | [-2.46, +1.90] |
| fifo_proj_c64 − swa_w32 | Brevo valid % | 7 | +29.35 | [+27.68, +30.80] |
| fifo_proj_c64 − swa_w96 | content CE | 7 | -0.3347 | [-0.4800, -0.1643] |
| fifo_proj_c64 − swa_w96 | Depo CE | 7 | +0.2165 | [-0.0768, +0.5624] |
| fifo_proj_c64 − swa_w96 | Brevo CE | 7 | -0.8860 | [-0.9102, -0.8606] |
| fifo_proj_c64 − swa_w96 | Depo gen % | 7 | -1.12 | [-4.69, +2.01] |
| fifo_proj_c64 − swa_w96 | Brevo valid % | 7 | +28.24 | [+26.23, +30.13] |
| swa_w96 − swa_w96_nope | content CE | 7 | -0.2205 | [-0.3648, -0.0980] |
| swa_w96 − swa_w96_nope | Depo CE | 7 | -0.3454 | [-0.6498, -0.0995] |
| swa_w96 − swa_w96_nope | Brevo CE | 7 | -0.0956 | [-0.1310, -0.0589] |
| swa_w96 − swa_w96_nope | Depo gen % | 7 | +3.35 | [+1.79, +5.13] |
| swa_w96 − swa_w96_nope | Brevo valid % | 7 | +2.34 | [+1.23, +3.79] |
| fifo_noproj_c128 − fifo_proj_c128 | content CE | 7 | +0.0452 | [+0.0256, +0.0653] |
| fifo_noproj_c128 − fifo_proj_c128 | Depo CE | 7 | +0.0643 | [+0.0227, +0.1066] |
| fifo_noproj_c128 − fifo_proj_c128 | Brevo CE | 7 | +0.0261 | [+0.0142, +0.0367] |
| fifo_noproj_c128 − fifo_proj_c128 | Depo gen % | 7 | +0.00 | [-1.34, +1.23] |
| fifo_noproj_c128 − fifo_proj_c128 | Brevo valid % | 7 | -1.23 | [-2.01, -0.45] |

| arm | parameters | persistent elements / head |
|---|---:|---:|
| fifo_proj_c64 | 8964744 | 4096 |
| fifo_noproj_c64 | 8964744 | 4096 |
| learned_proj_c64 | 8966032 | 4096 |
| learned_noproj_c64 | 8966032 | 4096 |
| swa_w32 | 8914304 | 4096 |
| swa_w96 | 8914304 | 12288 |
| swa_w96_nope | 8914304 | 12288 |
| full_attn | 8914304 | 24576 |
| fifo_proj_c128 | 8964744 | 4096 |
| fifo_noproj_c128 | 8964744 | 4096 |

Per-run synchronized wall time. Training excludes measured evaluation/generation, includes loop/logging overhead. Different implementations: these are not kernel speed comparisons.

| arm | seed | train loop + bookkeeping s | evaluation s | generation s | total train + eval s |
|---|---:|---:|---:|---:|---:|
| fifo_proj_c64 | 731 | 366.42 | 36.69 | 5.08 | 408.20 |
| fifo_proj_c64 | 732 | 372.60 | 37.30 | 4.92 | 414.82 |
| fifo_proj_c64 | 733 | 364.83 | 36.70 | 5.07 | 406.60 |
| fifo_proj_c64 | 734 | 379.42 | 38.14 | 5.60 | 423.17 |
| fifo_proj_c64 | 735 | 366.51 | 37.03 | 5.33 | 408.87 |
| fifo_proj_c64 | 736 | 379.16 | 38.28 | 5.36 | 422.80 |
| fifo_proj_c64 | 737 | 364.68 | 36.53 | 4.64 | 405.86 |
| fifo_noproj_c64 | 731 | 274.80 | 28.29 | 3.65 | 306.74 |
| fifo_noproj_c64 | 732 | 272.94 | 28.04 | 3.66 | 304.63 |
| fifo_noproj_c64 | 733 | 275.12 | 28.40 | 3.82 | 307.33 |
| fifo_noproj_c64 | 734 | 275.27 | 28.50 | 3.83 | 307.60 |
| fifo_noproj_c64 | 735 | 275.59 | 28.36 | 3.73 | 307.67 |
| fifo_noproj_c64 | 736 | 273.27 | 28.06 | 3.63 | 304.96 |
| fifo_noproj_c64 | 737 | 276.43 | 28.68 | 3.69 | 308.79 |
| learned_proj_c64 | 731 | 369.37 | 36.85 | 6.02 | 412.23 |
| learned_proj_c64 | 732 | 404.08 | 37.60 | 5.49 | 447.17 |
| learned_proj_c64 | 733 | 381.26 | 38.25 | 5.73 | 425.24 |
| learned_proj_c64 | 734 | 392.44 | 37.63 | 5.56 | 435.63 |
| learned_proj_c64 | 735 | 370.98 | 37.27 | 5.48 | 413.73 |
| learned_proj_c64 | 736 | 379.14 | 37.90 | 6.11 | 423.15 |
| learned_proj_c64 | 737 | 380.17 | 37.84 | 5.82 | 423.84 |
| learned_noproj_c64 | 731 | 290.48 | 27.85 | 3.66 | 321.99 |
| learned_noproj_c64 | 732 | 278.02 | 28.26 | 5.15 | 311.43 |
| learned_noproj_c64 | 733 | 285.35 | 28.20 | 4.13 | 317.68 |
| learned_noproj_c64 | 734 | 299.07 | 29.20 | 4.21 | 332.48 |
| learned_noproj_c64 | 735 | 280.73 | 29.76 | 4.31 | 314.80 |
| learned_noproj_c64 | 736 | 276.74 | 28.18 | 3.93 | 308.85 |
| learned_noproj_c64 | 737 | 273.82 | 29.18 | 4.36 | 307.37 |
| swa_w32 | 731 | 51.53 | 4.83 | 0.45 | 56.80 |
| swa_w32 | 732 | 51.60 | 4.78 | 0.48 | 56.86 |
| swa_w32 | 733 | 52.01 | 4.82 | 0.50 | 57.33 |
| swa_w32 | 734 | 51.57 | 4.81 | 0.47 | 56.84 |
| swa_w32 | 735 | 51.82 | 4.82 | 0.46 | 57.11 |
| swa_w32 | 736 | 51.57 | 4.88 | 0.49 | 56.94 |
| swa_w32 | 737 | 51.62 | 4.88 | 0.46 | 56.97 |
| swa_w96 | 731 | 51.67 | 4.62 | 0.47 | 56.76 |
| swa_w96 | 732 | 51.54 | 4.76 | 0.52 | 56.82 |
| swa_w96 | 733 | 50.61 | 4.55 | 0.48 | 55.64 |
| swa_w96 | 734 | 51.50 | 4.81 | 0.58 | 56.89 |
| swa_w96 | 735 | 51.76 | 4.66 | 0.48 | 56.90 |
| swa_w96 | 736 | 51.66 | 4.88 | 0.58 | 57.11 |
| swa_w96 | 737 | 50.92 | 5.16 | 0.52 | 56.59 |
| swa_w96_nope | 731 | 41.16 | 3.48 | 0.36 | 45.00 |
| swa_w96_nope | 732 | 42.85 | 3.58 | 0.35 | 46.78 |
| swa_w96_nope | 733 | 42.79 | 3.52 | 0.35 | 46.67 |
| swa_w96_nope | 734 | 42.62 | 3.59 | 0.42 | 46.63 |
| swa_w96_nope | 735 | 40.32 | 3.38 | 0.41 | 44.12 |
| swa_w96_nope | 736 | 42.69 | 3.62 | 0.33 | 46.64 |
| swa_w96_nope | 737 | 40.64 | 3.67 | 0.33 | 44.63 |
| full_attn | 731 | 49.39 | 4.67 | 0.58 | 54.64 |
| full_attn | 732 | 51.34 | 4.81 | 0.54 | 56.69 |
| full_attn | 733 | 49.99 | 4.60 | 0.49 | 55.09 |
| full_attn | 734 | 51.49 | 4.83 | 0.51 | 56.83 |
| full_attn | 735 | 50.79 | 4.69 | 0.54 | 56.02 |
| full_attn | 736 | 51.35 | 4.77 | 0.55 | 56.67 |
| full_attn | 737 | 50.38 | 4.88 | 0.55 | 55.81 |
| fifo_proj_c128 | 731 | 245.81 | 26.13 | 3.95 | 275.88 |
| fifo_proj_c128 | 732 | 253.04 | 26.95 | 4.29 | 284.29 |
| fifo_proj_c128 | 733 | 249.37 | 26.49 | 3.63 | 279.50 |
| fifo_proj_c128 | 734 | 250.08 | 26.38 | 4.34 | 280.80 |
| fifo_proj_c128 | 735 | 250.97 | 26.50 | 4.44 | 281.91 |
| fifo_proj_c128 | 736 | 252.66 | 26.77 | 3.84 | 283.27 |
| fifo_proj_c128 | 737 | 251.06 | 26.99 | 3.97 | 282.02 |
| fifo_noproj_c128 | 731 | 192.98 | 20.59 | 2.95 | 216.52 |
| fifo_noproj_c128 | 732 | 187.72 | 19.74 | 2.46 | 209.92 |
| fifo_noproj_c128 | 733 | 195.10 | 20.22 | 2.54 | 217.86 |
| fifo_noproj_c128 | 734 | 197.99 | 20.84 | 3.03 | 221.86 |
| fifo_noproj_c128 | 735 | 196.44 | 20.78 | 2.93 | 220.15 |
| fifo_noproj_c128 | 736 | 195.03 | 20.72 | 2.89 | 218.64 |
| fifo_noproj_c128 | 737 | 194.19 | 20.49 | 2.78 | 217.46 |
