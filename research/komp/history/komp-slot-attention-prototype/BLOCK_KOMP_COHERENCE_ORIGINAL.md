# 原始 coherence 实验：56次训练已完成

原始8臂各7seeds已全部完成；本报告不包含正在进行的0.995/0.999细扫。

**结论：在原始网格中，开启coherence没有优于关闭；0.99是开启项里最好的。与gate/learned叠加均损害质量，但Gram条件数确实大幅降低。**

| 配置 | Content CE ↓ | Depo生成 ↑ | Brevo合法率 ↑ |
|---|---:|---:|---:|
| FIFO / off | 2.0892 | 5.36% | 35.27% |
| FIFO / 0.99 | 2.1213 | 5.58% | 32.81% |
| FIFO / 0.95 | 2.4326 | 4.80% | 11.27% |
| FIFO / 0.80 | 2.3319 | 6.14% | 7.59% |
| Gate / off | 2.0426 | 7.25% | 38.95% |
| Gate / 0.99 | 2.1241 | 6.03% | 32.37% |
| Learned / off | 2.0158 | 7.37% | 36.94% |
| Learned / 0.99 | 2.1150 | 6.03% | 32.37% |

- FIFO+0.99相对关闭：CE +0.0321，95%CI[+0.0009,+0.0692]，仅略离零且未做多重比较校正；Brevo -2.46pp，CI[-6.14,+0.78]跨零。没有质量收益证据。
- 0.95与0.80分别损失Brevo约24.00pp与27.68pp，区间均远离零。0.80改善Depo CE，但Depo生成提升不稳定，不能抵消Brevo损失。
- Gate+0.99相对gate关闭：CE +0.0815，CI[+0.0348,+0.1323]；Brevo -6.58pp，CI[-10.83,-3.68]。
- Learned+0.99相对learned关闭：CE +0.0993，CI[+0.0336,+0.1664]；Depo -1.34pp，CI[-1.90,-0.89]；Brevo -4.58pp，CI[-8.71,-0.22]。原始0.99与两种配置都没有表现出正向叠加。
- E3 seed731：FIFO关闭的四层Gram条件数中位数为95207/73596/23059/14369；0.99降到105/1622/2198/2176。0.95/0.80下降更多，却损害质量。条件数改善不能直接证明TF32投影安全。

## 方法与可信度

全部臂均重新跑同一冻结coherence快照的FP32 flex；外围BF16 autocast；确定性算法开启；相同10,001,108token、2231steps、LR5e-4/WD0.1。没有混入旧Triton结果。preflight覆盖未来干预、训练后因果性、未来梯度、统计窗口隔离和E3只读捕获一致性。56次全部重载通过，14,336条生成记录独立审计一致。

最初非确定性scatter_add造成相同输入重复前向的末位差异，被BF16放大；开启确定性模式后因果检查严格为零。修正发生在正式训练之前；未改kernel，未放宽容差。

统计是seed配对20,000次bootstrap，固定数据与生成样本，无多重比较校正。阈值按7seed confirm CE选择，存在重复使用confirm与选参偏差；E3只覆盖每臂seed731。0.95/0.80在用户要求细扫之前已经跑完，作为负结果保留，不再追加。

## 脚本生成完整表

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

## E3完整表

E3:seed731 only, first64 confirm full documents, no padding. Every real block end including final partial blocks; Gram=exp((K Kᵀ−1)/0.5)+1e-4 I. FP32 frozen state measured in FP64. Pairs count unordered off-diagonal pairs once; coefficient Frobenius norm per head. No TF32 experiment.

| arm | layer | block-head observations | merged % | Gram condition median | Gram condition max | cos>0.95 pairs mean / max | coefficient Frobenius median / max |
|---|---:|---:|---:|---:|---:|---:|---:|
| coh_fifo_off | 0 | 216 | 0.00 | 95207.06 | 126066.38 | 27.79 / 56 | 3.3435 / 5.0779 |
| coh_fifo_off | 1 | 216 | 0.00 | 73595.69 | 287284.20 | 75.23 / 471 | 4.3172 / 13.6784 |
| coh_fifo_off | 2 | 216 | 0.00 | 23059.14 | 166051.49 | 59.06 / 199 | 8.2367 / 23.4257 |
| coh_fifo_off | 3 | 216 | 0.00 | 14369.08 | 148570.51 | 42.33 / 115 | 6.7694 / 20.4850 |
| coh_fifo_99 | 0 | 216 | 72.19 | 105.45 | 334.54 | 2.03 / 6 | 5.9776 / 8.4256 |
| coh_fifo_99 | 1 | 216 | 37.63 | 1621.57 | 8601.49 | 27.44 / 99 | 2.4119 / 6.5268 |
| coh_fifo_99 | 2 | 216 | 16.57 | 2197.79 | 18398.40 | 27.32 / 97 | 3.3125 / 17.6107 |
| coh_fifo_99 | 3 | 216 | 17.92 | 2176.23 | 17132.73 | 29.37 / 147 | 3.3478 / 15.7174 |
| coh_fifo_95 | 0 | 216 | 73.93 | 17.32 | 36.79 | 0.00 / 0 | 6.7071 / 9.8352 |
| coh_fifo_95 | 1 | 216 | 50.63 | 168.96 | 797.32 | 0.00 / 0 | 2.7542 / 3.6260 |
| coh_fifo_95 | 2 | 216 | 52.32 | 149.37 | 812.94 | 0.00 / 0 | 3.3959 / 8.5434 |
| coh_fifo_95 | 3 | 216 | 60.70 | 200.47 | 723.49 | 0.00 / 0 | 3.3588 / 6.4144 |
| coh_fifo_80 | 0 | 216 | 78.03 | 19.84 | 42.17 | 0.00 / 0 | 8.0029 / 16.7050 |
| coh_fifo_80 | 1 | 216 | 71.81 | 25.52 | 68.13 | 0.00 / 0 | 2.5386 / 3.6462 |
| coh_fifo_80 | 2 | 216 | 76.12 | 28.67 | 62.13 | 0.00 / 0 | 2.7727 / 5.1048 |
| coh_fifo_80 | 3 | 216 | 83.53 | 26.74 | 57.27 | 0.00 / 0 | 3.5512 / 6.5954 |
| coh_gate_off | 0 | 216 | 0.00 | 93926.04 | 125588.82 | 28.44 / 57 | 3.0492 / 4.5419 |
| coh_gate_off | 1 | 216 | 0.00 | 66102.98 | 295683.51 | 99.59 / 496 | 5.1494 / 15.1717 |
| coh_gate_off | 2 | 216 | 0.00 | 15364.30 | 159644.70 | 53.85 / 123 | 10.1776 / 37.6655 |
| coh_gate_off | 3 | 216 | 0.00 | 15610.11 | 148264.35 | 43.74 / 129 | 9.3151 / 23.8877 |
| coh_gate_99 | 0 | 216 | 72.19 | 40.37 | 358.60 | 2.05 / 6 | 5.2618 / 7.1731 |
| coh_gate_99 | 1 | 216 | 40.85 | 1362.95 | 6789.75 | 16.03 / 78 | 2.7979 / 5.4977 |
| coh_gate_99 | 2 | 216 | 32.18 | 965.93 | 21868.33 | 16.34 / 105 | 4.7198 / 22.9548 |
| coh_gate_99 | 3 | 216 | 39.39 | 1351.01 | 25398.95 | 56.51 / 439 | 5.1321 / 27.7404 |
| coh_learned_off | 0 | 216 | 0.00 | 138028.96 | 240542.70 | 42.65 / 127 | 2.8018 / 5.1782 |
| coh_learned_off | 1 | 216 | 0.00 | 12995.05 | 281338.39 | 87.88 / 483 | 3.3473 / 25.4397 |
| coh_learned_off | 2 | 216 | 0.00 | 3519.52 | 94114.12 | 17.35 / 75 | 8.5015 / 26.4276 |
| coh_learned_off | 3 | 216 | 0.00 | 7635.93 | 69359.97 | 29.90 / 95 | 8.3745 / 26.6478 |
| coh_learned_99 | 0 | 216 | 60.35 | 76.16 | 340.53 | 2.20 / 6 | 3.1478 / 6.5426 |
| coh_learned_99 | 1 | 216 | 38.33 | 346.14 | 6991.14 | 12.13 / 133 | 2.9947 / 6.7190 |
| coh_learned_99 | 2 | 216 | 15.60 | 690.02 | 18208.27 | 15.01 / 93 | 3.5239 / 24.8097 |
| coh_learned_99 | 3 | 216 | 41.00 | 1427.08 | 30823.05 | 46.63 / 449 | 7.1353 / 24.7865 |
