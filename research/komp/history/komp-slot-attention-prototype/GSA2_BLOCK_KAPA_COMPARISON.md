# GSA2 与 Block-KOMP / KAPA：7-seed 补充比较

GSA2 新增 21/21 次训练完成，三种配置各 7 seeds（731–737）；与上一轮 70 次冻结 Block/KAPA/基线结果合并展示，共 91 次。preflight、源码哈希、训练预算、checkpoint 重载和独立生成审计全部通过。没有更改冻结 kernel 或原有 70 次实验报告。

## 结果怎样解读

1. **GSA2 的 LR 选择会显著影响这里的比较。** 原生层 LR1e-3 相对 LR5e-4：content CE -0.6250，95% CI [-0.7153, -0.5212]（区间不跨零）；Depo 生成 +6.36 个百分点，95% CI [+3.91, +8.48]（区间不跨零）；Brevo 合法生成 +8.37 个百分点，95% CI [+5.92, +11.05]（区间不跨零）。1e-3 来自此前验证集搜索，本轮没有重新选参；不能仅凭统一 LR5e-4 的表现判定 GSA2 上限。
2. **高 LR GSA2 与 Block learned 应分任务看。** GSA2 LR1e-3 相对 Block learned-C64：content CE -0.0248，95% CI [-0.1172, +0.0715]（区间跨零）；Depo 生成 +4.91 个百分点，95% CI [+2.57, +7.37]（区间不跨零）；Brevo 合法生成 -16.96 个百分点，95% CI [-19.20, -14.96]（区间不跨零）。与 Brevo 均值最好的 Block gate-only 相比，Brevo 差值为 -19.75 个百分点，95% CI [-22.21, -17.41]（区间不跨零）。总 CE 的接近不能解释成两项任务能力相同，也不是统计等效性证明。
3. **KAPA 与 GSA2 的 Depo 生成差距仍需谨慎。** GSA2 LR1e-3 相对 scalar KAPA：Depo 生成 +2.23 个百分点，95% CI [-0.33, +4.80]（区间跨零）；Depo CE -0.0369，95% CI [-0.1934, +0.1089]（区间跨零）；Brevo 合法生成 +13.17 个百分点，95% CI [+11.05, +15.07]（区间不跨零）。不同任务的均值优势不能单独归因于某一种 transition。
4. **原生 short conv 在 LR5e-4 下没有带来更好的 CE。** 同 LR 下 conv 相对 no-conv：content CE +0.2918，95% CI [+0.1557, +0.4491]（区间不跨零）；Depo 生成 +0.56 个百分点，95% CI [-1.67, +2.57]（区间跨零）；Brevo 合法生成 -1.34 个百分点，95% CI [-4.13, +1.67]（区间跨零）。这是原生层配置消融，包含其对应激活和初始化行为；不能推导 short conv 普遍无效。本轮没有 LR1e-3 的 no-conv 臂。
5. GSA2 LR1e-3 相对 GDN：content CE -0.3940，95% CI [-0.4669, -0.3236]（区间不跨零），Depo 生成 +8.59 个百分点，95% CI [+6.36, +10.71]（区间不跨零），Brevo 合法生成 +9.15 个百分点，95% CI [+6.92, +11.38]（区间不跨零）。两者参数、状态和原生层配方并非完全一致。

## 配方、可比性与局限

- 直接使用已有 FLA 源码中的 `fla.layers.gsa2.GatedSlotAttention2`，Oja2 slot memory → GDN2 value memory；没有重写 transition 或改内核。上游目录无 Git 元数据，以冻结文件哈希为身份依据。`fla/layers/gsa2.py` SHA256：`831e7658047a92ad2422526fc7e6f1f2095ea6c45a3f1340d5928562e7fbd234`。
- 三个臂：原生 q/k/v/w conv4 + LR5e-4；相同原生 conv4 + LR1e-3；关闭 q/k/v/w conv + LR5e-4。WD=.1，并尊重原生 `_no_weight_decay` 标记。峰值 LR 以外沿用 5% warmup、cosine 到10%、残差输出 LR multiplier .1→.3。原生 optimizer 的 WD 例外与旧 Block/KAPA 配方不同。
- `1e-3 / WD .1` 来自此前 `gsa2_runs/selection.json` 的验证集选参记录。本轮只是把该配方与统一 LR 一并复验；不是在同等调参预算下比较所有架构的最优成绩。选参记录副本及哈希在本轮 evidence 中。
- 数据、训练顺序、非 mixer backbone、非 mixer 初始参数均与 v4 对齐：width128、4层、2 heads、K/V64、FFN256、vocab32000，10,001,108 个实际输入 token、2231 steps、batch64、1 epoch，右侧 padding 至192。每个 run 实际计数核对一致。
- **不是等状态预算比较。** 原生 GSA2 只支持 M64/128/256，本轮 M64；循环矩阵每 head 8192 元素。Block 的 keys+coefficients、GSA 两层 memory、GDN 矩阵均为4096；fixed-center KAPA 动态 coefficients 为2048，另有固定可训练中心参数。上述计数不含 short conv cache。GSA2 约938万参数，Block 约897万；详见配置表。
- **不是只替换 transition 的单变量实验。** GSA2 原生 mixer 的初始化、激活、gate、short conv、WD 规则与旧模型不同。只对齐并核验非 mixer 参数初值，不声称所有 mixer 初始权重相同。no-conv 臂保留原生关闭卷积时的行为。旧 Block/KAPA 均无 short conv。
- preflight 对三臂分别在初始化与16次更新后，检查 train/eval 模式下的未来 token 干预、31/32/63/64/65/127/128/129 边界、batch 隔离；未来梯度严格为0，所有参数有有限梯度，16步 loss 下降。preflight 在正式训练前执行，每次训练先核对 ticket 的源码哈希。
- 同一 confirm split；每任务固定前128条做 greedy 生成。Depo valid=exact；Brevo 允许不同合法拓扑序。独立审计重新执行输入字典跳转与输入 DAG 祖先闭包，新增 **5,376 条生成记录**全部与 summary 一致。
- 均值±样本标准差，seed 配对 percentile bootstrap 20,000次，固定 RNG91731；生成差值单位为百分点。CI只描述固定数据/数据顺序/评测样本下的初始化波动，没有多重比较校正。7个seed、128个生成样本仍有限；confirm 此前已被查看，这不是新的外部留出测试。
- `seconds` 是训练+验证+最终生成的整体时长，排除进程启动和 checkpoint 重载；它不是纯 kernel benchmark。GSA2 的更大状态、双层 memory 和原生 conv 都影响成本，不能把时长差直接归因于单个算子。

## 均值与 seed 波动

![Comparison](/Users/shom/Downloads/komp-slot-attention-prototype/physics_lm/block_runs_v4/gsa2/COMPARISON.png)

误差棒是样本标准差，不是置信区间。正式配对区间见下面的脚本生成表。

## 脚本生成的完整表

以下逐字插入 `physics_lm/block_runs_v4/gsa2/TABLE.md`。数值全部由 `gsa2_trials_report.py` 从91个 summary 生成；原先 v4 的 TABLE.md 与 statistics.json 未覆盖。

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

## KAPA 与 Block-KOMP 融合

融合推导与 CPU FP64 参考实现见 [KAPA_BLOCK_FUSION.md](/Users/shom/Downloads/komp-slot-attention-prototype/KAPA_BLOCK_FUSION.md)。结论是可以保留块仿射转移和 scan：分别加入“因果局部 KAPA 写入方向”和“块末保护式压缩”，优先测试后者。p=0 精确恢复冻结 Block baseline；505项数值检查通过，涵盖输出、终态、所有输入梯度与跨块因果性。

不能使用依赖整个 chunk 未来 key 的 Gram inverse 来定义当前写入方向；参考中有实际泄露反例。当前成果是数学原型和验收证据，没有融合 Triton、融合模型质量收益或 learned 淘汰 anchor-cache 的实现证据。既有 KAPA 与 Block 的任务偏好仅构成研究动机，不能证明融合必然两者兼得。

## 产物、重现与清理

- `physics_lm/gsa2_trials.py`：独立原生层实验 harness；未改旧 gsa2/lrwd/shortconv/decay/experiment harness。
- `physics_lm/gsa2_trials_report.py`：完整表与 statistics.json；`gsa2_trials_plot.py`：PNG/SVG 图；`gsa2_comparison_document.py`：本报告。
- `physics_lm/block_runs_v4/gsa2/`：21个 summary、manifest、metrics、生成记录、日志、preflight，不含 `.pt`。
- `evidence/`：冻结快照、源码哈希、计划、选参来源、H100环境、启动配置、source/result/generation审计与退出状态。
- 独立远端目录 `/gpfs/users/linlizhi/komp-gsa2-trials-20261101`；alias `komp-gsa2-trials-20261101-r1`，单H100。已保存作业终态，按精确 alias 删除，并查询确认为空。`job-terminal.json`、`job-delete.json`、`job-cleanup.json` 保存证据。

重建统计：`python3 physics_lm/gsa2_trials_report.py`。绘图：`uv run --no-project --python 3.12 --with matplotlib==3.10.7 python physics_lm/gsa2_trials_plot.py`。重建报告：`python3 physics_lm/gsa2_comparison_document.py`。
