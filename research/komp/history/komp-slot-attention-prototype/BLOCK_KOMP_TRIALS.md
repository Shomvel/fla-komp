# Block-KOMP 与 KAPA：冻结 kernel 的 7-seed Depo/Brevo 实验

**70/70 次训练完成，10 个臂各 7 seeds（731–737），所有 preflight、训练预算与 checkpoint 重载检查通过。** 本轮只修改或新增实验 harness 和报告；没有改任何 kernel，没有改 `BLOCK_KOMP.md`、`README.md` 或其他 agent 的 harness/runs。

## 哪些结论站得住

1. **逐 slot 衰减是 Block-KOMP 最稳定、最大的收益来源，主要体现在 Brevo。** 相对 scalar FIFO，slot FIFO 的 content CE 差值为 -0.3219，95% CI [-0.3787, -0.2624]；Brevo 合法生成提升 +23.88，95% CI [+21.76, +26.23] 个百分点。Depo CE 与生成率的区间跨零，不能说逐 slot 衰减在两个任务上同样有效。
2. **slot FIFO 优于逐 token gsa_slot 和 GDN 的总 CE/Brevo 结论在这 7 seeds 下保留。** 对 gsa_slot，content CE 差值为 -0.1553，95% CI [-0.1894, -0.1308]，Brevo 提升 +2.57，95% CI [+0.67, +4.69] 个百分点。对 GDN，分别为 -0.2842，95% CI [-0.3647, -0.1983]、+25.33，95% CI [+23.10, +27.34] 个百分点。跨 harness 的 WD 策略差异见方法部分。
3. **learned-C64 相对 slot FIFO 的平均 CE 改善稳定，但 Brevo 额外提升不稳定。** 总 CE 差值 -0.0850，95% CI [-0.1337, -0.0324]，Depo CE 差值 -0.1713，95% CI [-0.2717, -0.0685]，Depo 生成提升 +1.90，95% CI [+1.12, +2.68] 个百分点；Brevo 生成差值 +0.78，95% CI [-0.89, +1.90] 个百分点，区间跨零。
4. **C32 没有超过 C64。** learned-C32 相对 learned-C64 的总 CE 为 +0.0405，95% CI [+0.0131, +0.0730]，更差；生成率的区间跨零。Block-KOMP 的 chunk size 会改变压缩/淘汰频率，因此这是质量消融，不是纯计算分块等价性测试。

## learned 的收益来自门控还是淘汰

应分任务回答，不能只看总 CE。

- **门控本身有稳定收益。** gate-only 相对 FIFO 的总 CE 为 -0.0398，95% CI [-0.0551, -0.0263]，Brevo CE 为 -0.0488，95% CI [-0.0648, -0.0335]；Brevo 合法生成提升 +3.57，95% CI [+1.56, +5.13] 个百分点。gate-only 保留了与 learned 相同的 score 头、`beta *= 2*sigmoid(score)` 和初始 score 参数，但淘汰仍为 FIFO。
- **在相同门控上加 learned 淘汰，收益偏向 Depo，并损害 Brevo。** learned 相对 gate-only 的 Depo CE 为 -0.1405，95% CI [-0.2471, -0.0259]；Brevo CE 为 +0.0501，95% CI [+0.0297, +0.0770]，更差。gate-only 的 Brevo 合法生成比 learned 高 +2.79，95% CI [+2.12, +3.57] 个百分点，7 个配对 seeds 的区间不跨零。
- **不能断言淘汰主导总 CE 收益。** learned 相对 gate-only 的总 CE 为 -0.0452，95% CI [-0.1003, +0.0118]，区间跨零；两者 Depo 生成差值的区间也跨零。平均值上门控和追加选择都有贡献，但仅凭 7 seeds 不足以给出稳定的贡献比例。

这里的“追加淘汰收益”包含选择与门控的交互，也包含选择改变了 score 学习路径的影响。本轮没有 eviction-only 臂；硬 top-M 的排序本身不可微，score 通过写入门控获得梯度。`initial_score` 没有可微排序通路，gate-only 保留它是为了与 learned 对齐参数结构，不能将其描述成成功学到的初始淘汰分数。

若以 Brevo 合法生成率为目标，这一组设置中优先 gate-only；若以平均 content CE 或 Depo CE 为目标，learned-C64 的样本均值最好，但总 CE 相对 gate-only 的额外改善还未稳固。

## KAPA 的结果

KAPA 同样完成两个臂各 7 seeds，使用冻结的 chunk-parallel Triton fwd/bwd：固定可训练中心、p=2、chunk16、Gaussian 温度0.5、Gram ridge0.01、约束正则0.05。它在序列内不替换中心；原始 q/k/中心不做 L2 normalization。

- scalar KAPA：content CE **2.1423 ± 0.0945**；Depo CE **2.3123 ± 0.1574**，Depo 生成 **9.71 ± 1.85%**；Brevo CE **1.9724 ± 0.0553**，Brevo 合法生成仅 **7.14 ± 2.45%**。
- slot KAPA：content CE **2.2365 ± 0.1849**；Depo 生成 **6.25 ± 2.26%**，Brevo 合法生成 **7.37 ± 2.95%**。slot 相对 scalar 的总 CE 差值为 +0.0941，95% CI [-0.0213, +0.2150]，区间跨零；Depo 生成差值为 -3.46，95% CI [-5.36, -1.56] 个百分点，slot 更差。
- **当前 KAPA 偏向 Depo，不能替代 Block-KOMP 在 Brevo 上的表现。** 预设对比中，slot KAPA 相对 slot Block 的 Depo CE 为 -0.5435，95% CI [-0.7662, -0.2945]，但 Brevo CE 为 +0.8331，95% CI [+0.8037, +0.8583]、Brevo 合法率为 -29.13，95% CI [-31.03, -26.90] 个百分点。均值更好的 scalar KAPA 也有明显的 Brevo 短板。

表中 scalar KAPA 对 Block/GDN 的额外配对比较标记为 exploratory：是在看到前6个 KAPA seeds 后加的分析，不是预先锁定的对比。训练配置和剩余 seed 没有因此改变。没有 p=0/no-protection 消融，也没有 τ/ρ/λ 调参，因此不能把 Depo 收益单独归因于 anchor protection，或据此断言 KAPA 的性能上限。

## 方法与可比性

- 数据：同一个去重、完整文档、不拼接/截断的 PhysicsLM4 Depo/Brevo corpus，10,001,108 个非 padding 训练 token，1 epoch、2231 steps、batch64。每个 run 都核对实际 token/步数。文档右侧 padding 到192。
- 模型：width128、4层、2 heads、K/V64、M32、FFN256、词表32000；所有臂均不加 short conv。保留原 harness 的按模块命名 seed 初始化和固定样本顺序，报告脚本核对存档的共享参数初始指纹。
- 优化器：AdamW betas=(.9,.95)、峰值 LR5e−4、5% warmup、cosine 到10%，残差输出 LR multiplier .1→.3。Block/KAPA/GDN/GSA 的 WD=.1；原 `decay_experiment.py` 的 gsa_slot 对 decay.bias 免 WD，本轮保留其原契约。这是跨 harness 比较的局限，未把它伪装成仅 transition 的严格单变量实验。
- 内核：所有 `komp_slot/`、`komp_bench/`、`benchmark_fla.py` 均来自 `~/Downloads/komp-accel-scratch/frozen-20261101-r3/`。`block_fast.py` SHA256 为 `a8e4c72527473bb1928841df26c5e4625e9441f5b7f2b309f420cabf8d0c74ce`。快照内 preflight 后才训练；每次运行前、全部结束后都复核源码哈希。FLA 目录无 Git 元数据，保存并复核了595个 Python 文件哈希。
- baseline 用 `block_baseline_v4.py` 调用未改动的 `experiment.py` / `decay_experiment.py` 函数，只重定向结果路径；其原有 preflight 全部执行。所有731/732也重跑，未拼接旧结果。
- Block preflight 在初始化和16次更新后干预未来 token，检查 C32/C64 边界；新 score 头检查非零有限梯度。KAPA 检查 chunk 边界因果性、中心梯度和16步 loss 下降。所有臂 preflight 在任何正式训练前通过。
- 评测：val/test/confirm 用完整 split；生成使用固定的前128条 confirm 样本/任务、greedy。Depo valid=exact，Brevo valid允许不同合法拓扑序。独立审计从输入字典执行 Depo 跳转、从输入图计算 Brevo 祖先闭包，不用 gold 的节点集合替代推导；**17,920 条生成记录与 summary 全部一致**。
- 统计：每臂 n=7，均值±样本标准差；按 seed 配对差值，20,000 次配对 bootstrap、RNG91731、percentile95% CI。CI仅反映固定训练集、固定数据顺序和固定生成样本下的初始化波动，不包含数据集抽样不确定性，不做多重比较校正。特别接近零的区间不应当作强结论。
- 此前已看过相同 confirm 集上的探索结果，所以这不是全新、未触碰的外部测试集。新增 seeds 增强的是 seed 稳定性证据。
- E3 跳过：现有 harness 将长度192与 M32 写成共享常量，没有受支持的逐 run 参数入口；按约束不修改共享常量。也没有长序列、扩规模或超参数最优性声明。

## 脚本生成的完整表

以下内容逐字插入 `physics_lm/block_runs_v4/TABLE.md`，由 `physics_lm/block_report_v4.py` 从70个 `summary.json` 生成；没有手填表格数字。

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

## 产物与清理

- 结果、日志、preflight、生成记录、manifest：`physics_lm/block_runs_v4/`。基线在 `baselines/`，KAPA 在 `kapa/`，Block 在 `results/`；不含 `.pt`。
- 机器可读统计：`statistics.json`；所有输入 summary 均有 SHA256。运行 `python3 physics_lm/block_report_v4.py` 可重建表格。
- 冻结快照与启动配置：`evidence/snapshot/`、`snapshot-sha256.json`、`frozen-verification.json`、`job.yaml`、`runner.sh`、`run_trials.py`、`plan.json`、`METHOD.md`。
- 验收回执：`evidence/result-audit.json`、`independent-generation-audit.json`、`final-source-audit.json`、`environment.json`、`fla-provenance.json`、`completion.json`。
- 独立目录 `/gpfs/users/linlizhi/komp-block-trials-20261101`；任务 alias `komp-block-trials-20261101-r1`，单H100。作业正常完成、exit code0；保存终态后按精确 alias 删除，重新查询为空。证据为 `evidence/job-terminal.json`、`job-delete.json`、`job-cleanup.json`。未停止或修改其他作业。
