# BF16 投影与稳态速度测试

结论：在当前 alpha0=.995、temperature=.5、ridge=1e-4、M32 的已训练100M模型上，纯BF16投影实验和BF16数据+FP32求解实验都在真实batch监督位置产生非有限hidden，两个任务均拒收。BF16投影暂不可直接用于本配方。FP32投影的实际验证CE与FP64非常接近，但保留此前梯度偏差及边界压力测试失败记录。

模型输入Q/K/V原本已是BF16。此处只降低投影几何的表示/消元精度，状态始终FP32，不包含BF16状态传播改动。所有实验使用隔离内核副本，冻结训练源码未改。

## 精度口径

- FP64/FP32：原内核投影模式。
- BF16投影/消元：Gram、RHS及P/H/R/N为BF16，消元逐步舍入BF16；归一化、归约、状态及几何反向累加仍FP32。
- BF16数据+FP32求解：同样将Gram、RHS及P/H/R/N量化BF16，但加ridge与消元保留FP32。此路径不是纯BF16求解。
- 实验Gram构造仍使用FP32/tf32x3再量化，未声称是最优BF16TensorCore实现；量化后的反向属于近似梯度。
- BF16下1+1e-4舍入为1，当前ridge在纯BF16正则矩阵对角线上消失；FP32求解混合版仍出现非有限输出，说明仅保留求解精度不足以让该实验配置可用。

## 压力测试

B1/T65/H1/M32/K64/V64；random/near/duplicate/strong，FP32/BF16输入，输出/状态/七组输入梯度共80项。原阈值未放宽。

| 投影模式 | 通过项数 | 结论 |
|---|---:|---|
| FP64 投影 | 80/80 | PASSED |
| FP32 投影 | 79/80 | FAILED |
| BF16 投影/消元 | 44/80 | FAILED |
| BF16 投影数据 + FP32 求解 | 70/80 | FAILED |

## 单算子稳态速度

H100 80GB，BF16 Q/K/V，FP32门与状态；K/V=64，KOMP M32/FIFO/C16，GDN原生chunk64。包含归一化、final state及相应反向；不含模型线性投影、conv、optimizer。KOMP keys+coefficients与GDN state均4096元素/head，但算法不同，不代表质量匹配。

5次预热后测3轮×20次，以下为CUDA graph event每次耗时的轮间中位数，编译排除。BF16两行只是在随机计时输入上有限；真实模型已拒收，所以不能作为可用训练速度。

### B1 / T8192 / H2

| 模式 | 前向 ms | 前反向 ms |
|---|---:|---:|
| FP64 投影 | 3.6228 | 10.6985 |
| FP32 投影 | 0.7145 | 2.1140 |
| BF16 投影/消元 | 0.7482 | 2.3915 |
| BF16 投影数据 + FP32 求解 | 0.7203 | 2.2593 |
| FLA GDN（BF16输入） | 0.1364 | 0.4637 |

### B1 / T1024 / H8

| 模式 | 前向 ms | 前反向 ms |
|---|---:|---:|
| FP64 投影 | 1.8499 | 5.4879 |
| FP32 投影 | 0.3863 | 1.1183 |
| BF16 投影/消元 | 0.4027 | 1.2655 |
| BF16 投影数据 + FP32 求解 | 0.3878 | 1.1876 |
| FLA GDN（BF16输入） | 0.0468 | 0.1163 |

8K前反向FP32较FP64约5.06倍快；BF16严格版较FP32慢13.1%，混合版慢6.9%。Exact FP32较GDN慢约4.56倍。

## 完整模型固定真实batch速度

100,526,392参数、alpha0=.995的同一checkpoint。每GPU2篇文档，共16篇；各rank按自身真实文档长度padding，Depo T544–1232，Brevo T32–752。5次预热，3轮×10次CUDA event，取每轮最慢rank再取中位数。包含backbone与masked LM head CE；前反向含zero_grad和activation checkpoint，不含optimizer及DDP梯度通信。

| 任务 | 投影 | 前向 ms | 前反向 ms |
|---|---|---:|---:|
| depo2 | FP64 投影 | 73.93 | 289.73 |
| depo2 | FP32 投影 | 44.15 | 195.92 |
| depo2 | BF16 投影/消元 | 非有限，跳过计时 | 非有限，跳过计时 |
| depo2 | BF16 投影数据 + FP32 求解 | 非有限，跳过计时 | 非有限，跳过计时 |
| brevo2 | FP64 投影 | 47.62 | 190.92 |
| brevo2 | FP32 投影 | 44.13 | 200.50 |
| brevo2 | BF16 投影/消元 | 非有限，跳过计时 | 非有限，跳过计时 |
| brevo2 | BF16 投影数据 + FP32 求解 | 非有限，跳过计时 | 非有限，跳过计时 |

Depo FP32前反向约快1.48倍；Brevo未观察到提速，FP32三轮为203.12/183.18/200.50ms，不能把算子5.06倍的加速套用到模型或DDP训练。

## 对拍更正

原operator-results.json中8k_or_1k_difference_vs_fp64的梯度字段使用了各模式独立随机上游梯度，因此无效，不用于结论。速度、同文件stress检查及真实模型对拍不受影响。共享上游梯度的独立更正保存在evidence/operator-shared-grad-correction.json；原文件与脚本保留以便追溯。

原始证据：precision_bench_20261105/operator-results.json、precision_bench_20261105/variants.json、evidence/bf16-model-depo2.json、evidence/bf16-model-brevo2.json。

共享上游梯度更正已完成，两形状随机输入全部有限且通过原BF16输入门槛。8K对FP64参考的输出RMS / 最大七项输入梯度RMS：

| 模式 | 输出RMS | 最大梯度RMS |
|---|---:|---:|
| fp32 | 0.003193% | 0.003497% |
| bf16_projection | 0.329133% | 0.458643% |
| bf16_projection_mixed | 0.305915% | 0.305753% |

上述是随机输入结果，不能覆盖真实checkpoint出现非有限hidden的拒收。原算子性能数据不需要重测；错误梯度字段保留并由operator-gradient-erratum.json声明作废。
