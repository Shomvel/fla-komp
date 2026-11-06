# Block-KOMP 投影消融与 softmax SWA：Depo/Brevo 7-seed 实验

70/70 次训练完成：8 个主实验臂与 2 个 C128 补充臂，各 7 seeds（731–737）。所有 preflight、训练预算、共享参数初始指纹、checkpoint 重载和独立生成审计通过。kernel 未修改。

## 投影值多少

以下“开启投影的变化”都以无投影为基准，按相同 seed 配对。CE 越低越好；生成差异单位为百分点。

- **FIFO：投影稳定改善 CE，但没有稳定改善 Depo 生成，并小幅损害 Brevo 合法率。** 总 CE 变化 -0.2277（95% CI [-0.2791, -0.1876]）；Depo CE -0.3037（95% CI [-0.4144, -0.2179]），Brevo CE -0.1518（95% CI [-0.1667, -0.1375]）。Depo 生成 +1.00（95% CI [-0.45, +2.34]），区间跨零；Brevo 生成 -1.45（95% CI [-2.57, -0.45]），区间不跨零。不能把 CE 收益直接替换成完整答案成功率收益。
- **learned：投影的平均 CE 收益更大，并稳定提升 Depo 生成；Brevo 生成没有明确变化。** 总 CE 变化 -0.3155（95% CI [-0.3831, -0.2570]）；Depo CE -0.4458（95% CI [-0.5758, -0.3331]），Brevo CE -0.1852（95% CI [-0.2201, -0.1406]）。Depo 生成 +3.01（95% CI [+0.78, +5.02]）；Brevo 生成 +0.33（95% CI [-2.68, +3.12]）。两侧都保留相同的 learned score 写入门控，因此这是 learned 配置内部的投影消融，不是门控或淘汰选择的额外消融。

**按 CE 目标，投影有明确价值；按 Brevo 合法生成目标，不能声称投影必要。** FIFO 无投影的 Brevo 合法率为 36.72%，learned 无投影为 36.61%，都保留了较强的 Brevo 表现。本轮没有实现或测量无投影 Triton kernel，不能把 flex 训练用时减少直接换算成生产 kernel 加速。

## 同等状态预算与 SWA

**按用户指定的状态元素数对齐，Block 的优势主要在 Brevo。** FIFO 有投影与 SWA W32 都为每 head 4096 个状态元素。前者相对后者，总 CE -0.5756（95% CI [-0.6514, -0.4951]）；Depo CE -0.1896（95% CI [-0.3584, -0.0132]），Brevo CE -0.9616（95% CI [-0.9984, -0.9341]）。Depo 生成 -0.33（95% CI [-2.46, +1.90]），区间跨零；Brevo 合法率 +29.35（95% CI [+27.68, +30.80]）。Depo CE 的区间较接近零，且未做多重比较校正，其证据强度低于 Brevo 的大幅差异。

扩大到 W96、达到 Block 块内最大原子数后，SWA 的平均 content CE 为 2.4239。Block FIFO 相对它的总 CE 为 -0.3347（95% CI [-0.4800, -0.1643]），Brevo 合法率为 +28.24（95% CI [+26.23, +30.13]）。Depo CE 对比 +0.2165（95% CI [-0.0768, +0.5624]）、Depo 生成对比 -1.12（95% CI [-4.69, +2.01]），均不能得出 Block 稳定更好的结论。

RoPE96 相对 NoPE96 的总 CE 为 -0.2205（95% CI [-0.3648, -0.0980]）；Depo/Brevo 生成分别为 +3.35（95% CI [+1.79, +5.13]）、+2.34（95% CI [+1.23, +3.79]）。在本配方中 RoPE 有稳定收益，SWA 的较差表现不能简单归因于没有位置编码。

这里对齐的是元素数；实际状态字节数、瞬时字典、读写规则和有效参数量并不完全相同，详见方法。所有 SWA 使用 KOMP 的门控 backbone、初始化与统一 LR 配方，没有单独调参。

## 全注意力参考

全因果 RoPE192 的 content CE 为 **2.4168**，Depo CE 2.8013、Brevo CE 2.0322；Depo/Brevo 合法生成分别为 **6.36% / 6.70%**。W96 的对应总 CE 为 2.4239、生成率为 6.47% / 7.03%。

全注意力在这份固定训练配方下，均值没有展示出相对 W96 的大幅提升。它给出的是本次训练达到的参考水平，**不是 softmax attention 的能力上限**。尤其 Depo 的 seed 波动较大；没有更长训练或专门调参，不能据此宣称 Block 普遍优于 Transformer。

## C128 补充实验

C128 开启投影相对关闭投影，总 CE -0.0452（95% CI [-0.0653, -0.0256]）；Depo CE -0.0643（95% CI [-0.1066, -0.0227]），Brevo CE -0.0261（95% CI [-0.0367, -0.0142]）；Depo/Brevo 生成分别为 +0.00（95% CI [-1.23, +1.34]）、+1.23（95% CI [+0.45, +2.01]）。

这组结果必须结合序列长度解读：confirm 中只有约 4.5% 的 Depo、1.5% 的 Brevo 内容预测位置跨过第一次 C128 块末。C 的改变也改变压缩频率和块内可见字典，并非纯计算分块优化。因此不能把 C128 的较小投影效应外推为长上下文中投影无用。

## Block 是否只是“带 delta rule 的 sliding window”

**有投影的 Block-KOMP 包含显式的窗口外系数转移，不能等同于普通的滑动窗口。** FIFO 限定字典 key 来自最近的原子，但被淘汰原子的系数通过联合投影加到保留原子上，历史信息可以跨过字典 key 的窗口继续传递。learned 淘汰则连保留 key 的时间范围也不局限于最近 M 个。

关闭投影后，称其为“按块淘汰、在保留 key 上做 delta rule 的窗口模型”是合理的，但它仍不等同于 softmax SWA：块内字典大小随相位变化，写入系数是相对旧记忆预测的残差，留下的系数仍可能间接编码更早历史；多层堆叠也扩大模型的感受野。

实验支持“投影改善固定训练配方下的预测质量”，尤其 learned 的 Depo；它也显示 Brevo 的大部分生成优势在关闭投影后仍保留。**不能把 Block 相对 SWA 的全部收益归于联合投影**，也不能仅凭这次短序列消融证明改进全部来自长期信息保真。Gaussian 读出、delta 写入、逐 slot 衰减、初始原子与分块机制尚未分别消融。


## 方法与可比性

**冻结与隔离。** kernel 完全来自 `frozen-20261101-proj`，上传前逐文件核对 `SHA256SUMS`。当前主树的 `experiment.py`、`block_experiment.py` 只复制到独立快照；没有修改共享 harness 或 kernel。所有 Block 臂使用同一份 FP32 `block_flex`，q/k/v 由外围 BF16 autocast 产生。关闭 coherence、dedupe、projectron 和 short conv。全程启用确定性算法、设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，关闭 TF32。

**实验臂。** 主实验为 FIFO/learned × 投影开/关四臂，以及 SWA32-RoPE、SWA96-RoPE、SWA96-NoPE、全因果192-RoPE 四臂；每臂 seeds731–737，共56次训练。Block 使用 M32、C64、逐 slot 衰减；learned 同时使用现有 score 头的 `beta *= 2*sigmoid(score)` 写入门控。C128 的 FIFO 开/关投影为预先规定的14次补充实验，在主实验全部成功后才开始。沿用用户授权的16张独立H100并行；首个作业完成全部 preflight 后才提交其余作业。首次错误保留 failure.json 并阻止后续训练，无自动重试。

**SWA 模块与初始化。** 直接复用原 KOMP 已初始化的 q/k/v/out/gate/norm 模块对象，删除不用的 decay、beta 和初始字典参数。逐层核对全部共享参数，以及 embedding、head、FFN 和各归一化参数的 SHA256；报告再次按 seed 检查臂间一致性。SWA 的 q/k 不做 L2 normalization，注意力分数为 `q·k/sqrt(64)`；SDPA 不加 dropout，布尔掩码为 `0 <= query_position-key_position < W`，W 包含当前位置。注意力输出继续使用原有逐 head 的 FP32 RMSNorm、SiLU 输出门控和输出投影。

**RoPE。** 旋转完整的64维 head，按相邻坐标成对，base10000，序列第一个位置为0；角度与旋转在FP32计算，再转回 q/k dtype，v 不旋转。NoPE 仅跳过这一步。W192 覆盖长度192内的全部因果前缀。文档仅右侧补零；padding 位于有效预测的未来，因果掩码阻止它影响非padding的训练损失与生成查询。

**状态预算。** W32 与 M32 均为每 head 4096个常驻状态元素，即每层8192、四层32768。W96 为每 head12288，W192为24576。这是元素数对齐，不是字节数对齐：SWA KV 为BF16，Block的常驻key/系数状态为FP32，learned还携带score记录；Block临时字典在块内从M增长到M+C。本轮未实现增量KV cache，也未测量cache显存或kernel吞吐，生成沿用原harness重算完整序列的方式。跨家族还存在 RoPE、Gaussian/softmax读出、delta写入、可训练初始原子及有效参数量差异，因此它是匹配backbone的质量比较，而非仅投影的单变量比较。

**训练与评估。** width128、4层、2 heads、K/V64、FF256、词表32000；同一份10,001,108个非padding训练token，1 epoch、2231 steps、batch64，固定样本顺序。AdamW betas=(.9,.95)、WD=.1；峰值LR5e-4、5% warmup、cosine到10%，残差输出LR倍率由.1升到.3。没有逐家族调参。confirm CE使用完整split，总content CE为Depo与Brevo两任务content CE的算术平均；生成各取固定前128条样本、greedy。Depo合法率等于exact，Brevo允许其他合法拓扑序。confirm此前已用于探索，不能视作全新外部测试集。

**Preflight。** 十个臂均检查初始化与16步更新后的未来后缀干预，覆盖窗口和chunk边界；同输入重复forward必须逐位一致；16步loss下降；learned score梯度非零且有限。SWA另在单个mixer输入上检查窗口外扰动不影响当前输出，而最早仍在窗口内的位置能够影响输出。窗口局部性只针对单层mixer，多层网络会扩大感受野。W192的边界检查使用长度201输入；正式质量实验仍为长度192。

**统计。** 每臂报告均值和样本标准差；按seed配对差值，以RNG91731做20,000次配对bootstrap，给出percentile95%CI。未做多重比较校正；区间只描述固定数据、顺序与生成样本下的初始化波动。预设对比为FIFO/learned各自的无投影减有投影、FIFO有投影减SWA32/96、RoPE96减NoPE96，以及C128无投影减有投影。全注意力是本配方下的参考，不能视作已证明的性能上限。

**时间。** 对验证、评估和生成使用CUDA同步计时。训练循环加记录开销 = summary总时间 − 测得的评估时间 − 生成时间；checkpoint重载评估另行记录，不在summary总时间内。这不是纯优化步骤GPU计时。flex与SDPA实现不同，不能把逐run墙钟时间当作kernel速度比较。

**解释边界。** 关闭投影只去掉一条明确的跨chunk系数转移路径，并没有去掉全部较早token依赖：保留下来的delta系数本身来自含旧记忆的预测残差。FIFO约束的是字典key的时间范围，不一定是系数所含信息的全部历史；learned还可以保留非最近的key。因果性验收通过，也不意味着整个模型只有W-token感受野。

## 序列长度与投影影响范围

下表由只读数据脚本统计；预测位置 i 用于预测 token i+1。只有跨过块末的预测能在该层直接受到投影影响。C128 的答案大多出现在第一次压缩之前，不能把这组短序列结果外推到长上下文。

| split / task | documents | mean / max length | content predictions after C64 | after C128 |
|---|---:|---:|---:|---:|
| train/depo | 71392 | 85.79 / 171 | 59.03% | 4.72% |
| train/brevo | 71392 | 54.29 / 168 | 49.64% | 1.20% |
| confirm/depo | 1024 | 86.93 / 163 | 59.08% | 4.52% |
| confirm/brevo | 1024 | 55.10 / 148 | 51.86% | 1.45% |

## 脚本生成的完整结果

以下内容从 `physics_lm/block_runs_proj/TABLE.md` 原样插入，由 `block_report_projection.py` 从 `summary.json` 生成。

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

## 验收与资源清理

- 冻结 manifest 校验、FP64 数值对拍、所有十臂 preflight 均存档。提供的 CPU 脚本实际枚举 32 组配置；验收以实际输出计数为准。
- 独立生成审计检查 17,920 条记录：Depo 从输入字典执行跳转，Brevo 从输入图重新求祖先闭包，与所有 summary 一致。
- 两个有投影C64对照臂的14次独立重跑，其confirm指标和生成统计与上一轮确定性coherence-off对照完全一致；未复用旧结果文件。证据为 `evidence/control-reproduction.json`。
- 训练均为 2231 steps、10,001,108 个非 padding token；所有共享参数按 seed 完整对齐。最终源文件哈希保持不变。
- 结果与日志在 `physics_lm/block_runs_proj/`，不含 `.pt`。源码快照、计划、环境、启动配置及验收 JSON 在 `evidence/`。
- 本轮独立目录 `/gpfs/users/linlizhi/komp-proj-swa-20261101`；16 个自建作业 alias 为 `komp-proj-swa-20261101-r1` 至 `-r16`。均记录正常终态、按精确 alias 删除，并重新查询确认消失；证据在 `evidence/final-cleanup/`。
- 上一轮 coherence 的 r19–r30 已按用户要求取消，保存取消前 Running 状态、删除回执和空查询；没有把取消伪装为训练完成。证据在 `physics_lm/block_runs_coh/evidence/fine-cancel-cleanup/`。未停止 Claude 的作业。
