# KOMP 改动与评测汇总

整理日期：2026-11-13，北京时间。资料包含 kernel 优化、340M 模型训练与下游评测、1B 续训和 recurrent inference 候选。性能与分数来自已保留的实验回执；本次只读采集远端状态，没有重新运行 GPU 评测。

## 结果索引

| 内容 | 文件 |
| --- | --- |
| Kernel 改动、同卡计时与验收限制 | [KERNEL_RESULTS.md](KERNEL_RESULTS.md) |
| 8 模型全部 184 个下游分项 | [MODEL_RESULTS.md](results/20261113/MODEL_RESULTS.md) |
| 下游原精度得分及 checkpoint/plan 哈希 | [DOWNSTREAM_SCORES.csv](results/20261113/DOWNSTREAM_SCORES.csv) |
| 340M 最终 validation/test CE、PPL | [MODEL_LOSS.csv](results/20261113/MODEL_LOSS.csv) |
| 215 条分口径的 kernel/完整训练步计时 | [KERNEL_TIMINGS.csv](results/20261113/KERNEL_TIMINGS.csv) |
| 描述性均值、1B 最新状态 | [summary.json](results/20261113/summary.json) |
| 原始源码、配置、接受与失败回执 | [20261113 资料目录](history/20261113/README.md) |

## 实现与变体

仓库的包内 API 提供 Exact、Block 和 value decay，以及原有 flexible Block、投影、BSGD、KAPA 等实现。下表中的后续训练变体与优化候选保存在日期归档；这些冻结源码与包内默认实现分别标识。源码归档本身不证明候选已通过包内 API 的集成验收。

| 实现 | 主要改动 | 源码与设置 |
| --- | --- | --- |
| Exact Triton | 保留每 token 压缩；计算块用于执行并行 | [原始实现](history/20261113/komp-exact-triton-20261105/) |
| Exact 优化第一轮 | 稳定 FP32 geometry、去除逐 token 状态与逆矩阵 tape、拆分反向 | [源码与结果](history/20261113/komp-exact-opt-20261105/RESULTS.md) |
| Exact 优化第二轮 | 字典版本复用 LDLᵀ、直接 RHS sweep、分层 affine scan | [源码与结果](history/20261113/komp-exact-opt2-20261105/RESULTS.md) |
| Exact 模型重算优化 | 图内 geometry cache、去除未使用的全局 gc tape | [固定 340M checkpoint 对照](history/20261113/exact-komp-efficiency-research-20261106/REPORT.md) |
| Block C128 | 四个 32×32 子块求三角逆、拆分输出反向 | [源码与验收](history/20261113/komp-c128-triton-20261101/C128_TRITON.md) |
| Value decay | 换坐标复用原生 Exact/Block；累计通道残差跨度超过 24 时回退 | [设计与计时](history/20261113/komp-slot-attention-prototype/VALUE_DECAY_NATIVE.md)；[包内集成](VALUE_DECAY_INTEGRATION.md) |
| +123 | sigmoid 输出门、rank64 门投影、多时间尺度初始化 | [FF2272 设置](history/20261113/komp-kda-options-ff2272-340m-20c-20261106/EXPERIMENT.md) |
| GDN slot + learned | `-exp(A_log)*softplus(z+dt_bias)` 衰减、learned top-M 淘汰、score 写入门 | [定义与性质](history/20261113/komp-gdn-slot-learned-123-340m-20c-20261107/PROPERTIES.md) |
| Value + learned +123 | 在 value 通道衰减中组合 learned 与 123 | [训练与验收](history/20261113/komp-value-decay-learned-123-340m-20c-20261107/README.md) |
| Learned+123 C128 v2 | 槽位衰减反向直接归约、线程布局、两阶段 UT 反向、保留 BF16 舍入的 FF 融合、图内重算缓存 | [优化报告](history/20261113/komp-learned123-c128-opt-20261109/OPTIMIZATION_REPORT.md) |
| Recurrent inference r1/r2/r3 | token 路径、数值诊断、复用 native Block 的有界开放块缓存 | [r3 设计和限制](history/20261113/komp-recurrent-inference-20261107-r3/README.md) |

`+123` 同时打开三项选项。当前结果没有把三项各自的因果贡献分离。早期 FF1824 组合被 FF2272 参数规模接近的重启组替代；两份训练设置均保留。

Learned 的离散排序与索引选择使用 detach。score 的连续梯度经过 `beta*2*sigmoid(score)` 写入强度；不能据此声称离散淘汰本身可微。衰减参数属于物理 slot。learned metadata 和 Block 开放块临时状态需要单独计入推理缓存。

## 340M 质量结果

8 个 checkpoint 已完成同一套 23 分项，共 163,656 条结果。除 Exact partial 仅训练 3.758B tokens，其余训练 6.8B。模型约 338–341M 参数，存在参数量与 FFN 设置差异；当前为单 checkpoint 比较。

| 模型 | Validation CE | Test CE | NIAH single1 四长度均值 | NIAH single2 四长度均值 |
| --- | ---: | ---: | ---: | ---: |
| KDA | 2.527196 | 2.522922 | 19.35% | 33.15% |
| Learned | 2.593220 | 2.587377 | 36.35% | 39.30% |
| Learned+123 | 2.554563 | 2.549447 | 68.60% | 50.70% |
| Value+learned+123 | 2.555611 | 2.550842 | 未评测 | 未评测 |

Validation/test 每个 split 各 4,194,304 个监督 tokens。NIAH 均值只在同一变体的 1024/2048/4096/8192 四长度间等权计算，没有生成跨任务总分。

Learned+123 在 single1 的 1K/2K/4K/8K 分别为 91.60/83.00/44.60/55.20%。single2 的 8K 降至 3.20%，低于 Learned 的 8.60% 与 KDA 的 5.00%；single3 各模型均接近失败。

BASED 的 NQ/DROP 仍由 KDA 领先：KDA 为 10.23/15.52%，Learned+123 为 8.24/14.14%。SWDE 两者为 8.01/8.10%，没有显著性证据支持稳定优势。PhoneBook 使用历史 `single_break`，实际 prompt 约 80–244 tokens，其分数不支持长上下文能力结论。

Value+learned+123 已完成 6.8B 训练，validation/test CE 来自完成回执。该 checkpoint 没有纳入上述 8 模型下游计划；其下游分数保持为空。

## 1B 与推理候选状态

2026-11-13 只读核对 BP 文件与 Northjob API：Learned+123 1B 的目标为 50,000,000,000 监督 tokens，平台 identity 为 `job-35a0f599ed70-20261110172046`。当前平台状态为 `Aborted`，没有最终训练完成回执。

| 项目 | 已观测结果 |
| --- | --- |
| 参数量 | 1,003,181,056 |
| 最后 metrics | step 40579，42,550,165,504 tokens，85.10% 目标预算 |
| 最后已记录 checkpoint | step 39936，41,875,931,136 tokens |
| 最近 validation | step 39936，CE 2.208300，PPL 9.100236，4,194,304 监督 tokens |
| 最后 100 步平均耗时 | 1.648554 s/step，原始 32 H100 训练记录 |
| 下游评测 | 未启动，0/23 项；watcher 因训练未完成而退出 |

4 个节点均留下 exit code 0，但平台 `Aborted`、最终 50B 完成回执缺失且实际 metrics 未达预算。不能将节点退出码解释为 50B 训练完成。中止原因未查明。[文件快照](results/20261113/remote-filesystem.json)与[平台快照](results/20261113/remote-platform.json)保留了此次检查结果。

Recurrent r1 的算子通过没有转化为完整模型通过。Block T31 的 logits RMSrel 为 0.00300777，超过冻结的 0.003 门限；r2 只完成诊断。r3 改为复用 native Block 的开放块缓存，目前有 CPU 性质检查，尚未取得 GPU 模型、greedy 分数与同卡速度的完整验收。Value 两变体在 r3 入口中明确拒绝。不能将旧 token 路径的算子计时标成 r3 推理加速。

## 重建与验证

在仓库根目录运行：

```bash
python research/komp/summarize_results.py --check
python validation/komp/verify_sources.py
```

第一条命令验证本次 5,219 个归档路径的内容哈希与仓库内链接，并逐行核对 184 个得分的样本量、分数、预算、plan 和 checkpoint 哈希。它从原回执重建 215 条计时记录和模型表，检查生成文件是否过期。去掉 `--check` 可重新生成结果文件。

第二条命令验证此前保留的源码与包内算子 AST 对应关系。两项检查均不能替代 GPU 数值、完整模型或长训质量验收；历史 GPU 结果的适用范围见 [KERNEL_RESULTS.md](KERNEL_RESULTS.md) 与 [VALIDATION.md](VALIDATION.md)。
