# 第二轮 Exact KOMP：最终验收与 8K 实测

本轮保留 FP32 geometry/状态与 tf32x3 计算，保持每 token 压缩和原有 learned 选择语义。基准是上一轮已优化实现（约 4 ms 前反向），不是最早的 FP64 版本。

## 同卡性能

H100 80GB，B1/T8192/H2/K=V64，KOMP M32。KOMP keys+coefficients 与 GDN recurrent matrix 各 4096 个 FP32 持久状态元素/head，learned metadata 另计。下表 CUDA Graph 排除编译、capture 和 Python 调度；3 组×20 次调用取中位数。显存是 eager 测得的峰值 allocated 增量，不含常驻输入、上游梯度或 graph pool。

| 输入精度 | 算子 | C | 前向 ms | 前向＋反向 ms | 训练显存增量 MiB |
|---|---|---:|---:|---:|---:|
| float32 | gdn | 64 | 0.3088 | 1.0499 | 60.44 |
| float32 | baseline_fifo | 16 | 1.6552 | 4.0238 | 98.50 |
| float32 | baseline_learned | 16 | 1.8089 | 4.1914 | 98.50 |
| float32 | fifo | 16 | 0.7188 | 2.1319 | 99.75 |
| float32 | fifo | 32 | 0.8401 | 2.2301 | 85.13 |
| float32 | learned | 16 | 0.6746 | 1.6767 | 101.75 |
| float32 | learned | 32 | 0.7952 | 1.8101 | 87.13 |
| bfloat16 | gdn | 64 | 0.1375 | 0.4657 | 30.44 |
| bfloat16 | baseline_fifo | 16 | 1.6263 | 3.9669 | 88.50 |
| bfloat16 | baseline_learned | 16 | 1.7800 | 4.1385 | 88.50 |
| bfloat16 | fifo | 16 | 0.7164 | 2.1205 | 89.75 |
| bfloat16 | fifo | 32 | 0.8375 | 2.2243 | 75.13 |
| bfloat16 | learned | 16 | 0.6734 | 1.6878 | 91.75 |
| bfloat16 | learned | 32 | 0.7939 | 1.8008 | 77.13 |

BF16 C16 与上一轮同卡基准比较：

- fifo：前向加速 2.27×，前反向加速 1.87×；当前前反向为 GDN 的 4.55×。显存增量 88.50 → 89.75 MiB。
- learned：前向加速 2.64×，前反向加速 2.45×；当前前反向为 GDN 的 3.62×。显存增量 88.50 → 91.75 MiB。

普通 eager（含 Python 调度）：

| 输入精度 | 算子 | C | 前向 ms | 前向＋反向 ms |
|---|---|---:|---:|---:|
| float32 | gdn | 64 | 1.0540 | 2.0856 |
| float32 | baseline_fifo | 16 | 1.6899 | 4.0910 |
| float32 | baseline_learned | 16 | 1.8410 | 4.2662 |
| float32 | fifo | 16 | 0.7599 | 2.2071 |
| float32 | fifo | 32 | 0.8809 | 2.2949 |
| float32 | learned | 16 | 0.7143 | 1.7690 |
| float32 | learned | 32 | 0.8354 | 1.8652 |
| bfloat16 | gdn | 64 | 0.7424 | 2.1523 |
| bfloat16 | baseline_fifo | 16 | 1.6604 | 4.0341 |
| bfloat16 | baseline_learned | 16 | 1.8124 | 4.1898 |
| bfloat16 | fifo | 16 | 0.7560 | 2.1902 |
| bfloat16 | fifo | 32 | 0.8775 | 2.2888 |
| bfloat16 | learned | 16 | 0.7131 | 1.7925 |
| bfloat16 | learned | 32 | 0.8337 | 1.8348 |

## 实现与减少的工作量

- learned：固定 32-token 窗口内按字典版本复用 LDLᵀ 分解，批量解 RHS；反向同样重算每版本 factor，并合并 Gram 梯度。频繁替换窗口、M>32 或 FP64 诊断模式用独立 token 路径。
- dense/FIFO：直接 RHS sweep 前后向；反向解 Az=dp，不再形成完整 inverse 后乘 dp。单 RHS LDLᵀ 实测慢，未无条件采用。M32/K64 的该路径选 2 warp，其余保留 4 warp。
- 读取保存的 H/R/N；局部状态与 adjoint 用稳定的仿射前后缀扫描，替代稠密三角矩阵重建。
- NC>=32 使用分层仿射边界扫描，Z 由独立 chunk kernel 并行计算；短序列保留原串行扫描。增加少量工作区以换取时间并行度。
- 移除 chunk maps 的最后一轮零幂矩阵乘。

本轮 learned 8K 输入共 392 次接受；495 个稀疏窗口、17 个密集窗口。每个前向或反向的分解次数由 16384 降为 1241，减少 92.43%。同一版本跨窗口仍会重算；并非无限制全局缓存。

## 消融与未采用候选

下表是最终 dense RHS 布局优化之前的独立候选，不能与上表混为同一版本。current 使用逐 token LDLᵀ 与多 RHS 版本复用；每一项按标记独立开关。

| 候选 | FIFO 前反向 ms | learned 前反向 ms |
|---|---:|---:|
| baseline | 3.9913 | 4.1317 |
| solve_only | 5.1030 | 5.2594 |
| current | 3.9675 | 1.7719 |
| no_versions | 3.9714 | 4.1272 |
| no_hierarchy | 4.9855 | 2.7959 |
| dense_local | 4.0865 | 1.8976 |
| reuse_z | 5.4855 | 3.3084 |
| mixed_state | 3.9114 | 1.7230 |
| hierarchy_only | 2.9490 | 3.1137 |

- solve_only：逐 token LDLᵀ、保存 H/R/N、去除零幂乘法；保留旧边界和局部重建。它比旧版慢，说明减少 FLOPs 不一定缩短 GPU 执行时间。
- no_versions / no_hierarchy / dense_local：分别关闭版本复用、分层扫描、局部仿射扫描。
- reuse_z：保留串行扫描，采用 S′=ES+LZ；本次比不复用 Z 的串行路径慢，未采用。
- mixed_state：只把状态矩阵乘操作数转 BF16，FP32 累加和 geometry 不变。只做了本组 8K 随机输入对照，未作为默认或发布接口。

mixed_state FIFO 的最大 RMS 偏差 0.00354336；实验门槛 1%，通过不等于完整精度/训练验收。
mixed_state learned 的最大 RMS 偏差 0.00528149；实验门槛 1%，通过不等于完整精度/训练验收。

## 正确性

最终版本 586 项检查通过（preflight/full/edge/scan/versions）。另外 GDN 16 项参考对拍通过。涵盖输出、最终状态、输入梯度、近重复 keys、M64/K128、零 beta、零/小范数、强遗忘、FP64 诊断模式、分组尾部、流式及 8K 前缀因果性。

完整 suite 对冻结 FP64 逐 token 参考的最大误差（含梯度）：

| 输入 | 最大 RMS 相对误差 | 最大归一化绝对误差 |
|---|---:|---:|
| torch.float32 | 0.0034119981 | 0.0031329211 |
| torch.bfloat16 | 0.0036061805 | 0.003622829 |

RMS 分母至少 1e-6；归一化绝对误差分母为 max(1,max(abs(reference)))。沿用原 suite 门槛，未为本轮实现放宽。

8K 对上一轮优化 kernel，torch.float32 最大 RMS 偏差 1.8563234e-06。
8K 对上一轮优化 kernel，torch.bfloat16 最大 RMS 偏差 3.896367e-05。
12 步 AdamW FIFO：loss 0.38645622 → 0.31559625；对 FP64 参考最大参数轨迹差 8.3684921e-05。
12 步 AdamW learned：loss 0.44374764 → 0.36476743；对 FP64 参考最大参数轨迹差 8.5681677e-08。

## 剩余成本与边界

- fifo C16 profiler：backward 0.919 ms；forward 0.512 ms；local_gradients_scan 0.289 ms；local_adjoint 0.076 ms；coarse 0.034 ms；maps 0.058 ms。
- learned C16 profiler：project 0.228 ms；local_gradients_scan 0.293 ms；read_backward 0.195 ms；select 0.158 ms；local_adjoint 0.076 ms；backward 0.072 ms。

Profiler 含仪器开销，独立于性能表计时。版本复用收益依赖真实接受率；FIFO 或始终接受的 learned 无法共享不同字典的分解。此轮没有声称与 GDN 相同表达能力或完整模型质量，也未进行 PhysicsLM 质量训练。

RESEARCH.md 记录设计、来源和取舍；compare.py、ablations.py 与 evidence/ 保存原始计时、精度、环境、源码 SHA256 和作业回执。编译索引类型问题与测速脚本的捕获冲突在 queue 日志中保留，未计作成功数值结果。

现有项目包已完成 GPU 导入与反向验收，源码哈希与实测版本一致。

GPU 测试作业已完成，精确 alias 删除后复查为空。
