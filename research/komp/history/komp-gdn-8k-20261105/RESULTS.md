# Exact KOMP vs FLA GDN：8K 实测

当前 exact KOMP 在 8K 下明显慢于原生 FLA GDN。BF16 CUDA Graph 下，FIFO / learned 的前向加反向分别为 GDN 的 44.81 / 48.55 倍，训练峰值显存增量约为 15.4 倍。

## 条件与验收

H100 80GB HBM3；Torch 2.10.0a0+b558c986e8.nv25.11、Triton 3.7.1、CUDA 13.0。B=1、T=8192、H=2、K=V=64，KOMP M=32。两边 keys/coefficients 或 recurrent matrix 的持久状态各 4096 个 FP32 元素/head；learned 元数据另计。

共享原始 q/k/v、标量 FP32 log decay / beta；包括归一化、最终状态和所有输入梯度。KOMP 保留 FP64 投影，状态计算 FP32；GDN 使用原生精度。KOMP 计算块 16/32 不改变逐 token 字典压缩规则。没有模型投影层、卷积、优化器或 score 网络。

GDN FP32/BF16 输出与梯度共 16 项 naive 对拍通过，10 组 8K 用例输出与梯度均为有限值。KOMP 源码与此前通过 315 项验收的版本完全一致；FLA 595 个 Python 文件与此前成功比较的快照一致（commit 97874af2400916e2885c33ea30559a70ff686b66）。本次本地源码与 GPU 实际执行源码的 SHA256 全部一致。

## CUDA Graph：单次完整调用重放

预热、编译、autotune、capture 均不计时；每组 20 次，共 3 组取中位数。用于降低 Python dispatch 对比较的影响。显存单独测 eager 的峰值 allocated 增量，基线为常驻输入及上游梯度，不包括 graph pool。

| 输入精度 | 算子 | 计算块 | 前向 ms | 前向＋反向 ms | 前反向 / GDN | 训练增量峰值 MiB |
|---|---|---:|---:|---:|---:|---:|
| float32 | gdn | 64 | 0.3043 | 1.0385 | 1.00× | 60.44 |
| float32 | fifo | 16 | 4.1188 | 21.8658 | 21.05× | 482.88 |
| float32 | fifo | 32 | 4.4158 | 22.1394 | 21.32× | 482.88 |
| float32 | learned | 16 | 5.7588 | 23.5890 | 22.71× | 482.88 |
| float32 | learned | 32 | 6.0596 | 23.8406 | 22.96× | 482.88 |
| bfloat16 | gdn | 64 | 0.1385 | 0.4681 | 1.00× | 30.44 |
| bfloat16 | fifo | 16 | 4.1019 | 20.9739 | 44.81× | 468.88 |
| bfloat16 | fifo | 32 | 4.3929 | 21.2846 | 45.47× | 468.88 |
| bfloat16 | learned | 16 | 5.7444 | 22.7254 | 48.55× | 468.88 |
| bfloat16 | learned | 32 | 6.0408 | 23.0039 | 49.15× | 468.88 |

## Eager：包含 Python 调度开销

| 输入精度 | 算子 | 计算块 | 前向 ms | 前向＋反向 ms |
|---|---|---:|---:|---:|
| float32 | gdn | 64 | 0.7496 | 1.9970 |
| float32 | fifo | 16 | 4.1521 | 22.0224 |
| float32 | fifo | 32 | 4.4546 | 22.1843 |
| float32 | learned | 16 | 5.7902 | 23.7380 |
| float32 | learned | 32 | 6.0917 | 23.9128 |
| bfloat16 | gdn | 64 | 0.7418 | 2.0378 |
| bfloat16 | fifo | 16 | 4.1380 | 21.0363 |
| bfloat16 | fifo | 32 | 4.4367 | 21.3343 |
| bfloat16 | learned | 16 | 5.7800 | 22.7789 |
| bfloat16 | learned | 32 | 6.0763 | 23.0751 |

## 结论与边界

- 本次两种输入精度、两种淘汰规则下，KOMP 计算块 16 均快于 32。
- BF16 使 GDN 的 CUDA Graph 前反向从 1.0385 ms 降至 0.4681 ms；KOMP 仍保留 FP64 投影和 FP32 状态，因此收益有限。
- Eager BF16 前反向：GDN 2.038 ms、FIFO 21.036 ms、learned 22.779 ms；相差约 10.3–11.2 倍。GDN 本身执行短，Python 调度开销更显著，所以不能把 eager 倍数当纯 GPU 执行差异。
- 源码中仍存在每 token 的 FP64 Gram 求解、选择预处理、串行系数反向和大量状态存储。这些是后续优化方向；本次没有做 kernel profiler，不能据此分配各项耗时占比。
- 这是当前 correctness 实现的算子性能，不是优化后算法的速度上限，也不代表模型质量或端到端训练吞吐。

完整原始计时、三个 trial、显存、源码哈希及数值检查见 evidence/comparison.json；环境与任务回执同目录。测试作业 komp-gdn-8k-20261105-r1（job-f4b08b74e015-20261105141748）已完成并删除，精确 alias 复查为空；见 job-terminal.json、job-delete.json、job-absent.json。
