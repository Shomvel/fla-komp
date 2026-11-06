# Exact KOMP 优化验收与 8K 性能

默认投影改为稳定 FP32，状态保持 FP32；移除每 token M×V 状态和 M×M Gram/inverse tape。系数反向拆成 chunk 边界反扫与块内并行梯度，learned 选择批量跳过拒绝区间。保留 `projection_dtype="float64"` 作为精度对照。

## 相同 H100 上的 8K 结果

B=1、T=8192、H=2、K=V=64，KOMP M=32。每 head 的 keys+coefficients 与 GDN 的 recurrent matrix 均为 4096 个 FP32 元素；learned metadata 另计。表中 CUDA Graph 排除编译、autotune、capture 和 Python 调度开销，峰值显存增量在 eager 路径独立测量。各自包含最终状态及全输入反向。

| 输入精度 | 算子 | 块长 | 前向 ms | 前向＋反向 ms | 训练显存增量 MiB |
|---|---|---:|---:|---:|---:|
| float32 | gdn | 64 | 0.3072 | 1.0473 | 60.44 |
| float32 | baseline_fifo | 16 | 4.1150 | 21.8756 | 482.88 |
| float32 | baseline_learned | 16 | 5.7560 | 23.6246 | 482.88 |
| float32 | fifo | 16 | 1.6395 | 3.9956 | 98.50 |
| float32 | fifo | 32 | 1.7522 | 4.2179 | 84.66 |
| float32 | learned | 16 | 1.7938 | 4.1703 | 98.50 |
| float32 | learned | 32 | 1.9064 | 4.3930 | 84.66 |
| bfloat16 | gdn | 64 | 0.1379 | 0.4621 | 30.44 |
| bfloat16 | baseline_fifo | 16 | 4.0977 | 20.9525 | 468.88 |
| bfloat16 | baseline_learned | 16 | 5.7431 | 22.7099 | 468.88 |
| bfloat16 | fifo | 16 | 1.6251 | 3.9661 | 88.50 |
| bfloat16 | fifo | 32 | 1.7326 | 4.1915 | 74.50 |
| bfloat16 | learned | 16 | 1.7837 | 4.1317 | 88.50 |
| bfloat16 | learned | 32 | 1.8871 | 4.3538 | 74.50 |

同一轮 BF16、块长 16 对比：

- fifo：前反向加速 5.28×，显存增量降低 81.1%；当前前反向仍是 GDN 的 8.58×。
- learned：前反向加速 5.50×，显存增量降低 81.1%；当前前反向仍是 GDN 的 8.94×。

普通 eager 耗时（含 Python 调度）另列：

| 输入精度 | 算子 | 块长 | 前向 ms | 前向＋反向 ms |
|---|---|---:|---:|---:|
| float32 | gdn | 64 | 0.7532 | 2.0264 |
| float32 | baseline_fifo | 16 | 4.1579 | 21.7080 |
| float32 | baseline_learned | 16 | 5.8313 | 23.4468 |
| float32 | fifo | 16 | 1.6750 | 4.0653 |
| float32 | fifo | 32 | 1.7899 | 4.2807 |
| float32 | learned | 16 | 1.8289 | 4.2364 |
| float32 | learned | 32 | 1.9411 | 4.4529 |
| bfloat16 | gdn | 64 | 0.7431 | 2.2778 |
| bfloat16 | baseline_fifo | 16 | 4.1382 | 21.0285 |
| bfloat16 | baseline_learned | 16 | 5.7877 | 22.7629 |
| bfloat16 | fifo | 16 | 1.6637 | 4.0258 |
| bfloat16 | fifo | 32 | 1.7690 | 4.2640 |
| bfloat16 | learned | 16 | 1.8158 | 4.1921 |
| bfloat16 | learned | 32 | 1.9182 | 4.4252 |

## 精度与训练

最终版本：469 项输出/状态/梯度/流式/因果性检查通过；另有 20 组 selector 的完整索引、slot、分数、birth 轨迹与旧版逐位一致，包含 8K。GDN 本轮参考对拍另计 16 项。

FP64 冻结逐 token 参考的 full suite（最大值包含输出、最终状态和输入梯度）：

| 输入精度 | 最大 RMS 相对误差 | 最大归一化绝对误差 |
|---|---:|---:|
| torch.float32 | 0.0034119109 | 0.0031328095 |
| torch.bfloat16 | 0.0036104654 | 0.003622829 |

相对误差分母为参考 RMS，至少 1e-6；归一化绝对误差分母为 max(1,max(abs(reference)))。保留既有 FP32 0.5% / BF16 2.5% RMS 检查门槛，没有因初版失败放宽。最难用例为近重复 key 的字典梯度。原始 FP32 直接翻译失败证据保存在 fp32-naive-full-failure.json；最终稳定距离形式、直接 RHS 消元与差分梯度通过。

8K 与旧版 FP64 geometry kernel 的全部输出、状态和梯度对拍：

- torch.float32 最大 RMS 相对偏差：1.8992065e-06。
- torch.bfloat16 最大 RMS 相对偏差：9.8506535e-05。

优化器级别：FIFO / learned 各 12 步 AdamW，真实 GDN-slot 参数化与 FP64 逐 token 参考对拍；两者 loss 均下降。

- FIFO：loss 0.38645622 → 0.31559625；最大参数轨迹差 8.3684921e-05。
- learned：loss 0.44374764 → 0.36476746；最大参数轨迹差 2.6077032e-07。

## 剩余瓶颈

以下是 warmup 后 profiler 的 kernel attribution，每个算子执行 3 次，独立于上面的计时：

- fifo C16：backward 1.229 ms；forward 0.895 ms；state_scan 0.626 ms；boundary_scan 0.572 ms；local_gradients 0.403 ms。
- learned C16：backward 1.248 ms；forward 0.891 ms；state_scan 0.625 ms；boundary_scan 0.573 ms；local_gradients 0.400 ms。

仍有 chunk 边界扫描、直接 Gram 求解和局部重算成本；在 B1/H2 下不能充分利用所有 SM。当前不是与 GDN 等速的实现，也未做完整模型质量训练。learned 的批量跳过优化适用于存在拒绝区间；全替换时仍需依次更新物理 slot。

## 证据与复现

- DESIGN.md：GDN/Mesa 源码依据、推导及状态存储说明。
- compare.py：3 组 × 20 次 CUDA events，中位数；同卡测旧/新/GDN。
- evidence/comparison.json：全部 trial、显存、运行源码 SHA256。
- evidence/kernel-profile.json：各 kernel 耗时。
- evidence/environment.json：实际 Torch/Triton/CUDA/H100 环境。
- job.yaml / runner.sh：WeaverRSI/proj-32，固定镜像 digest，独立任务目录。
- evidence/candidate1-*.json：第一次优化后的测量，最终结果未混用。

现有项目包 `komp_slot.exact_komp` 已完成 GPU 导入、输出/状态和反向验收；本地包与实测源码哈希一致。
测试作业已记录 Completed 终态，按精确 alias 删除并确认消失；回执保存在 evidence/。
