# 第二轮 Exact KOMP 优化

## 问题与比较基准

上一轮 BF16 8K C16 的 FIFO 前反向约 3.97 ms。已保存 profiler 指向：geometry 约 2.125 ms，边界扫描约 1.198 ms，局部重建与梯度约 0.479 ms。扫描只是其中一项；消除扫描也无法接近 GDN，因此本轮同时考察减少计算与增加并行度。

`baseline_komp/` 固定为上一轮已验收的 FP32 优化版本，而不是最初的 FP64 实现。`compare.py` 在同卡重新测量该基准、新版与 FLA GDN；算法字典压缩仍为 C=1。

## 减少投影计算

先实验无 pivot 的 LDLᵀ 分解和两次三角求解。Gaussian Gram 加正 ridge 为 SPD；前向解 `Ap=c`，反向解 `Az=dp`。实测单 RHS 的 LDLᵀ 更慢，因此最终 dense 路径保留直接 RHS symmetric sweep：它仍采用 sweep 的矩阵更新方式，但只返回解，反向同样直接求解，不再计算 inverse 后乘 dp。稀疏 learned 多 RHS 路径采用 LDLᵀ 分解与三角求解。不能把 dense 路径表述为已经改成 Cholesky/LDLᵀ。前向与反向维持稳定的 FP32 距离 kernel 及原归一化 clamp 语义。反向读取已保存的 H/R/N，而不是再计算读取 kernel。

learned 选择已经产生完整物理 slot 轨迹。相邻拒绝 token 的保留字典相同；利用这点，在固定 32-token 工作窗口内按字典变化分段，一次 LDL 分解处理该段的多个 RHS。反向同样按版本重算分解，批量求 z，合并 `z pᵀ+p zᵀ` 后只做一次字典 Gram 梯度传播。拒绝 token 的 residual 投影仍全部计算。

为避免动态 host count、CUDA Graph 不兼容或分配最坏 T×M² factor tape，每个窗口至多 4 次接受时走版本复用；更密集的窗口走每 token 并行求解。跨窗口的同一字典可能再次分解，这是有界工作区与调度并行性的明确取舍。M>32 与 FP64 诊断路径仍走独立 token 求解。新增常驻量只有线性的 cross RHS。

## 局部状态重建

每 slot 的重建由已知低秩驱动组成：`S_t=α_t S_{t-1}+u0_t z0_t+u1_t z1_t`。对仿射对 `(α,b)` 作结合扫描，组合规则为 `(a₂a₁,b₂+a₂b₁)`，即可重建前向状态。反向应用相同组合到反向序列，恢复 adjoint。没有除累计 decay；强遗忘时避免 inverse decay 溢出。

这替换了原来的 C×2C 与 C×C 稠密三角矩阵乘，减少块长增长时的重复计算。Triton 的并行 associative scan 采用树形计算，工作/通信结构不同于线性递推；这里不把它表述为无成本 O(C) GPU 操作。

## 边界扫描的独立候选

1. `S'=E S+L Z`：复用本来必须计算的 Z，把原四次矩阵乘变成三次加缩放。作为 `reuse_z` 独立消融，保留串行边界扫描。
2. 分层 affine scan：先并行求各组的仿射映射与 source；串行扫组间边界；并行恢复组内 chunk 边界。Z 在边界确定后完全按 chunk 并行计算。增加计算/临时存储以缩短串行路径，不与第一项混称为减少总算量。

分层扫描 NC>=32 时启用，短序列保留原路径。组长通过独立 8/16/32 实验比较，输出、梯度、尾部和因果性必须验证。

## 其他候选与精度口径

- 严格按 token 下三角的 coupling 满足 L^C=0。删除平方展开的最后一轮，只覆盖到 L^(C-1)，少两次零矩阵乘。
- warp/列索引布局实验保存在 geometry-tuning 和 geometry-gather-tuning。它们先于用户关于减少重复计算的补充开始，不能将其优化顺序描述为主攻方向。不同 M/D 前后向选择可能不同，按实测判断。
- 混合精度作为独立 `mixed_state` 候选：geometry FP32；状态矩阵乘操作数 BF16，累加 FP32。必须报告偏差；未通过精度门槛则不进入默认实现。此实验不是把全部数值路径与 GDN 等同。

## 源码依据与验收

参考已固定的 FLA `gated_delta_rule/chunk.py`、`common/chunk_delta_h.py`、`utils/solve_tril.py`，以及 MesaNet 的前向/反向分离、重算与多 RHS 线性系统思路。本轮不把 MesaNet 的有限步 CG 当作 exact KOMP 解法。

- `validate_gpu.py`：冻结 FP64 逐 token 参考，包含病态近重复 keys、梯度与流式。
- `edge_acceptance.py`：零 beta、零/小 norm、强遗忘、FP64 诊断、8K 全梯度。
- `validate_scan.py`：非交换仿射矩阵独立参考、分组尾部、8K 严格前缀因果性。
- `validate_versions.py`：全拒绝/全替换/同分，以及窗口边界混合接受；验证输出、完整梯度与 metadata。
- `ablations.py`：独立关闭版本复用、分层扫描、局部扫描，额外比较 Z 复用和混合精度。
- `train_smoke.py`：FIFO/learned 的 AdamW 轨迹差分。

所有候选数字以 evidence/ 的实际运行记录为准；这份文档描述实现与实验设计，不预先宣称收益。

## 当前实测取舍

第一组 8K BF16 C16 消融（最终 dense RHS 布局优化之前）：learned 版本复用使完整前反向由 4.127 ms 降至 1.772 ms；去掉分层扫描变为 2.796 ms；保留旧 dense 局部重建变为 1.898 ms。各行仅改变对应开关，其它实现保持一致，不将多个候选的收益简单相加。

复用 Z 的串行路径为 3.308 ms，比同样其余路径下的不复用 Z 串行扫描 2.796 ms 更慢；当前不采用。实际 GPU 时间受访存和线程布局影响，矩阵乘次数减少不自动等于速度更快。

混合状态操作数实验在该输入下通过 1% 的对照门槛，但 learned 最大梯度 RMS 偏差约 0.528%，只使前反向从 1.772 ms 降至 1.723 ms。默认未引入这项精度取舍；这里只验证了 8K 随机输入，未宣称病态 key 或完整训练验证通过。

最终默认为 FP32 状态/geometry 与 tf32x3；dense M32/K64 RHS 求解采用实测更快的 2-warp 布局，其它尺寸保留 4-warp。最终速度以 RESULTS.md 与 comparison.json 为准。
