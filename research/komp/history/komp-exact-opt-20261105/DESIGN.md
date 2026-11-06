# Exact per-token KOMP：FP32 与 chunk 反向

## 参考的成熟实现

- FLA `fla/ops/gated_delta_rule/chunk.py`：WY 表示，chunk 边界状态扫描；反向区分 boundary adjoint 与 chunk 内局部梯度；重算中间量。参考副本位于 `../komp-gdn-8k-20261105/fla-source/`，commit 与逐文件哈希见此前验收记录。
- FLA `fla/ops/common/chunk_delta_h.py`：FP32 寄存器状态、按 value tile 分工，边界只按 chunk 写回。
- MesaNet `chunk.py`：反向重算 hidden states，分别计算跨块状态梯度与局部量；现有阅读副本位于 `../mesanet-komp-research-20261105/upstream-chunk.py`。

没有把 MesaNet 的有限次 CG 近似移植为 exact KOMP 求解器。KOMP 仍用直接消元，且每个 token 插入、压缩字典；计算块长度只影响执行方式。

## 持久状态与精度

默认投影、chunk 状态和累加均为 FP32，矩阵乘使用 tf32x3。q/k/v 支持 FP32/BF16。M32、K=V64 时，keys+coefficients 共 4096 个 FP32 元素/head，与 GDN 的 64×64 状态一致；learned 的 score/birth 元数据另计。`projection_dtype='float64'` 保留为数值对照，状态仍为 FP32。

直接把原 Gram 算法转为 FP32 在近重复 keys 下出现梯度误差。最终版本使用单位向量 Gaussian kernel 的等价式 `exp(-||a-b||²/(2τ))`；Gram 的距离在字典首向量为中心的坐标下计算，减少点积相减的消去误差。归一化 clamp 产生的零/小范数向量另保留原 dot-kernel 的范数修正。消元同时求解右端向量，避免显式 inverse 乘近全一向量。反向也用向量差和中心坐标，避免巨大径向项相消。

## 前向和反向

原实现保存每 token 的 `M×V` 衰减状态、`M×M` Gram 及 inverse。新实现不生成这些全序列 tape。

每块状态为 `S_exit = F S_entry + Fv V`。原 rank-two 低秩更新写作 `Z=A S_entry+B V`。把逐 token 读出也收缩为块矩阵后：

`O = Qmap S_entry + Omap V`。

因此前向只需边界状态与小矩阵乘法。反向分为：

1. 每块并行计算 `J=Qmapᵀ dO`。
2. 只在边界反扫 `dS_entry=Fᵀ dS_exit+J`，共 ceil(T/C) 步。
3. 各块并行求 `di/dc` 两组逐 token 低秩 adjoint，块内最多 C 步。
4. 按 chunk/slot/value tile 并行，从边界和低秩更新重建需要的 `X`、`dS`，在寄存器中完成局部梯度，不写 `T×M×V` tape。
5. 重算每 token 的 Gram 和消元，计算投影梯度，不保存 `T×M²` tape。

零 beta 的梯度直接计算 residual 的内积，不通过除 beta 恢复 residual。标量遗忘门的 chunk 内交互矩阵走矩阵乘法；逐 slot 遗忘门保留稳定的相对指数路径。FIFO 索引用闭式公式并行生成，learned 预取一批 scores，只在实际接受新原子时重算最低分/最老 slot；连续拒绝区间共享字典，一次批量写出索引。保持原有新者优先的同分规则与物理 slot 语义。

保存空间变为线性 token 几何/低秩更新、chunk 边界与块矩阵，约 `O(BH[T(K+M+V)+(T/C)(M²+MV)])`，无逐 token 的二次状态 tape。仍有 chunk 边界串行扫描、learned 选择扫描及 FP32 原子梯度累加，尚不等于完全沿时间并行或确定性位级复现。
