# 逐 token learned KOMP：推导、实现与数值边界

日期：2026-11-05。结论：已有实验的 learned 规则也可以每 token 压缩，并复用 rank-2 块内三角求解。已有可微原型；CPU 234 项验收通过。选择与槽位分配预处理仍逐 token 执行，GPU 吞吐尚未测量。

## 适用的 learned 定义

本实验的 score 为当前层输入的线性投影 `s_t = W_score x_t + b_score`，每个 head 一个标量。原子插入时记录 score，之后不根据状态重新打分。每步在旧 M 个原子与新原子中保留 score 最大的 M 个；同分优先保留较新的原子。可选写入门控为 `beta_eff = beta * 2 sigmoid(s_t)`，默认开启，与先前实验一致。

这些 score 不依赖本层当前递归系数 S；因此字典选择轨迹可以先算出，之后并行计算投影与块内系数更新。若改为根据当前 S、残差或实际函数删除误差决定淘汰，不能直接使用这一提前求几何量的方法。

当 score 在插入后固定，保留的原子集合就是初始原子与此前输入的 prefix top-M（按 score、出生时间排序）。集合上的 top-M 合并有结合性，可以研究并行 prefix top-M。不过，已有逐槽衰减绑定的是物理槽，物理槽分配取决于淘汰历史；只算集合并按 score 重新排序会改变模型。当前代码用与系数无关的时间预处理循环保存精确槽位和 tie-break；不声称已完全并行化该阶段。

## 两种压缩分支

令 `Sbar = D_t S_(t-1)`，`h_i = kappa(old_key_i,k_t)`，`r = beta_eff*(v_t^T-h^T Sbar)`。

### 删除一个旧原子

设 e 指向被删槽，新 key 接管此槽；p 是旧 key 在新 M 原子字典上的正则化投影。更新为

\[
S_t=\bar S+(p-e)e^T\bar S+e r.
\]

这与 FIFO 的 rank-2 公式完全一致，只是被删槽不再固定为 `t mod M`。

### 删除刚加入的新原子

旧 key 全部保留。设

\[
p=(K_{old,old}+\lambda I)^{-1}\kappa(D_{old},k_t).
\]

此时仍需转移新原子的残差系数：

\[
S_t=\bar S+p r
=(I-\beta_{eff}p h^T)D_tS_{t-1}+\beta_{eff}p v_t^T.
\]

所以“新原子没保留”不等于“忽略这个 token”。它仍参与当前输出，其写入也通过投影留在旧字典里。这个分支是 rank-1。

两种分支都可以用统一的两个更新量表示，复用 FIFO 原型的 `(I-L)Z=R` 求解。rank-1 分支把第一个修正设为零，第二个 U 列取 p。

## 自动求导与流式状态

- 硬 top-M 和同分排序不传梯度；这与原实现一致。
- score 通过 `2 sigmoid(score)` 写入门控取得梯度。
- q/k/v、衰减、基础 beta、初始 key 和系数保持可微；投影对 key 的梯度保留。
- 流式调用除了 key/系数，还必须携带每个物理槽的 score 和出生时间，并传入下一段的 start_position。否则同分优先顺序或槽位会改变。

入口：`learned_c1_komp.py::parallel_learned`。返回 output、(keys,coefficients) 以及 scores/births/dropped_slots 元数据。算法压缩间隔始终为 1；compute_chunk 仅控制三角求解的计算块长。

## FP32 近重复 key 问题与处理

第一版全部使用 FP32 的近重复 key 用例未通过，原始失败保留在 `learned-validation-failure.json`：

| 指标：新实现对旧 FP32 参考 | 最大绝对差 |
|---|---:|
| 输出 | 4.0654e-5 |
| 最终系数状态 | 0.00815845 |
| 保留 key | 0 |

最终状态的归一化最大误差为 0.00613387（0.613%），超过原定 0.003 门槛。归一化分母是 `max(1,max(abs(reference)))`，不是逐元素相对误差。

静态核对选择/物理槽/读出时机后，以 FP64 原参考诊断：

- 旧 FP32 参考对 FP64 的状态最大绝对误差：0.00448622。
- 第一版新 FP32 实现对 FP64 的状态最大绝对误差：0.00367222。
- Gram 最大条件数：39823。
- 批量和单矩阵 Gram 构造最大差：2.3842e-7；对应一次投影求解的最大差达 7.4072e-4。

因此旧 FP32 实现本身也受到近奇异 Gram 的舍入放大影响，不能作为此类情况下的精确 oracle。诊断记录在 `learned-fp32-diagnosis.json`。

本原型采用可明确复现的精度基准：**投影的 key normalization、Gram 构造与 solve 用 FP64；系数更新、块内 solve、读出仍可用 FP32。** 只把已舍入的 FP32 Gram 转成 FP64 不够，所以提高精度覆盖整个投影几何计算。默认 `projection_dtype=torch.float64`，保留 FP32 选项供后续性能/数值研究。

验收改用原冻结实现的 FP64 运算作为 oracle，未放宽 0.003 的 FP32 门槛。修正后，原失败用例（compute_chunk=1）的输出误差为 5.4602e-7、最终状态误差为 1.59495e-7。此结果**不是全 FP32 内核通过验收**，也不是低精度问题已经零成本解决。

## 验收结果

运行 `python3 validate_learned.py`。基准是旧冻结 `block_komp_token_reference(...,chunk_size=1,evict='learned')`。

| 检查组 | 数量 | 最大归一化误差 |
|---|---:|---:|
| FP64 输出、状态、字典 | 84 | 3.7994e-12 |
| FP32 系数计算＋FP64 投影，对 FP64 oracle | 84 | 5.4602e-7 |
| 八类梯度，包括 score 门控 | 48 | 7.4134e-12 |
| 流式、因果、多 batch、无额外 score 门控 | 18 | 详见 JSON |

共 234 项通过，结果见 `learned-validation.json`。覆盖只删旧原子、只删新原子、全同分、混合同分、随机 score、近重复 key、M32；计算块长 1/4/16/32。未来后缀干预下，两组前缀输出差均严格为 0。

共享 rank-2 求解器重构后，FIFO 原有 284 项也重新通过。所有验证目前均为 CPU；没有新增完整训练结果或 GPU 加速声明。下一步工程问题包括选择/槽位预处理、批量投影的稳定低精度实现、三角求解融合。
