# MesaNet 并行机制与逐 token FIFO-KOMP 原型

日期：2026-11-05。结论：FIFO-KOMP 可以保持每 token 插入、每 token 压缩的语义，并用 rank-2 块内三角求解计算。已实现可微 PyTorch 原型并与旧冻结 C=1 参考实现完成 CPU 数值对拍。GPU 速度、完整训练质量和融合内核尚未验证。

后续补充：已有 learned 规则也已实现并完成 234 项验收，见 [LEARNED.md](LEARNED.md)。其中记录了近重复 key 的 FP32 失败、FP64 诊断，以及 FP32 系数计算＋FP64 投影的精度基准；没有将它描述成全 FP32 验收通过。

## 1. MesaNet 实际做了什么

论文：*MesaNet: Sequence Modeling by Locally Optimal Test-Time Training*，arXiv:2506.05233，阅读版本 v1。主要依据第 2 节、附录 B/C/G/I。

- https://arxiv.org/abs/2506.05233
- https://arxiv.org/html/2506.05233v1

官方 FLA 实现固定到 commit `9e82a80fdaefcd7c9cc3da51ea36d6a9e1587bd1`；下载的代码保存在本目录 `upstream-*.py`。这是用于此次研究的源码版本，不是对先前实验版本的替换。

MesaNet 每个 head 保存两份充分统计量。采用列向量表示：

\[
H_t=\gamma_t H_{t-1}+\beta_t k_tk_t^T,
\quad G_t=\gamma_t G_{t-1}+\beta_t v_tk_t^T.
\]

读出为

\[
(H_t+\Lambda)q_t^*=q_t,\qquad o_t=G_tq_t^*.
\]

FLA 源码的 `h_kv` 存储的是上述 G 的转置；公式与代码的方向差异不改变算法。

它不维护有限个真实 key 原子的字典，也不进行 KOMP 式删除投影。相似性在于正则化 Gram 系统求解和记忆读出，而非完全相同的状态压缩规则。

并行分成两部分：

1. 用 gated linear attention 的块状态更新得到每个计算块入口的 H/G。
2. 每个位置独立求自己的 q*。CG 每一轮所需的 `(H_t+Λ)p_t` 可以展开为“块入口矩阵乘法＋块内因果加权的 key 点积”，合并成矩阵乘法，无需为每个位置保存完整 H_t。

块内令 `a_t=prod(gamma_1...gamma_t)`，`W_ts=1[s<=t]*beta_s*prod(gamma_(s+1)...gamma_t)`，把每个位置的方向 p_t 放在矩阵 P 的一行，则一次算子应用为

\[
\mathcal A(P)=\operatorname{diag}(a)P H_{in}+
[(P K^T)\odot W]K+P\Lambda.
\]

替换最后一个 K 为 V 并使用 `G_in^T`，即可对求出的 Q* 批量读出。每个位置仍使用自己的真实因果前缀，不是整块共用冻结的最优解。

源码证据：

- `upstream-naive.py`：精确逐位置 solve、CG 参考与分块公式。
- `upstream-chunk_h_fwd.py`：`for i_t in range(NT)` 跨块传播 H/G。实际实现保留跨块循环，不能描述成整个时间轴完全无串行。
- `upstream-chunk_cg_solver_fwd.py`：`grid=(NT,H*B)`；`chunk_update_once` 将上述算子写成 `tl.dot`；CG 的迭代轮数依次执行，位置/块并行。
- `upstream-chunk.py`：计算块大小 64，默认 30 次 CG；反向再次用 CG 解伴随系统。该反向按隐式求导构造，不是对有限次 CG 轨迹逐步自动求导。论文附录 I 也明确指出了这个近似前提。

论文以固定正则项和直接求解替代旧 Mesa 的 Sherman–Morrison 逆状态递推。旧递推在强遗忘下会有数值问题，且初始化引入的正则项也会衰减。CG 本身不自动解决任意模型的递归依赖，关键是 H/G 的更新不依赖上一步求解得到的 q*。

## 2. FIFO-KOMP 的精确 rank-2 重写

使用物理环形槽顺序，保留 M 个 key，系数矩阵为 `S [M,V]`。每一步将 `r=t mod M` 槽的旧 key 淘汰，把新 key 放到同一槽。其余 key 不变。令 e 为第 r 个单位向量。

定义：

- `D=diag(alpha_t)`：逐槽衰减，兼容已有 GDN-slot 公式。
- `h_i=kappa(old_key_i,k_t)`：新 key 对旧字典的查询向量。
- `p=(K_new,new+ridge*I)^(-1) k_new,drop`：被删旧 key 投影到新字典的系数。这里 new 指压缩后的 M 个 key，包含刚写入的 key。
- `beta`：当前输入生成的标量写入强度。

完整单步更新可直接写成：

\[
\bar S=D S_{t-1},\quad
c_{drop}=e^T\bar S,\quad
r_t=\beta(v_t^T-h^T\bar S),
\]

\[
S_t=\bar S+(p-e)c_{drop}+e r_t.
\]

第一项保留衰减后的状态，第二项移除旧槽并转移旧系数，第三项在新 key 上写入残差。因此

\[
S_t=(I+U_t V_t^T)D_t S_{t-1}+e_t\beta_tv_t^T,
\]

\[
U_t=[p_t-e_t,\ e_t],\qquad
V_t=[e_t,\ -\beta_t h_t].
\]

这不是近似。每个时刻的投影 p 在 FIFO 下只依赖输入 key 和固定 ridge，不依赖 S，因此所有时刻的 p 可以批量提前求解。本原型使用 `torch.linalg.solve`，没有用截断 CG 引入额外近似。

当前输出遵守旧参考的顺序：先对 M+1 字典读出，然后压缩，下一 token 使用压缩后的状态。令

\[
\rho_t=\kappa(q_t,D_{old})-\beta_t\kappa(q_t,k_t)h_t,
\]

则当前输出为

\[
o_t^T=\rho_t^T D_t S_{t-1}+\beta_t\kappa(q_t,k_t)v_t^T.
\]

注意不要把输出改成压缩后的 query 读出，否则会改变参考算法。

## 3. 每 token 压缩，同时成块计算

设计算块内有 B 个 token，定义两个更新量组成的 `z_t [2,V]`：

\[
z_t=V_t^T D_t S_{t-1}+b_t,
\quad b_t=\begin{bmatrix}0\\\beta_tv_t^T\end{bmatrix},
\quad S_t=D_t S_{t-1}+U_tz_t.
\]

用 `D_(s+1:t)=D_t...D_(s+1)` 表示衰减乘积，展开得到：

\[
z_t=V_t^T D_{1:t}S_{in}+b_t+
\sum_{s<t} V_t^T D_{s+1:t}U_s z_s.
\]

所以把所有 z 堆起来，可以一次解

\[
(I-L)Z=R,\qquad L_{ts}=\begin{cases}
V_t^T D_{s+1:t}U_s,&s<t,\\0,&s\ge t.
\end{cases}
\]

`I-L` 是 2B×2B 的单位下三角矩阵。解出 Z 后，可以批量计算每个位置的输出和块末状态，不必逐 token 运行状态更新 Python 循环。跨计算块仍传递 M×V 状态。

因此：

- 算法压缩间隔始终为 1，字典在每个 token 后都恢复到 M。
- `compute_chunk=1/4/16/32/128` 只控制执行组织，不改变淘汰/投影时机。
- 这不同于旧 Block-KOMP 的 C：旧 C 同时改变算法压缩间隔和字典大小。
- 三角求解内部存在数据依赖；数学上可以组织为块算法，但并不承诺现成 PyTorch 求解器已提供最佳 GPU 并行效率。

这是借鉴 MesaNet“把每步优化写成可批处理线性系统”的方向，并结合低秩状态更新得到的 KOMP 路线；不是把 MesaNet 的两份充分统计量直接套到 Gaussian 字典上。

## 4. 可运行实现与验收

- `c1_komp.py::sequential`：直接逐 token 的环形槽实现。
- `c1_komp.py::parallel`：rank-2 块内三角求解；支持 scalar/per-slot decay、计算块尾部、流式继续和自动求导。
- `validate.py`：与 `/Users/shom/Downloads/komp-accel-scratch/frozen-20261101-r3/komp_slot/block_reference.py::block_komp_reference(chunk_size=1)` 差分验收；不是只比较自己写的两个实现。
- `validation.json`：逐项误差和容差。

执行：

```sh
cd /Users/shom/Downloads/mesanet-komp-research-20261105
python3 validate.py
```

本次 CPU/PyTorch 2.12.1 结果：284 项通过。

| 检查组 | 数量 | 最大归一化绝对误差 |
|---|---:|---:|
| FP64 输出、状态、字典 | 119 | 9.86e-13 |
| FP32 输出、状态、字典 | 119 | 2.252e-4 |
| FP64 全部 7 类输入/初始状态梯度 | 42 | 3.373e-13 |
| 未来后缀干预的前缀输出 | 1 | 0 |
| 不对齐分段的流式继续 | 3 | 5.115e-16 |

归一化绝对误差定义为 `max(abs(actual-reference))/max(1,max(abs(reference)))`，不是 RMS 相对误差。用例覆盖 T=1/7/19/33/65、M=4/8/32、K/V64、近重复 key、强遗忘、scalar decay、非零初始系数、计算块 1/4/16/32/128。梯度覆盖 q/k/v/log_decay/beta/initial_keys/initial_coefficients。CPU 数值对拍不能替代 GPU 混合精度稳定性、吞吐和完整模型质量实验。

## 5. 仍需解决的问题

1. **每 token 的 Gram 求解成本。** 原型批量构建每个位置的 M×M Gram 并直接求解，时间成本约 O(T M^3)，存储约 O(T M^2)。M=32 是否划算需要实测。可以研究 batch Cholesky、Mesa 式 CG 或滑动 Gram 的结构更新；有限步 CG 会引入近似，必须单独验收。
2. **rank-2 三角内核。** 每计算块求解 2B×2B 系统；原型还显式构建 B×B×M 衰减因子。生产实现应融合构造、solve 和读出，控制寄存器及临时显存，不宜把本原型的中间张量布局直接搬到大规模训练。
3. **数值稳定性。** 原型使用因果方向的 `exp(prefix_t-prefix_s)`，避免负前缀指数取倒数溢出；仍需更长序列和真实训练中的低精度验收。
4. **淘汰策略范围。** 本文最初的证明和实现针对 FIFO；后续已扩展到插入时固定输入 score 的 learned（见 LEARNED.md）。如果选谁淘汰依赖当前系数或函数删除误差，就不能直接提前获得全部 U/V。
5. **质量尚未知。** 精确重现 C=1 不表示质量优于旧 C128。新算法每步有投影误差，必须重新训练比较；计算块不改语义与压缩频率不改质量是两件不同的事。

建议下一步先做 M32、GDN-slot、FIFO 的 GPU kernel 对拍及基准，再接回统一训练 harness 比较 C1 与既有 C128。训练和数据口径保持一致后，才能回答吞吐和生成质量是否值得这次改动。
