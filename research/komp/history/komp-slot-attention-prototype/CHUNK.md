# Rank-2 chunk recurrence

本页推导仍适用于默认全 Triton 后端；下方混合实现的细节对应保留的 `backend='chunk'`。最新 kernel 拆分与反向算法见 [ALL_TRITON.md](ALL_TRITON.md)。

使用与 recurrent 版本完全相同的转移，给定已准备好的 `h,p,r`、标量门和整数路由。`S_t` 是 `[M,Dv]` value 系数状态，`e_t` 是被替换槽位的 one-hot。定义

```text
X_t = [p_t-e_t, -beta_t e_t]       # [M,2]
Y_t = [e_t, h_t]                  # [M,2]
b_t = beta_t e_t                  # [M]
S_t = alpha_t S_{t-1} + X_t z_t + b_t v_t^T
z_t = Y_t^T alpha_t S_{t-1}       # [2,Dv]
o_t^T = r_t^T S_t
```

块内时间下标 `t,s = 0,...,C-1`，入口状态记作 `S_in`。令

```text
gamma_t = product_{j=0}^t alpha_j
d_ts = product_{j=s+1}^t alpha_j    # d_tt=1; only used when s<=t
L_ts = 1[s<t] d_ts Y_t^T X_s       # [2,2] block
A_t  = gamma_t Y_t^T               # [2,M] block row
B_ts = 1[s<t] d_ts Y_t^T b_s       # [2,1] block
Z = (I-L)^-1 (A S_in + B V)
```

`V` 在本页仅表示当前块的输入 value 矩阵 `[C,Dv]`，不表示系数状态。`L` 为 `[2C,2C]` 严格块下三角矩阵；同一 token 的两列之间没有反馈依赖。代码以不含除法的 cumprod 计算 `d_ts`，因此 `alpha=0` 及其梯度有定义。不能用 `gamma_t/gamma_s` 替代。

进一步令 `Z_S=(I-L)^-1 A`、`Z_V=(I-L)^-1 B`，则 `Z=Z_S S_in+Z_V V`。把它代回每个 token 的读出和块末状态，可以预先计算四个 transfer map：

```text
O     = OS S_in + OV V           # [C,Dv]
S_out = FS S_in + FV V           # [M,Dv]
```

所有块的四个 map 可以并行准备；只有上述 `S_in -> S_out` 在块间递推。Triton scan 内循环次数为 `ceil(T/C)`，没有沿 token 的 value-state 循环。块内读出也是矩阵乘法。

`chunk_size` 支持 16/32。尾块把无效位置设为 `alpha=1,beta=0,p=e,h=r=v=0`，转移为恒等映射，截去无效输出。

## 实现边界

- `chunk_coefficients`：PyTorch 批量矩阵乘法构造 `L` 和四个 map；Triton 计算 `(I-L)^-1`。逆矩阵使用分治合并：`[[A,0],[-B,D]]` 的逆矩阵左下块为 `D^-1 B A^-1`。从对角块逐级合并，避免显式构造 `L` 的高次幂。
- `chunk_fwd_kernel` / `chunk_bwd_kernel`：Triton 块间 scan、块内输出及解析反向。PyTorch autograd 继续传播四个 map 的梯度；逆矩阵反向为 `inv.T @ grad @ inv.T` 的严格下三角部分。
- scan 的状态 tape 从每 token 一份降为每块边界一份，空间为 `O(B H ceil(T/C) M Dv)`。系数构造仍保存 `O(B H ceil(T/C) (C²+CM+M²))` 级别中间量，反向 partial buffer 还随 value tile 数增长；这并不保证整个算子的峰值内存更低。
- key 轨迹、核 Gram 求解依然是原有 PyTorch 几何阶段。当前实现没有加速这部分，也没有把四个 map 的构造全部融合进 Triton。
- 路由 `slots` 必须事先给定。若要按当前 value 状态动态选择最小 KOMP 删除代价，不能直接将整段路由作为独立输入预计算；该算法不在此原型范围。
- 代数上与逐 token 转移等价；浮点运算顺序不同。分块求逆及状态转移的数值稳定性取决于实际系数，测试通过不代表任意长度和任意核病态条件下都稳定。

调用高层入口时指定 `backend='chunk', chunk_size=16`。`backend='chunk_torch'` 是同一块算法的 PyTorch 版本；独立数值 oracle 仍是 `recurrent_reference` 的 float64 稠密逐 token 转移。
