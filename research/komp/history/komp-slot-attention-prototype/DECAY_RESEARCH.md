# KOMP 的 scalar / slot decay 研究

2026-10-31。两个只读 subagent 分别核对 GDN scalar gate 和 GSA slot gate；本文合并公式、实际源码接口和建议的消融。这里没有实现新 GPU kernel，也没有新增 decay 训练结果。

## 结论

当前 KOMP 已有 GDN 式的 scalar decay-before-residual 转移。新增研究应分两步：先换成 GDN 的可学习时间尺度参数化，复用当前 recurrence；再引入作用在旧字典系数上的逐 slot pre-decay，改写块内系数构造。两者均保留输入条件下的仿射递推；逐 slot 版本不能直接沿用 scalar 系数 kernel。

现有 Depo/Brevo 研究中的 GDN 和 KOMP **都使用 sigmoid scalar gate**，没有使用上游完整 GatedDeltaNet 的 A_log/dt_bias 门。因此不能把已有结果解释成“没有 decay 的 KOMP 对比原生 GDN”。

## 统一符号与当前规则

省略 batch/head。旧字典上的系数为 \(S\in\mathbb R^{M\times d_v}\)，\(e\) 是待替换槽位的 one-hot；\(p\) 把被删 atom 投影到新字典，\(h=\kappa(K_{old},k_t)\) 是新 key 对旧字典的核读向量。定义

\[
R=I+(p-e)e^\top,\qquad C=R-\beta e h^\top.
\]

当前规则是

\[
\bar S=\alpha S,\quad \epsilon=v-\bar S^\top h,\quad
S_{new}=R\bar S+\beta e\epsilon^\top
=\alpha CS+\beta ev^\top.
\]

参考 `komp_slot/reference.py:39`、`physics_lm/experiment.py:44`。FLA GDN naive 同样先遗忘旧状态、再计算 residual：`fla/ops/gated_delta_rule/naive.py:50`。

scalar 可以穿过旧状态的线性变换 C，但不能移到整个更新之后，否则新写入也被衰减。若 residual 改读未衰减的 S，则额外改变 \(-\beta(1-\alpha)e h^\top S\)。当前 ridge=1e-4，也不要假设投影前后读值严格相同。

## GDN 风格 scalar 参数化

采用上游公式（`fla/ops/gated_delta_rule/gate.py:325`）：

\[
a_{t,h}=W_{a,h}x_t,\quad \lambda_h=\exp(A_{\log,h}),\quad
g_{t,h}=-\lambda_h\operatorname{softplus}(a_{t,h}+b_{\Delta,h}),\quad
\alpha_{t,h}=\exp(g_{t,h}).
\]

等价于 \(\alpha=\sigma(-a-b_\Delta)^\lambda\)。\(\lambda=1\) 时只是符号反转后的 sigmoid；学习 \(\lambda\) 才新增每 head 的时间尺度。

首轮初始化精确匹配当前 alpha=.98：

\[
A_\log=0,\quad W_a=0,\quad
b_\Delta=\operatorname{softplus}^{-1}(-\log .98)=-3.891820298.
\]

仅考虑这个 scalar 门，半衰期为34.31 tokens，32步保留0.524，128步保留0.075。这不包含字典替换及 delta 更新，不能当成完整模型记忆寿命。

原生 GDN 的随机 A/dt 初始化会同时改变时间尺度分布，应另做实验。上游排除 A_log/dt_bias 的 weight decay，但本项目 optimizer 不读取 `_no_weight_decay` 属性，必须显式分组；若改变这项策略，对应 sigmoid 对照也需采用一致策略并重跑。

Triton 第一版只新增 FP32 gate 生成及其 backward，继续传 `alpha[B,T,H]` 给现有 chunk。导数为

\[
\partial_a\alpha=-\alpha\lambda\sigma(a+b_\Delta),\qquad
\partial_{A_\log}\alpha=\alpha g.
\]

现有块 rank-2 结构和边界 scan 均不变。门不能提前量化成 BF16；也不需要再给状态重复乘一个 scalar。门若依赖未知 value 状态 S，而非输入或可预知的 key 字典，则一般不再能预先构造当前仿射块系数。

## GSA 风格逐 slot pre-decay

原生 GSA 使用 `g=logsigmoid(f_proj(x))/8`、`s=1-exp(g)`，同时衰减 key-slot 状态的列和 slot-value 状态的行（`fla/layers/gsa.py:197`，`fla/ops/gsa/naive.py:54,68`）。KOMP 的 key 字典是核中心，不能把这两套状态机械等同；首版只对旧字典的系数行做衰减：

\[
g_t=\log\sigma(W_fx_t+b_f)/\tau_g,\quad
a_t=\exp(g_t)\in(0,1]^M,\quad D_t=\operatorname{diag}(a_t),
\]
\[
\bar S_t=D_tS_{t-1},\quad \epsilon_t=v_t-\bar S_t^\top h_t,
\boxed{S_t=\bar S_t+(p_t-e_t)e_t^\top\bar S_t+\beta_te_t\epsilon_t^\top}.
\]

gate 属于替换前物理 slot 中的旧系数。被删槽位 j 的信息先乘 a_j，再按 p 分配；新写入当步不再衰减。下一步同一个物理 slot 可能承载新 atom，门下标不是 key 的持久身份。

顺序不能交换：

\[
RD-DR=[(a_j\mathbf1-a)\odot p]e^\top\ne0.
\]

保留独立 beta 是首轮推荐。`beta=1-a_j` 可作为写入耦合消融，但不等价于 GSA 的全 slot 写入；在当前初始值下它还会把 beta 从.2降到.02。

门均值仍匹配.98。若采用 /8，则零输入的匹配 bias 为 `logit(.98**8)`；各slot权重独立小随机初始化并检查实际alpha分布，不能重现旧GSA全零门的slot对称。这样零输入基准匹配，非零随机输入的均值需实际记录，不声称逐token精确相同。

## 精确 chunk 结构

令
\[
X_t=[p_t-e_t,-\beta_te_t],\quad Y_t=[e_t,h_t],\quad w_t=\beta_te_t.
\]
\[
A_t=(I+X_tY_t^\top)D_t=D_t+X_t(D_tY_t)^\top.
\]

仍是 diagonal + rank-2。令 \(P_{i:j}=\prod_{u=j}^iD_u\)，空乘积为 I，\(z_i=Y_i^\top D_iS_{i-1}\)。块内严格下三角项为

\[
L_{ij}=Y_i^\top P_{i:j+1}X_j,\quad
B_{ij}=Y_i^\top P_{i:j+1}w_j\quad(j<i),
\]
\[
Z=(I-L)^{-1}(Y_{prefix}S_0+BV),\quad
(Y_{prefix})_i=Y_i^\top P_{i:1}.
\]

输出与块终态仍能写成四个 map `OS/OV/FS/FV`，所以块间 scan 与边界 backward 接口可保持。独立 FP64 小例子检验的脚本和结果保存在 `decay_research/`；它验证公式，不代表GPU kernel通过验收。

## 必需代码改动与数值风险

| 位置 | 修改 |
|---|---|
| `operator.py` | 支持 `[B,T,H,M]` alpha，保留 scalar 快路径 |
| `utils.py:24` | 拆分 alpha/beta/slot 的长度 NA/NB/NS；现有共用NA会在alpha扩维后越界 |
| `reference.py:49` | 向量 gate 对 state 行广播，保留独立 FP64 oracle |
| `triton_chunk.py:34,48` | `[C,M]` gate；标量区间乘积改为逐slot加权内积，不能提出内积外 |
| `triton_chunk.py:140,154,169,174` | 重计算/backward向量化；da只归约value维，DA按slot存储，再归约value tiles |
| `physics_lm/experiment.py` | 新门模块、独立slot初始化、明确参数分组和实验variant |

核几何 h/p/r 本身不变。初版建议使用零门安全的乘积构造，分块计算pair contraction，避免物化整个 `[2C,2C,M]`。正gate可用log区间差，但将其拆成 `exp(G_i)*exp(-G_j)` 会有溢出风险，即使真正区间乘积不大。需要参考FLA的重标度，不能仅凭公式可分解就宣称数值安全。

逐slot门使普通GEMM利用更困难，不承诺提速。它也不会消除现有Gram/投影成本。即使0<a<=1，系数欧氏范数收缩也不保证RKHS函数范数收缩：非正交atom之间原有抵消可能被破坏。建议记录系数范数、输出范数和Gram条件数代理。

## 实施及消融顺序

1. scalar sigmoid现有基线、无decay、匹配.98的GDN power-sigmoid；只改变遗忘参数化。
2. 在同一种门参数化/初始衰减下比较scalar与slot，beta维持独立。
3. 单独比较GSA /8参数化和写入耦合；避免把时间尺度、粒度、beta同时改动。

每组沿用同一LR、两seed、10M唯一输入token、最后checkpoint评估。新增no-weight-decay策略时应重跑匹配对照。继续报告按Depo hop和Brevo图大小分桶的自由生成，不能只看CE。

实现验收覆盖：逐slot常数退化为scalar的输出/终态/全梯度；零门和尾块；重复路由；跨chunk流式；同时置换字典/系数/gate/route的slot置换一致性；FP64 finite difference da/dg；高层输入长度；单步锁定旧信息衰减而新写入不衰减的顺序。

推荐先做scalar时间尺度版本，然后实现slot pre-decay。两者在理论上可行，当前没有新增实验能证明它们改善质量或容量。
