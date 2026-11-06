# KAPA × Block-KOMP：因果局部写入与保护式压缩

结论：可以融合，并且保留 key-only 的块转移矩阵与 affine scan。最值得拆开验证的两个位置是“每次写入保护过去预测”和“块末压缩保护近期预测”。不要把整个 chunk 的 Gram inverse 直接用于当前写入方向：带 ridge 时它依赖未来 key，即使最终输出再加 causal mask 也不能消除泄露。

本研究只添加独立 FP64 PyTorch 数学原型，没有改动冻结 kernel、生产 Triton 或现有质量实验。原型覆盖 FIFO；learned 淘汰的额外缓存契约在下文明确说明，尚未实现或训练。

## 现有结果能支持的动机

7-seed 结果显示，当前 KAPA scalar 在 Depo 上较好，而 Block slot/gate 在 Brevo 上明显较好。但两者同时存在固定/动态字典、raw/normalized Gaussian、Gram ridge、写入方向等差异，不能将差异直接归因于 KAPA protection。融合只是可检验假设，尚无质量收益证据。

生产规则来源：

- `komp_slot/kapa_reference.py: protected_direction`：固定中心，P B(BᵀP B+λI)⁻¹e₀，保护相对衰减后状态的预测；不是标签 replay。
- 冻结 `block_reference.py: block_komp_token_reference`：块内新 key 写新原子，块末选择 M 个原子并投影被淘汰系数；keep 系数在原始投影中不收缩。
- KAPA 文献：Liu & Príncipe, *Kernel Affine Projection Algorithms* (2008), §3.2, [DOI](https://doi.org/10.1155/2008/784292)。下述局部 RKHS 方向是对 KAPA-2 的序列适配，并非原文完整增长字典算法。

## 路线 W：局部 RKHS KAPA 写入

使用 Block-KOMP 的 normalized Gaussian：κ(x,y)=exp((x̂ᵀŷ−1)/τ)，故 κ(x,x)=1。令 A_t=[k_t,k_(t−1),…,k_(t−p)]，开头缺少的历史直接省略。G_t=κ(A_t,A_t) 的尺寸最多(p+1)²。

计算

    c_t = (G_t + λI)⁻¹ ((1+λ)e₀)
    w_t = E_t c_t
    C⁻_t = D_t C_(t−1)
    e_t = v_t − h_tᵀ C⁻_t
    C_t = C⁻_t + β_t w_t e_tᵀ
    o_t = r_tᵀ C_t

h_t=κ(U_t,k_t)、r_t=κ(U_t,q_t)。E_t 将当前与过去 p 个原子的局部坐标映射到当前字典 U_t；每列只有一个1，多个源映射同一原子时需要 scatter-add。w_t 最多 p+1 个非零项，与原来的 fresh-slot one-hot write 相比，修正会同时落在过去原子上。

这里有意乘(1+λ)：**p=0 时 c_t=1，精确恢复原 Block-KOMP 的 β 写入**。去掉此因子才是未缩放的局部 KAPA-2 方向，p=0 会变成 β/(1+λ)。因此实验应标为 normalized local-KAPA，不能声称与当前有限字典、ρ>0 的 KAPA 完全相同。

其 RKHS 目标可写成，对局部修正 Δf=Σ_a κ(a,·)d_a：

    min  1/2 ||Δf||²_H
       + 1/(2λ) || Δf(A_t) − (1+λ)e₀ e_tᵀ ||²_F

在 Gram 正定时，一阶条件给出 d=(G_t+λI)⁻¹((1+λ)e₀e_tᵀ)。PSD/重复 keys 时正则解仍有限，但当前拟合与过去保护可能冲突：相同 key 不可能同时获得非零当前增量与零历史增量。β 在解后缩放；λ>0 的保护是软约束，不能写成严格不遗忘。

**这不是简单把现有 fixed-dictionary KAPA 的 w 搬过来。** 当字典包含所有局部 anchor，ρ=0 且 Gram 可逆时，G_U⁻¹κ(U,A)=E，BᵀG_U⁻¹B=κ(A,A)，可约化到局部表达。实际固定字典 KAPA 用ρ>0，且可能不含原始 anchor；这时等式一般不成立。选择局部 representer rule 是一个新的、明确的数学规则。

### 当前误差的收缩不等于整体状态稳定

对单位范数 keys，G 的对角为1。令 α_t=h_tᵀw_t=(1+λ)e₀ᵀG(G+λI)⁻¹e₀，则 0<α_t≤1：正性来自 G 的半正定性，上界来自 [(G+λI)⁻¹]₀₀≥1/(1+λ)。一次写入后的当前 key 残差恰好为 (1−β_tα_t)e_t，因此 β_t∈[0,2] 时该残差范数不增加。这包含当前 sigmoid β 再乘 2·sigmoid(score) 的门控范围；比较对象是本次衰减之后、写入之前的残差。

这个局部结论不控制其他 query、全部系数范数或多个块的组合。接近共线的 keys 会产生较大的正负系数，逐 slot 衰减与压缩又会改变状态几何。融合训练必须记录系数范数、局部 solve 条件数及压缩前后的 anchor 误差，不能以单步收缩代替长期稳定性验收。

### 因果性边界

当前方向只使用当前与过去 keys；把整个 chunk 的 U 作为存储容器可以，但当前 w_t 在未来原子坐标上必须为零。不能依赖“未来系数初始化为零”来挽救一个已依赖未来几何的 w_t。

反例验证采用现有 fixed-dictionary 形式，将一个未来原子加入 U，再计算 P=(κ(U,U)+ρI)⁻¹、B=κ(U,A)。仅改变这个未来原子就改变当前 κ(q,U)P B(BᵀP B+λI)⁻¹e₀。具体数值见验证结果；不是泛泛的风险猜测。

### 跨块 anchor 必须保存什么

FIFO 且 p≤M 时，最近 p 个真实 token key 已在 carried dictionary 中，可按来源索引定位；第一块不能将随机初始 centers 当成真实过去 token。

learned eviction 可能删除近期 key，不能按“上一块最后 p 个 slot”猜 anchor。两种合法设计：

1. 单独保存最近 p 个 raw keys、birth/source IDs；新块把缺失 anchor 加为零系数 support，字典最多 M+p+C。已有同一 source ID 则复用坐标；不能用浮点相等判断身份。
2. 强制保留最近 p 个 key，其余 M−p 按 score 选。无需额外 support，但改变了淘汰策略，应是单独消融。

推荐先用第一种以保留 learned 排名语义。slot decay 时 inherited atoms 使用其实际 physical slot，缺失 anchor support 使用明确记录的临时 slot/birth 路由；必须保存映射。普通 FIFO 原型按 birth position mod M，与冻结 baseline 一致。

## 路线 C：KAPA 风格的块末保护式压缩

这条路线不改变块内 fresh-slot 写入，只改变原子淘汰后的投影。设 U 为当前块结束时的全部字典，K 为选出的 M 个保留原子，H 为最近至多 p 个已见 raw key；首块不足 p 个时只保护真实 token，不把初始化 centers 当历史。

原始 Block-KOMP 映射为

    J₀ = I_keep + (G_K + ρI)⁻¹ κ(K,E) I_drop
    C_K = J₀ C_U

这里 I_keep/I_drop 是坐标选择矩阵，保留系数原样传递、仅投影被淘汰部分。令

    P_K = (G_K + ρI)⁻¹
    A   = κ(H,K)
    R   = κ(H,U) − A J₀
    J   = J₀ + P_K Aᵀ (A P_K Aᵀ + ηI)⁻¹ R

于是 C_K=J C_U。它最小化相对原始压缩结果的修正代价

    min_Δ  1/2 tr(Δᵀ P_K⁻¹ Δ)
          + 1/(2η) ||AΔ − R C_U||²_F

并使 anchor 预测残差从 R C_U 变为

    η (A P_K Aᵀ + ηI)⁻¹ R C_U.

η>0 时这个矩阵的特征值在(0,1]，所以该 anchor residual 的 Frobenius 范数不增加；不能据此推出其他 query 的误差、LM loss 或生成率也不增加。保护可能牺牲全局 RKHS 近似，且修正后不再保留“keep 系数绝不改动”的原 baseline 性质。

**计算上，这是更容易先接入的融合点。** 复用已有 M×M inverse，只增加 M×p 与 p×p 的小矩阵运算，J 仍是 key-only。输出发生在压缩之前，本块输出不回看 J；修正只影响后续块。可对 FIFO 或 key-score learned selection 应用，coefficient-dependent selection 则会破坏 key-only map 并行性。

注意“写入保护”与“压缩保护”互不替代：W 可以在写时保护，但块末投影又破坏预测；C 只能保护被选的压缩锚点，不能撤回此前块内相互干扰。

## 为什么仍然可以 chunk parallel

固定一个块的临时字典 U（未来行在当前 update 不可写），令 E₀ 将 carried M 行嵌入其中，D_i 为逐原子对角衰减，

    A_(i:j) = D_i … D_j，空乘积为 I。
    u_i = β_i(v_i − h_iᵀD_i C_(i−1))
    C_i = A_(i:0)E₀ C_in + Σ_(j≤i) A_(i:j+1)w_j u_j

得到严格下三角矩阵

    L_ij = −β_i h_iᵀ A_(i:j+1)w_j  (j<i)
    S_i  = −β_i h_iᵀ A_(i:0)E₀
    T    = (I−L)⁻¹
    Uval = T S C_in + T diag(β) V

再令 R_ij=r_iᵀA_(i:j+1)w_j（j≤i），Q_i=r_iᵀA_(i:0)E₀，E 的第j列为 A_(last:j+1)w_j，则

    O_S = Q + R T S
    O_V = R T diag(β)
    F_S = J [A_(last:0)E₀ + E T S]
    F_V = J E T diag(β)

仍然有 O=O_S C_in+O_V V、C_out=F_S C_in+F_V V。可复用 affine scan 和逆序边界伴随框架，不能直接复用假设 one-hot fresh writes 的 block backward 公式。需重新对 w/local solve/J correction 传播梯度。

标量 decay 可分离窗口乘积。slot decay 需要按每个支持原子的 physical slot 应用窗口积；w_j 的 p+1 个坐标可能属于不同 slot，不能给它们统一乘一个 token 标量。所有窗口积可以无除法计算，覆盖 exp(g)=0。

复杂度：W 的方向求解每 token O(p³)，p=2 即3×3；effective kernel contraction可按稀疏p+1 support组织，避免每 token 对 M+C 字典求逆。块内 triangular solve 仍 C×C，压缩 inverse 仍 M×M。C 的额外 correction 小于全字典 KAPA 每 token 求解的成本；具体 Triton 性能尚未测量，不能把代数复杂度写成实测加速。

## 已实现与验证范围

- `research/kapa_block_fusion/prototype.py`：单 batch/head、FIFO、FP64 PyTorch，token reference 与独立块映射路径；W/C 可分别开关；normalized Gaussian，scalar/slot decay，p≤M。
- `research/kapa_block_fusion/validate.py`：比较所有输入梯度、p=0 与冻结 Block reference、跨块未来干预和梯度；含精确零衰减、β=0、尾块、C<M/C>M、初始非零系数、重复 anchor 冲突及压缩残差检验。
- `research/kapa_block_fusion/validation.json`：可追溯数值结果。运行 `python3 research/kapa_block_fusion/validate.py` 可重现；依赖本地 PyTorch，不需 GPU。

目前没有融合 Triton、训练质量、learned-cache、batched 实现或长期稳定性证据。生产 kernel 保持冻结。

## 下一轮最小质量消融

固定同一个 Block slot FIFO + score gate、C64、M32、相同 normalized kernel、LR/WD/数据，做 W×C 的2×2：baseline / write-only / compression-only / both。p=0 必须先数值等价 baseline；p=2 固定首轮，不同时改核或调 gate。先用验证集看方向，再对锁定配方补7 seeds；不要从 confirm 上挑 λ/η。

建议先实现 compression-only 的 Triton：它可复用已有 J 与大部分 block maps，验证成本最低；接着实现 local-write，并保留 W/C 独立开关。必须增加跨块 learned 物理 slot/source-ID、未来中心干预、局部 solve 梯度、J correction 梯度和 chunk-aligned streaming cache 的验收，再谈把两条路线合并。

## 本次数值结果

CPU FP64 共 **505 项检查通过**，最大绝对差 **2.442491e-15**。其中 140 项未来输入干预与 20 项未来梯度检查均严格为零。

朴素 full-block ridge-KAPA 反例中，仅改变未来原子就使当前输出变化 **3.445651e-04**。保护式压缩示例将近期 anchor 预测残差范数从 **6.404180e-03** 降至 **3.030649e-04**；这只是代数数值验证，不是语言模型质量结果。

## 两个 key 的具体例子

取当前 key=(1,0)，过去 anchor=(0.8,0.6)，τ=.5，λ=.05，β=1、当前残差=1。核相关为 0.670320。普通 fresh-slot 写入令当前预测增加1，旧 anchor 预测也增加 0.670320。局部 KAPA 使用系数 [1.687919, -1.077568]，使当前预测增加 0.965604，anchor 只增加 0.053878。这说明它通过给旧原子一个负修正减少干扰，同时牺牲一点当前拟合。β=.2 时这些预测增量统一乘.2。原始数值见 `research/kapa_block_fusion/toy.json`。
