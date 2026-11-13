# KOMP value decay 原型

2026-11-06。实现位于 `komp_slot/value_decay.py`，独立于当前冻结长训包。

## 数学契约

每 head 的 keys 为 `[M,K]`，系数 C 为 `[M,V]`。遗忘替换为
`C_bar = C * exp(g)[None, :]`，`g:[V]`，不再叠加逐 slot 遗忘。
同一个 value 通道在所有 carried slots 和 Block 内临时原子上使用同一 gate。
随后用 `error = v - kernel(dictionary,k)^T C_bar` 写入新原子，读出在本步压缩之前。

Exact 每个 token 压缩一次。Block 每 chunk 压缩一次，尾块也执行压缩；FIFO保留最新M原子，旧原子用带ridge的Gram投影分摊。默认tau=.5、ridge=1e-4，beta独立于decay。

`P(C D_value) = (P C)D_value`，投影和同一时刻的value缩放可交换。核字典固定时，每个value通道的函数统一乘以0到1的数，因此总RKHS能量不会被该遗忘步骤放大。这不保证写入/整个神经网络收缩，也不消除FIFO投影误差。

## 调用

```python
from komp_slot.value_decay import (
    ValueDecayGate, exact_komp_value_decay, block_komp_value_decay,
)

# 在实验mixer初始化时创建并注册；不要在每次forward里重建。
decay = ValueDecayGate(hidden_size=1024, num_heads=16, value_dim=64,
                       rank=64, alpha_init=.995).to(device)

# hidden:[B,T,1024]；q/k/v:[B,T,16,64]；beta:[B,T,16]
g = decay(hidden)  # [B,T,16,64], FP32 log decay
out, (keys, coefficients) = block_komp_value_decay(
    q, k, v, g, beta, (initial_keys, initial_coefficients),
    chunk_size=128, backend='chunk',
)
# Exact调用相同输入，使用exact_komp_value_decay，无chunk_size参数。
```

`backend='recurrent'` 调用逐token参考；`backend='chunk'` 用每个value通道独立的下三角系统计算delta writes。二者支持CPU/CUDA上的普通PyTorch autograd，首轮只实际运行CPU。

新增 `backend='triton'`：CUDA状态前向与手写反向在 `komp_slot/value_decay_triton.py`；Q/K归一化、核相似度、Gram求解及其梯度仍由PyTorch完成。这是混合实现，不是全Triton实现。Exact与Block均通过同一入口选择压缩周期；不支持FP64或二阶梯度。GPU结果见独立 `komp-value-decay-triton-20261106-r*` 包中的证据。

Triton扫描沿value分tile，前向只存块入口、各token写入与累计log decay；反向重算衰减后的状态，避免逐token完整状态tape和每value通道chunk²三角矩阵。核相似度与投影仍有自己的显存开销，Exact依然要做逐tokenGram求解。该版首先验证正确性，不预先承诺比现有slot-decay内核快。

现有 `block_komp_reference(..., decay_axis='value')` 也可直接调用，原默认 `decay_axis='slot'` 行为保留。该参数只新增于参考算子；不能把新gate张量传入旧 `block_komp_fast` 或 `exact_komp` Triton入口假装支持value轴。

## 初值与存储

- 门保留当前 `logsigmoid(z)/8` 公式，初始每个通道alpha=.995；没有顺便换KDA时间尺度初始化。
- 低秩门为1024→64→1024，down随机、up权重零、up bias匹配alpha。初始down梯度为零是预期，up更新后down获得梯度。
- 相比rank64、输出512维的slot门，门参数从98,816增到132,096：每层增加33,280，24层增加798,720；若接到340,753,280参数的FF2272组合上，理论总参数为341,552,000，尚未构建/实测该完整模型。
- recurrent核心状态仍为每head的`32×64 keys + 32×64 coefficients`，均FP32；24层16head单序列共6 MiB，不含短卷积及训练临时缓存。
- 输出dtype与v一致，状态与计算使用FP32；任一输入为FP64时使用FP64。门显式关闭autocast并计算FP32 logits。

## 分块实现说明

块内累计log decay记作G，过去write j对token i的衰减为exp(G_i-G_j)。实现仅对因果位置计算该差值，从不计算全序列逆衰减exp(-G)，避免强遗忘下溢后产生除零、inf或NaN。

每通道的三角系统为

    (I + L_v) w_v = beta * (v_v - old_prediction_v)
    L_v[i,j] = beta_i * kernel(k_i,k_j) * exp(G_i,v-G_j,v), j<i

空间复杂度包含`O(B H V chunk_size²)`临时量。相比原slot版本，多了value通道上的系统维度；“投影可交换”不代表现有高速内核可以零成本复用。这是可微正确性原型，不是生产性能实现。

## 状态接口与边界

- 返回keys/coefficients均按年龄排列，最老在前；生产Exact的物理ring顺序需显式转换，不能直接互换缓存。
- Exact可以在任意token边界拆分调用；Block只有在压缩边界拆分才与整段一致。不保留未完成块的临时字典，不能把每token调用Block当成chunk128流式推理。
- 当前未提供FLA generation cache、完整训练adapter或checkpoint迁移。
- slot/value维度即使恰好相等也必须显式选轴；不能依赖shape猜测。

## 验证

```bash
cd /Users/shom/Downloads/komp-slot-attention-prototype
python3 validate_value_decay.py --output evidence/value-decay/acceptance-r1.json
```

本轮CPU PyTorch 2.12.1共130项检查通过。包括独立逐token递推对分块三角解的输出/初末状态/七组输入梯度、有限差分gradcheck、M/K/V不同维度、实际M32/K64/V64且T129/C128的尾块、全通道相同gate退化到旧标量decay、压缩边界续接、因果、g=0/-1000、beta=0/1、零key/query、FP32/BF16输入、投影可交换与RKHS能量收缩，以及token IDs→投影/门→mixer→CE→Adam的两条3步合成LM链路。

FP64差分对拍最大绝对误差1.4210854715202004e-14。两条合成LM链路均有限且门的两个低秩因子最终获得梯度；该loss下降不构成真实文本收益证据。

验收文件保留三个源文件SHA。没有CUDA数值/速度、8rank full/tail、真实数据或完整FLA训练验收；没有启动新增训练，没有修改五项实验的冻结source或正在使用的模型定义。
