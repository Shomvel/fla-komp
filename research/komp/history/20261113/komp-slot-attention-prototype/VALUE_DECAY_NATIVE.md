# Value decay：复用原生 KOMP 的实现

本次优化不修改冻结的预训练源码。接口、压缩时机、FIFO、值通道门与原 value-decay 原型保持一致。

## 换坐标，不丢弃 value gate

设每个 value 通道的 log decay 为 g[t,v]。取 s[t]=mean_v(g[t,v])，r[t,v]=sum_{i<=t}(g[i,v]-s[i])，并令 C'[t]=C[t] diag(exp(-r[t]))。

KOMP 的字典更新、核读出、delta write 的线性部分和压缩投影都作用在系数矩阵左侧，与右侧对角缩放可交换。因此在新坐标中，原生 KOMP 只需接收 scalar decay s[t] 与缩放后的 value v'[t]=v[t] exp(-r[t])，输出及末态再乘 exp(r)。逐通道差异没有被平均掉，它完整保留在缩放中，g 的梯度穿过这条路径。

- Exact 复用 `exact_komp/operator.py` 的 Triton 几何、直接投影求解、分块映射、状态扫描和手写反向。算法压缩仍为逐 token。
- Block 复用 `block_fast.py` 的投影、三角系统、状态扫描和手写反向；BF16 输入沿用旧内核的 attention / backward tape 精度策略。
- Exact 的物理 ring 末态按 T mod M 旋转回年龄顺序，保证原 value-decay streaming 契约。
- 门、状态及缩放后的 value 使用 FP32，最后输出转回输入 value dtype。

## 数值范围

逐 batch/head/value 选择一个 detached 的常数坐标原点，使累计残差的范围居中。原点包含 t=0，初始系数相应缩放。原点是坐标自由度，保持常数不改变所求梯度。

若累计残差跨度超过 24，回退到 `value_decay_hybrid.py` 中已验证的无逆衰减实现；不 clamp gate，不改变压缩边界。均匀强遗忘不必然触发回退，因为公共 scalar decay 仍由旧内核处理。任意合法 chunk（例如 C=3/4）以及不受原生内核支持的 dtype/维度也继续使用混合后端。

这是浮点运算顺序的改变，不保证 bitwise 等价。旧内核近共线字典的 FP32 限制仍然适用。

## 验证

`validate_value_decay_coordinates.py` 独立用 FP64 PyTorch reference 对拍换坐标前后的输出、keys/coefficients 末态和七组输入梯度；60 项通过，最大绝对差 2.22e-15。覆盖相同通道门、非均匀通道门，以及 C=1/4/16。

GPU 数值、测速与显存结果以本次独立 job 的 evidence 为准，尚未以数学推导代替实测。测速比较原生 slot decay、原生 scalar decay、旧混合 value decay、新复用 value decay，使用同卡同 shape、相同损失及梯度范围。

## H100 实测结果（2026-11-06）

独立单 H100 80GB，torch 2.10.0a0 / Triton 3.7.1 / CUDA 13。288 项 GPU 验收全部通过，保持原验收门限：一般 FP32 RMSrel≤.003 或 abs≤3e-4，BF16 RMSrel≤.025 或 abs≤3e-4；强遗忘专用门限沿用原测试。随机对拍最大 RMSrel：FP32 .002798（Block C128/T257 的 dg），BF16 .015079（同项）；原生 Block 的 attention/state 算法保留其 TF32/BF16 精度，误差高于旧混合后端，未声称严格或 bitwise 等价。均匀 -1000 强遗忘的参考门梯度为零附近，观察到约 2.53e-7 的绝对误差，此项以绝对误差判断，不使用相对误差宣传精度。

验收包括七组输入梯度、输出和末态、FP32/BF16、Exact、C16/32/64/128 及尾块、强遗忘与回退、同值通道门下非零门梯度、因果性、跨压缩边界续接和小型合成 LM Adam 更新。独立 FP64 CPU 换坐标验证另 60 项通过。

同一张卡、B16/T2048/H16/K64/V64/M32，BF16 q/k/v、FP32 gate/state，包含输入检查、完整算子前后向及输出/末态损失；不含网络投影、MLP、CE、DDP。每项 3 次预热、7 次 CUDA event 样本，取中位数。slot 基线原生接口没有 value 包装器的完整输入扫描；value 计时含扫描，未在计时中扣除。

| 模式 | 门分布 | 旧 slot ms | scalar 基线 ms | 旧 value ms | 新 value ms | 新/旧 slot 耗时 | 新 value peak allocated GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| Exact | init | 72.64 | 69.76 | 174.27 | 76.68 | 1.056× | 3.82 |
| Exact | random | 72.70 | 69.78 | 174.41 | 76.87 | 1.057× | 3.82 |
| Block C128 | init | 32.08 | 16.24 | 60.24 | 22.58 | 0.704× | 4.59 |
| Block C128 | random | 32.04 | 16.23 | 60.23 | 22.57 | 0.704× | 4.59 |

init 为所有通道 log(.995)，random 为各 token/channel 独立 U(-.1,0)。相对旧 value 原型，随机门 Exact 快 2.27×、Block 快 2.67×。Exact 峰值 allocated 从 21.97 降到 3.82 GiB；Block 从 3.86 增至 4.59 GiB。旧 slot 的对应峰值为 Exact 3.20、Block 3.96 GiB。

速度结论限于测量门分布；累计通道差异跨度超过 24 会走原混合后端，速度会下降。大 shape 只检查所有梯度有限，完整 FP64 差分在小 shape 上完成。没有新增真实语料长训或全模型/8rank 验收。

## 源码及作业留证

- 新实现：`komp_slot/value_decay_triton.py`；旧实现保留为 `komp_slot/value_decay_hybrid.py`。默认 `backend='triton'` 已接入新版，fallback 保持独立。缩放和其 autograd 使用 PyTorch，几何、求解、扫描和核心反向直接复用原 Triton；不称所有操作均用 Triton。
- `evidence/value-decay/native-source-reuse.json` 核对复用的 10 个核心文件与冻结 FF2272 训练源码逐字节一致。
- 本轮结果：`/Users/shom/Downloads/komp-value-decay-native-20261106-r2/evidence/acceptance.json`。源清单和完整日志保留在同目录及同名 GPFS 包。
- r1 identity `job-f2a8d8f8466a-20261106183322` 因 GPFS 冷缓存编译开销主动停止并精确删除；已完成的数值用例无失败，未将中止当成 PASS。源码/日志/停止原因保留。
- r2 identity `job-241c682b524b-20261106184153` 使用同一冻结实现和门限、容器本地编译缓存；确认 Completed 后精确删除，复查 absent。两次短测均为 WeaverRSI/proj-32 单卡，没有操作任何长训作业。
