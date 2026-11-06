# Projected slot attention: Triton prototype

实现固定槽位的核投影转移和 delta 写入，支持前向、反向及流式状态续算。**默认后端已改为全 Triton**：块系数、几何 Gram 求解及反向均在 GPU kernel 内完成，详见 [ALL_TRITON.md](ALL_TRITON.md)。原型采用 FLA 的 `[B,T,H,D]` 布局、FP32 recurrent state 和 value 维分块方式，独立于上游包运行。

参考 checkout：`../flash-linear-attention-komp`，上游 `fla-org/flash-linear-attention`，commit `97874af2400916e2885c33ea30559a70ff686b66`。已核对 `fla/ops/gsa/`、`fla/ops/gated_delta_rule/` 和 `fla/ops/generalized_delta_rule/iplr/`。没有修改或发布上游代码。

本轮新增 GDN scalar gate 和 GSA 逐 slot pre-decay，均支持 Triton 前反向。四种门对照、各两seed的完整训练结果见 [decay 实验](physics_lm/decay_runs/RESULTS.md)，同卡加速结果见 [DECAY_ACCELERATION_RESULTS.md](DECAY_ACCELERATION_RESULTS.md)。长序列 `T>=192, M>=16` 默认用并行整数route scan，原IEEE求解不变；固定旧基线可调用 `triton_geometry_serial`。

## 更新规则

每个 batch/head 独立维护 key 状态 `K: [M,Dk]` 和系数状态 `V: [M,Dv]`。输入 `slots[t]` 给出替换位置 j，e 为其 one-hot 向量。

```text
κ(u,k) = exp((normalize(u) · normalize(k) - 1) / temperature)
h       = κ(K_old, k_t)
K_new   = (I - e eᵀ) K_old + e normalize(k_t)ᵀ
p       = solve(κ(K_new,K_new) + ridge I, κ(K_new,K_old[j]))
V_pre   = alpha_t V_old
delta   = v_t - V_preᵀ h
V_new   = V_pre + (p-e) V_pre[j] + beta_t e deltaᵀ
o_t     = V_newᵀ κ(K_new,q_t)
```

`beta` 为 `[B,T,H]` 标量门；`alpha` 支持 `[B,T,H]` 或 `[B,T,H,M]`，范围均为 `[0,1]`。逐slot版本把 `V_pre` 改成 `alpha_t[:,None] * V_old`，作用于旧字典系数，当前token的新写入不受当步遗忘缩放。逐slot只支持 `all_triton` 和 `torch` 后端。更新后读取；未来 token 不影响此前输出。读出使用未做 softmax 归一化的核展开。

门模块接口为 `komp_slot.decay.gdn_decay(logits, A_log, dt_bias)` 和 `gsa_decay(logits, normalizer=8)`，返回FP32 alpha。`normalizer=1`是sigmoid。完整模型为匹配初始alpha/beta而将门Linear也保留FP32。`physics_lm/decay_experiment.py --preflight`生成与源码hash绑定的预检票据后，才可运行相应 `--kind sigmoid|gdn_scalar|gsa_scalar|gsa_slot --seed ...` 训练。

`ridge=0`、Gram 可逆时，是给定删除槽位后的精确核投影。默认 `ridge=1e-4` 对被删除原子的投影做正则化，允许重复 key；它改变投影系数，并非精确 KOMP。`temperature=0.5` 是待调实验参数。同一个调用的读出、历史相似度和 Gram 使用相同温度。

删除位置由调用者提供。没有实现完整 KOMP 的最小删除代价搜索。温度和 ridge 当前是固定标量；不支持对它们求梯度。零向量采用 `normalize(..., eps=1e-6)`，其核对角值不一定为 1。

## 代码

| 文件 | 作用 |
|---|---|
| `komp_slot/geometry.py` | Triton 归一化、key 来源追踪、并行 Gram 求解及解析反向 |
| `komp_slot/triton_chunk.py` | 融合块系数、块间 scan、块内重算反向及归约 |
| `komp_slot/utils.py` | Triton 初始化与输入数值检查 |
| `komp_slot/reference.py` | 可微的 PyTorch key 轨迹和核求解；稠密转移参考；独立的全函数投影 oracle |
| `komp_slot/kernels.py` | 融合 recurrent 前向与解析反向，CUDA/FP32 |
| `komp_slot/chunk.py` | rank-2 块三角求解、块间 scan 和反向；块系数由 PyTorch 构造 |
| `komp_slot/operator.py` | 输入检查、低精度输入转换、两阶段算子入口 |
| `validate.py` | 输出、状态、所有支持梯度、因果性、流式续算、训练 smoke 与计时 |
| `validate_chunk.py` | chunk 与独立稠密递推/fused recurrent 的数值对照和计时 |

旧 `backend='triton'` 的前向在每个 program 内顺序处理 token，沿 batch/head/value tile 并行。反向沿时间逆序计算伴随状态，跨 value tile 的梯度用独立 partial buffer 归约，无原子加法。key 轨迹与 Gram 求解由 PyTorch 执行，梯度穿过 solve、归一化和替换操作。另有 `backend='chunk'`：rank-2 块三角求解、块间递推及反向使用 Triton，四个块转移矩阵由可微 PyTorch 批量矩阵乘法准备；支持 chunk size 16/32。推导和实现边界见 [CHUNK.md](CHUNK.md)。这两个旧后端保留用于比较；默认 `all_triton` 将块系数和几何阶段都搬到了 Triton。

fused recurrent 训练保存每一步 value 状态，额外空间为 `O(B*T*H*M*Dv)`，不适合直接用于长序列大模型。无梯度推理不保存该 tape；整段几何轨迹仍占 `O(B*T*H*M)` 以上空间。逐 token 调用的缓存仅为固定 key/value 状态。chunk scan 仅保存块边界状态，但块系数和反向 partial buffer 仍有额外开销，尚未完成峰值内存比较。

## 使用

运行环境：Python >=3.10、PyTorch >=2.7、Triton >=3.3、NVIDIA CUDA GPU。验收脚本额外读取 protobuf 版本。原型不需要安装 FLA 的 transformers 依赖。

```python
import torch
from komp_slot import projected_slot_attention

B, T, H, Dk, Dv, M = 2, 32, 4, 32, 64, 8
q = torch.randn(B, T, H, Dk, device='cuda', requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn(B, T, H, Dv, device='cuda', requires_grad=True)
alpha = torch.full((B, T, H), .98, device='cuda', requires_grad=True)
beta = torch.full((B, T, H), .2, device='cuda', requires_grad=True)
slots = torch.randint(M, (B, T, H), device='cuda')
keys0 = torch.randn(B, H, M, Dk, device='cuda', requires_grad=True)
values0 = torch.zeros(B, H, M, Dv, device='cuda', requires_grad=True)

output, state = projected_slot_attention(
    q, k, v, alpha, beta, slots, (keys0, values0),
    backend='all_triton', chunk_size=16,
)
# backend='chunk' 保留旧混合 chunk；'triton' 为 fused recurrent；'torch' 为稠密 oracle。
output.square().mean().backward()
# 下一段使用 state；保留计算图可跨段训练，detach 后可截断反向传播。
```

支持非空 dense batch、1–64 个槽位、任意正的 QK/value 宽度。高层入口支持 FP32/BF16/FP16 特征，内部转换成 FP32，并返回 FP32 输出与状态；PyTorch 参考额外支持 float64。没有 varlen、GQA、二阶梯度、可学习温度或 softmax 读出。硬 slot index 不求导；可用 router 的选中概率调节 beta，训练选中分支，见 `train_smoke()`。

旧后端的 GPU 验收覆盖 FP32/BF16；使用旧后端时应关闭外部 autocast，避免 PyTorch 几何阶段再次降精度。默认全 Triton 后端在 kernel 内显式使用 FP32，不依赖 autocast 设置；最终覆盖范围和证据见下文。

低层 `fused_recurrent_slot` 和 `chunk_slot(..., backend='triton')` 仅支持 FP32 CUDA，同 device。其调用者负责合法的 slot 范围及有限数值；高层入口在启动前验证这些条件。输入检查会同步 GPU，因此高层接口不是 CUDA Graph 捕获入口。

## 验证与数值预算

```sh
python validate.py --cpu
python validate.py --benchmark
python validate_chunk.py --cpu
python validate_chunk.py --benchmark
python validate_all_triton.py --benchmark
```

验证在禁用 TF32、开启未初始化内存 NaN 填充的条件下运行。核比较用 float64 稠密矩阵递推作为 oracle；`ridge=0` 另用完整函数的 Gram 投影检查秩二改写。几何链路用有限差分 gradcheck 验证。

预先设定的核心 FP32 容差：输出和状态 `atol=3e-5, rtol=3e-4`，各梯度 `atol=6e-5, rtol=6e-4`。完整几何的 FP32 梯度 `atol=2e-4, rtol=1e-3`；BF16 输入梯度返回 BF16，使用 `atol=3e-3, rtol=1e-2`。具体检查和误差保存在 `evidence/*.json`。

检查覆盖槽位/value tile 尾部、M=1、M=64、T=1、T=129、标量门的 0/1 边界、非零初态、仅输出/仅终态 loss、非连续输入、重复和零 key、流式状态及其梯度、未来输入扰动、非法路由索引。训练 smoke 只验证合成任务能反向更新，不构成语言模型或记忆容量收益证据。

计时分开报告预计算几何后的 recurrence，以及含几何和输入检查的完整调用。recurrent 验收对照同一算子的 PyTorch 参考，chunk 验收直接对照 fused recurrent；均不是与 GSA、Raven 或 GDN 的模型效果/吞吐比较。

## 历史 recurrent 验收

2026-10-31，在 BP Dev 的 `WeaverRSI` 项目使用单张 NVIDIA H100 80GB HBM3，PyTorch `2.10.0a0+b558c986e8.nv25.11`、CUDA 13.0、Triton 3.6.0。105 项 GPU 数值检查通过，CPU 全函数投影对照和有限差分 gradcheck 通过，Triton CPU 解释执行另有 55 项检查通过。

30 步合成训练 loss 从 `0.0183066` 降至 `0.0093090`，query/key/value 映射、初始 key 和选中 router 分支均得到有限梯度。该结果仅证明训练链路可执行。

| 工作量 `(B,T,H,M,Dv)` | PyTorch 前向 | Triton 前向 | PyTorch 前反向 | Triton 前反向 |
|---|---:|---:|---:|---:|
| `(2,128,4,8,32)` | 32.755 ms | 0.111 ms | 123.765 ms | 0.418 ms |
| `(2,256,4,16,64)` | 65.094 ms | 0.258 ms | 250.690 ms | 0.761 ms |

以上只测已准备好核几何的 recurrence，PyTorch 对照有逐 token Python 和 kernel launch 开销。完整调用的独立小例子 `(B,T,H,Dk,Dv,M)=(2,32,2,7,11,8)`：PyTorch `33.804 ms`，Triton `25.898 ms`，其中几何准备 `24.988 ms`。目前的主要成本是 PyTorch key/Gram 阶段。

日志和逐项误差在 `evidence/gpu-job.log`、`evidence/gpu.json`；后者保存实际运行源码 SHA-256。GPU 作业 alias 为 `komp-slot-prototype-20261031-r1`。使用当前 `project: WeaverRSI`；`default` 项目会被平台以 HTTP 403 拒绝。


## 历史混合 chunk 验收

新增 `backend='chunk', chunk_size=16|32`。2026-10-31 在同一镜像和 WeaverRSI 单张 H100 上，310 项 GPU 检查通过；CPU float64 块算法另有 122 项检查通过。GPU 检查包含两种块长的所有输入梯度、初末状态梯度、零门及重复路由、仅输出/仅终态 loss、非连续输入、FP32/BF16 完整几何、流式和因果性，并与 fused recurrent 对照到 T=2048。合成训练 30 步 loss 为 `0.0183066 -> 0.0093090`。

| `(B,T,H,M,Dv)` | C | recurrent 前向 | chunk 前向 | recurrent 前反向 | chunk 前反向 |
|---|---:|---:|---:|---:|---:|
| `(2,256,4,16,64)` | 16 | 0.254 ms | 1.900 ms | 0.768 ms | 6.295 ms |
| `(2,1024,4,16,64)` | 16 | 0.964 ms | 1.917 ms | 3.068 ms | 6.321 ms |
| `(2,2048,4,32,64)` | 16 | 2.324 ms | 2.209 ms | 6.849 ms | 6.359 ms |
| `(2,2048,4,32,64)` | 32 | 2.324 ms | 3.028 ms | 6.849 ms | 6.345 ms |

计时包含所有块系数构造和 chunk scan，不包含 key/Gram 几何；预热 30 ms、采样 100 ms，排除首次编译。这是单轮微测，不足以确定几个百分点的稳定收益。系数准备单独测得约 1.66–1.72 ms；短序列明显慢于 fused recurrent。各部分独立计时有缓存和调度差异，不能直接相加。当前实现证明 chunk 化和训练链路可行，尚未完成系数构造融合或性能调优。

第一次 GPU 验收发现有限幂级数求逆在重复槽位写入时发生数值抵消，已改为分块三角求逆合并，并增加直接逆矩阵前反向回归；没有放宽数值容差。失败证据保留在 `evidence/chunk-gpu-r2-failed.json`。通过的报告在 `evidence/chunk-gpu.json`，日志为 `evidence/chunk-gpu-job.log`；报告保存源码 SHA-256。完成作业 alias 为 `komp-slot-prototype-20261031-r3`，identity 为 `job-e7436aed8c4c-20261031150248`。


## 全 Triton 验收与加速结果

2026-10-31，在 WeaverRSI 单张 H100 80GB、Triton 3.6.0 上完成 **422 项检查**，原数值容差不变；FP32/BF16/FP16、输出/终态、全部输入梯度、非连续输入、零门、重复路由、尾块、跨块流式、因果性及长序列对照均通过。30 步合成训练 loss `0.0183066 -> 0.00930903`。Profiler 确认该路径前反向没有 ATen GEMM、solve、exp、norm、乘加除法或归约；内存分配、autograd 包装和必要的布局复制仍使用 PyTorch。

同轮 eager 微测，包含主机调度间隙；预热 30 ms、采样 100 ms、排除编译。TF32x3 块乘法、IEEE 几何乘法，开启未初始化内存 NaN 填充。小幅计时差异可能受调度影响，结果不代表通用模型吞吐。

| recurrence `(B,T,H,M,Dv)`，C=16 | fused recurrent 前向 | 旧混合 chunk 前向 | 全 Triton 前向 | fused 前反向 | 全 Triton 前反向 |
|---|---:|---:|---:|---:|---:|
| `(2, 256, 4, 16, 64)` | 0.255 ms | 2.412 ms | 0.128 ms | 0.743 ms | 0.811 ms |
| `(2, 1024, 4, 16, 64)` | 0.988 ms | 2.443 ms | 0.147 ms | 3.029 ms | 0.869 ms |
| `(2, 2048, 4, 32, 64)` | 2.363 ms | 2.447 ms | 0.265 ms | 6.952 ms | 0.818 ms |

以上只比较已给定 `h,p,r` 的核心 recurrence；全 Triton 的计时已包含块系数构造。C=16 的系数构造分别约 0.026/0.066/0.088 ms。2048-token 配置前向相对 fused recurrent 约 8.9 倍加速，前反向约 8.5 倍。

| 完整调用 `(B,T,H,Dk,Dv,M)`，C=16 | 旧混合前向 | 全 Triton 前向 | 旧混合前反向 | 全 Triton 前反向 |
|---|---:|---:|---:|---:|
| `(2, 32, 2, 7, 11, 8)` | 36.263 ms | 0.503 ms | 96.912 ms | 1.485 ms |
| `(2, 256, 2, 7, 11, 16)` | 262.389 ms | 0.516 ms | 706.254 ms | 1.551 ms |
| `(2, 1024, 2, 7, 11, 16)` | 1036.324 ms | 0.526 ms | 2862.904 ms | 1.521 ms |

完整调用包含归一化、路由追踪、核/Gram、投影求解、输入检查及 value recurrence。这里的极大加速主要来自消除了逐 token PyTorch 几何阶段的 Python/kernel launch 开销；不是与已有优化 GSA/GDN 算子的比较。尚未验证模型质量或容量提升。

主实现和内存边界见 [ALL_TRITON.md](ALL_TRITON.md)。反向仍分配逐 token 重算缓冲，几何梯度使用 FP32 atomic add，不保证逐位确定性；高层输入检查有一次 CPU 同步，不能直接捕获成 CUDA Graph。

证据：`evidence/all-triton-gpu.json`（误差、环境、源码 SHA-256、profiler 事件和完整计时），`evidence/all-triton-gpu.log`。通过的最终作业 alias 为 `komp-slot-prototype-20261031-r5`，identity 为 `job-924789b5713c-20261031160802`；本轮 r4/r5 作业均已删除并查询确认。历史 r4 是融合输入检查之前的版本，不应混用它的计时与最终版本。

## 2026-11-01 加速与 Block-KOMP

- 逐 token KOMP：`backend='all_triton'` 现为新路径（`geometry_chunk.py` 块内逆矩阵 Woodbury 链 + 一步精化；`fast_chunk.py` 两级仿射 scan + 单 warp 局部反向），旧实现保留为 `backend='all_triton_v1'`。验收：`validate_geometry_chunk.py`、`validate_recurrence.py --candidate fast`、`validate_slot_decay.py --core fast`，证据在 `evidence/accel-20261101/`。公开 API 的 `validate_all_triton.py` 全量回归尚未在新默认上重跑。
- Block-KOMP（块同步淘汰、GDN 式对数衰减、FLA 式分阶段 kernel）：见 [BLOCK_KOMP.md](BLOCK_KOMP.md)。第 3 轮（同日）：learned 淘汰进入 Triton；跨 kernel 融合后 fwd+bwd 共 18 个 kernel（learned 21；GDN 21，GSA 31–34）；注意力侧用 BF16 tensor core。8k（B1 T8192 H64 D128 M64）标量 9.89 ms（GDN 4.40，GSA 9.11），per-slot 11.03，learned 12.84。573 项验收通过，证据在 `evidence/accel-20261101c/`。第 4 轮：C==M 特化、BF16 tape、单遍串行 scan 后 8k 标量 7.59 ms（GDN 4.40 的 1.72 倍），per-slot 8.83，learned 11.65；fwd+bwd 14 个 kernel。证据在 `evidence/accel-20261101d/`。

## 2026-11-01 BSGD-inspired merging 原型

新增独立的 `komp_slot/bsgd.py`：Gaussian RBF 中心合并、同步系数修正，支持可学习合并门和有限网格 RKHS 误差搜索。前向与一阶反向均为 Triton，支持 scalar/slot decay 和任意 token 断点续接。H100 上 402 项验收通过；当前仍是逐 token 递归原型，速度明显慢于 block KOMP。公式、API、实测结果与限制见 [BSGD_MERGING.md](BSGD_MERGING.md)。

新增 gated 策略的等价 chunk 路径 `komp_slot/bsgd_chunk.py`：按槽中心轨迹、rank-1 块内系数、两级状态 scan 和完整一阶 backward。支持 scalar/slot decay；不改变逐 token merging 语义。设计与验收见 [BSGD_CHUNK.md](BSGD_CHUNK.md)。

## 2026-11-01 KAPA 保护式写入原型

`komp_slot/kapa.py` 实现固定字典上的 soft-protected KAPA：以过去 key 的预测为软保护约束，拟合当前残差。Gaussian 特征、Gram 求逆、方向求解、递归及完整 fwd/bwd 均为 Triton，保留独立 PyTorch reference。H100 上原有 441 项和新增 690 项检查通过（共 1131 项），含原始 q/k/中心梯度与因果性；当前 recurrence 串行，无 streaming anchor cache。原文与适配的区别、公式、API、结果和限制见 [KAPA_PROTOTYPE.md](KAPA_PROTOTYPE.md)。


## 2026-11-01 KAPA chunk parallel

`komp_slot/kapa_chunk.py` 新增 `chunk_protected_kapa` / `chunk_gaussian_protected_kapa`，实现块内下三角求解、双层 affine scan，以及逆序扫描和各块独立反向。标量/逐 slot 衰减都支持，chunk_size 为 16/32/64，内部 chunk 边界保留 anchor 历史。H100 上 1,608 项检查通过，240 项因果干预/梯度检查严格为零。完整 raw-key 测速与大 batch 下的性能限制见 [KAPA_CHUNK.md](KAPA_CHUNK.md)。
