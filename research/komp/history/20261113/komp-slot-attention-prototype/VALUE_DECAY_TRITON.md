# Value decay Triton 原型验收

后续优化已完成：见 [VALUE_DECAY_NATIVE.md](VALUE_DECAY_NATIVE.md)。以下保留初版混合后端的历史验收与失败证据。

2026-11-06。独立短测，未修改冻结模型、未启动新长训。

## 实现范围

调用 `block_komp_value_decay(..., backend='triton')` 或
`exact_komp_value_decay(..., backend='triton')`。

- `komp_slot/value_decay_triton.py`：Triton状态前向、手写反向及autograd包装。
- Q/K归一化、核相似度、FIFO字典几何与批量Gram求解及其反向保留PyTorch。因此是混合后端，不能称全Triton算子。
- 支持首阶q/k/v/g/beta/初始keys/初始coefficients梯度，输出及末态梯度均验收。无二阶梯度。
- FP32状态计算，支持FP32/BF16/FP16输入接口；实际GPU数值验收覆盖FP32/BF16。FP64使用独立PyTorch参考。
- 现阶段限制M≤64、chunk≤128；返回年龄顺序状态，沿用CPU原型的压缩边界与读出语义。
- 新增临时缓存为块入口系数、各token写入值和累计log decay；没有每value通道的chunk²系统。反向利用缓存重算每步衰减状态，强遗忘不依赖逆衰减。

## 验证结果

单张 NVIDIA H100 80GB HBM3；PyTorch 2.10.0a0+b558c986e8.nv25.11、Triton3.7.1、CUDA13.0。

168项检查全部通过，包括FP64逐token参考输出、keys/coefficients末态、七组输入梯度、不同K/V/M、M32/K64/V64、Exact逐token压缩、Block C128/T129尾块、强遗忘g=-1000、无遗忘、beta0/1、零key/query、因果性、跨压缩边界续接、两条小型合成LM的Adam更新。实际B16/T2048/H16算子另执行前反向并确认梯度有限；该大shape没有做全FP64对拍。

随机对拍中最大RMS相对误差：FP32约1.95e-6；BF16约2.11e-3。该结论限于验收样本，不覆盖任意近共线字典。没有8rank/full模型恢复或真实语料质量验收。

## 测速

输入B16/T2048/H16/K64/V64/M32，BF16 q/k/v、FP32 gates/state，单层算子输出和末态平方损失反向；3个暖态样本中位数。包含PyTorch几何/求解、Triton扫描和两部分的反向，不包含网络投影/卷积/MLP/CE/优化器/DDP。

| 压缩方式 | 前向+反向 | 峰值allocated | 峰值reserved |
|---|---:|---:|---:|
| Exact，C1 | 174.62 ms | 21.97 GiB | 41.20 GiB |
| Block，C128 | 60.37 ms | 3.86 GiB | 4.57 GiB |

小shape B1/T256/H4：Exact5.80ms、Block5.96ms，主要用于说明这些数据不是完整模型吞吐，也不能以单一shape外推加速比。本轮没有对旧slot-decay Triton算子做同卡计时，不声称相对旧内核加速。

Exact仍显式批量构建每token Gram及其autograd图，开销很大；后续优先将几何/求解及梯度改为重算或融合，并分阶段profile。Block还保留顺序状态扫描。当前是经过GPU数值验证的可训练算子原型，不是已可替换正式24层长训的优化版本。

## 首次失败与修复

r1单步通过，多步Exact前向失败（相对RMS .7987）。没有继续盲跑：独立CPU复现同一批量几何与FIFO公式，相对参考最大绝对误差约5.55e-13；检查状态指针、压缩和tail后，将同kernel中Entry tape的store/reload改为寄存器state的gather，消除跨布局内存往返。r2覆盖原失败用例以及全部其他验证并通过。该修复的实测效果已确认；底层可见性/编译重排的具体机制未单独做汇编级诊断。

首次失败代码、日志和receipt保存在 `/Users/shom/Downloads/komp-value-decay-triton-20261106-r1`。通过版本在 `/Users/shom/Downloads/komp-value-decay-triton-20261106-r2`，`evidence/acceptance.json`包含源SHA、完整数值与测速结果；`evidence/cleanup.json`记录作业终态清理。两个作业均使用WeaverRSI/proj-32、独立单卡，不操作长训作业。
