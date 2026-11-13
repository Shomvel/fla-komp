# KOMP kernel 改动与性能

表内比较均来自各自实验内的同卡对照。所有 GPU 为 H100 80GB；不同实验的输入、版本、精度和计时方法分别记录，不跨实验相除生成加速比。原精度计时、显存口径、回执路径与 SHA256 见 [KERNEL_TIMINGS.csv](results/20261113/KERNEL_TIMINGS.csv)。

## Exact：两轮算子优化

固定 B1/T8192/H2/K64/V64/M32，BF16 输入，FP32 状态，计算块 C16。表中是完整前向加反向的 CUDA Graph replay；编译、capture 和 Python 调度不计时。每组 20 次，3 组取中位数。显存为 eager 单独测得的峰值 allocated 增量。

| 实验 | 对照 | 候选 | 前反向 ms：对照 → 候选 | 加速比 |
| --- | --- | --- | ---: | ---: |
| 第一轮 FIFO | FP64 geometry | 稳定 FP32 geometry、拆分反向 | 20.9525 → 3.9661 | 5.28× |
| 第一轮 learned | FP64 geometry | FP32 geometry、拒绝区间批量跳过 | 22.7099 → 4.1317 | 5.50× |
| 第二轮 FIFO | 上轮优化版 | 直接 RHS、分层与局部 affine scan | 3.9669 → 2.1205 | 1.87× |
| 第二轮 learned | 上轮优化版 | 字典版本复用 LDLᵀ、分层 scan | 4.1385 → 1.6878 | 2.45× |

第一轮 BF16 显存增量由 468.88 降至 88.50 MiB。第二轮 FIFO/learned 为 89.75/91.75 MiB。第二轮同卡 GDN 为 0.4657 ms、30.44 MiB，因此最终 FIFO/learned 前反向耗时仍为该 GDN 的 4.55/3.62 倍。状态元素数匹配不代表算法表达能力或模型质量相同。

第一轮 469 项检查与 20 组 selector 轨迹检查通过；第二轮 586 项检查通过，另有各轮独立的 GDN 对拍。第二轮包含输出、终态、输入梯度、近重复 keys、尾块、强遗忘、流式与 8K 前缀因果性。FP32 近共线字典仍有边界，不能将这些通过解释成任意输入的 FP64 等价。

原报告及证据：[第一轮](history/20261113/komp-exact-opt-20261105/RESULTS.md)、[第二轮](history/20261113/komp-exact-opt2-20261105/RESULTS.md)。`solve_only`、`reuse_z` 等较慢候选与独立开关消融均保留。`mixed_state` 只完成特定随机输入实验，未作为默认接口。

## Block C128

C128 使用四个 32×32 子块的三角逆，以及两阶段输出反向。C64 和 C128 具有不同压缩边界；表中两者用于观察各自执行成本，不能据此声称数值等价。Flex 使用 FP32、Triton 使用 BF16，原始精度口径保持。

固定 M32/K64/V64，前反向包含 autograd，编译不计时。以下摘录稳定测量的完整前向加反向结果：

| B/T/H | 淘汰 | Triton C64 ms | Triton C128 ms | Flex C128 ms |
| --- | --- | ---: | ---: | ---: |
| 1/512/8 | FIFO | 0.949 | 1.044 | 29.494 |
| 8/512/8 | FIFO | 0.836 | 1.959 | 27.618 |
| 1/8192/8 | FIFO | 0.958 | 3.976 | 455.627 |
| 1/8192/8 | learned | 1.618 | 5.437 | 460.878 |

600 项 C128 验收、573 项旧路径回归、18 项集成检查分别通过。最大相对 RMS 为 0.00527463/0.00716693/0.00351007；这些是各 suite 的最大值，不使用统一门限合并判断。[完整报告](history/20261113/komp-c128-triton-20261101/C128_TRITON.md)保留每组门限、显存与输入规模。

## Value decay

将通道相关衰减换坐标后，复用原生 Exact/Block geometry、scan 与手写反向。累计通道残差跨度超过 24 时回退到稳定 hybrid 后端，不裁剪 gate。换坐标与缩放的部分操作使用 PyTorch。

同卡 B16/T2048/H16/K64/V64/M32，BF16 q/k/v、FP32 gate/state，随机门为各 token/channel 独立 U(-0.1,0)。包括包装器检查和完整算子前后向，不含模型投影、MLP、CE 或 DDP。每项 3 次预热、7 次 CUDA events，取中位数。

| 模式 | 旧 hybrid value ms | native value ms | 加速比 | 峰值 allocated GiB：旧 → 新 |
| --- | ---: | ---: | ---: | ---: |
| Exact | 174.41 | 76.87 | 2.27× | 21.97 → 3.82 |
| Block C128 | 60.23 | 22.57 | 2.67× | 3.86 → 4.59 |

288 项 GPU 检查与独立 60 项 FP64 CPU 坐标检查通过；GPU 覆盖 7 组梯度、终态、尾块、强遗忘、回退与短 Adam 更新。随机小输入最大 RMSrel 为 FP32 0.002798、BF16 0.015079；大 shape 仅检查梯度有限。强遗忘与回退的速度不由上表代表。

包内 value-decay 集成另在 H100 上通过 34 项测试，并运行 Exact/Block 默认 341,552,000 参数模型的一步 Adam 与 checkpoint 加载。一步加载不等于后续训练轨迹一致性。[原生复用报告](history/20261113/komp-slot-attention-prototype/VALUE_DECAY_NATIVE.md)、[包内集成回执](VALUE_DECAY_INTEGRATION.md)分别保留。

## 完整模型重算优化

这两组计时包含网络 forward/backward、checkpoint 重算、clip 与 Adam。使用真实训练 checkpoint 和固定输入，同卡 ABBA，每阶段 10 个暖步；profile 与正式计时分开。

| 模型与形状 | 改动 | 原版 s/step | 候选 s/step | 吞吐倍数 |
| --- | --- | ---: | ---: | ---: |
| Exact 339,966,848 参数；B16×2/T2048 | 图内 geometry cache、去 gc tape | 5.241388 | 4.467272 | 1.1733× |
| Learned+123 1,003,181,056 参数；B16/T2048/C128 | 反向归约、线程布局、UT 反向、FF 融合、重算缓存 | 2.594977 | 1.469367 | 1.7661× |

Exact 候选的峰值 allocated 从 30.948 增至 35.242 GiB；该组只有单卡固定输入验收，没有推广到正式 8 卡长训。去 gc tape 单项未测出独立速度收益，主要计时收益来自 geometry cache。[对照与限制](history/20261113/exact-komp-efficiency-research-20261106/REPORT.md)。

Learned+123 的 v2 保留 IEEE UT 和原 PyTorch RMSNorm，采用槽位衰减反向直接求和、实测线程布局、两阶段 UT 反向、兼容 BF16 中间舍入的 FF 融合、图内不可变 UT/dictionary 缓存。单算子基线 64.202 ms；直接衰减归约候选 38.119 ms；两阶段 UT 候选 35.569 ms。这些是独立候选计时，收益不能相加。

最终整模型 CE 与原版相同，均为 2.704592466；579 组参数梯度和 checkpoint Adam 更新按预先固定的噪声感知门限通过。全梯度 RMS 为 0.00515245，原版重复为 0.00468493。早层 k 参数的相对 Adam delta 噪声较大，不能将“通过”解释为所有参数误差低于 0.1%。原始逐参数结果保留。

完整的 [v2 源码差异](history/20261113/learned123-v2.patch)对应原冻结实验的 `snapshot/` 和 adapter `models.py`。该补丁用于查看与重建该实验候选，不能直接应用到当前包内默认实现。重建脚本会核对性能/Adam 回执中保留的 KOMP 源码哈希。

包含 tf32x3 UT 和 FLA RMSNorm 的第一组合未通过完整模型检查，未部署。v2 的 FP32 近共线 FP64 参考结果仍保留 `passed=false`；配对无新增失败检查通过，两类结论分别记录。v2 `block_fast.py` SHA256 为 `9ab491fcbc7b356aeb74d90f95dd1f85bcfac74b2fe52f33dd68f85ef98a456b`。

正式 32 卡迁移后曾测得前 82 步、排除首步的平均 1.691709 s/step；父作业采样为 2.857984 s/step。该观测约 1.6894×，与单卡受控 ABBA 分开报告。最新训练最终状态为 Aborted、预算未达 50B，见 [RECENT_RESULTS.md](RECENT_RESULTS.md)。[优化报告](history/20261113/komp-learned123-c128-opt-20261109/OPTIMIZATION_REPORT.md)保留各候选、全部失败与部署指纹。

## Recurrent inference

r1 的 8 类算子检查通过，Block 完整模型在 T31 首次 logits 检查失败。r2 诊断曾使用错误的 top1 计算；r3 修正为 argmax 并保留原错误回执。r3 的 native Block 开放块缓存仅有 CPU 状态性质验收，Value 路径拒绝；没有可报告的 r3 GPU 模型通过结果或加速比。

相关源码与固定验收计划见 [r3 README](history/20261113/komp-recurrent-inference-20261107-r3/README.md)。[远端 attempts 快照](results/20261113/remote-inventory.json)补充了此次整理时读取的历史回执。
