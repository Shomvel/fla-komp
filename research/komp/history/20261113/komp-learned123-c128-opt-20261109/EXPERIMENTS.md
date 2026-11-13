# C128 / 1B / mb16 优化记录

固定工作负载：Learned+123，D2048 L24 H32 K64 M32 FF3072，seq2048，per-rank mb16，BF16 attention / FP32 projection。正式训练继续，不改变 global512、优化器、学习率、样本流或 50B 有效监督 token 预算。

基线 checkpoint010240（10,737,418,240 tokens）在 GPFS 硬链接固定，防止 retention 删除。相同已训练权重与固定 validation IDs；profile 与正式计时分开，单张 H100 80GB 同卡比较。

| 方向 | 假设与实现 | 验证与采纳条件 |
|---|---|---|
| 衰减反向 | C128/M32 one-hot 槽位映射不必做 IEEE dense GEMM，改直接按槽求和 | 算子 64.202→38.119 ms；8组检查通过，前向逐位一致；完整模型 2.5925→1.9692 s，继续检查全参数梯度 |
| 前向 UT | 非对角 32×32 FP32 IEEE 乘法改为 tf32x3 Tensor Core；保留 Gram 精度 | 用户明确要求对齐 GDN：BF16 out/state RMS≤.005，dq/dk/dv/dh0≤.008，dg/dbeta≤.02；整模型 loss 差≤.001，检查全部参数梯度与 Adam 更新 |
| 线程/矩阵布局 | C128 比历史 C64 tile 更大；8 warps 或更窄 key block 可减少 spill | 有界独立 sweep，每项单独计时；仅合并有效项并重验组合 |
| 反向 UT | 两阶段降低同时存活的 C×C 矩阵；复用最后一次使用后的 DWV 存 FP32 dT | 不增加显存；生产/尾块梯度对照与实际算子计时 |
| 通用槽位衰减 | C%M=0 时环槽和物理槽顺序相同，去掉重复 prefix scan；先在 C×M 上选对角/末行 | 前后向都需数值检查；不能只凭少 kernel 或少指令认定更快 |
| FF 激活融合 | 保留 BF16 SiLU 后舍入再乘的原语义，前后向融合，少存一个 FF 激活 | 专门舍入/宽值域测试、全部 1B 参数梯度对照、完整步速度与显存 |
| 归一化 | 原 nn.RMSNorm 前后向约100ms；以参数/eps/state键兼容的 wrapper 调用现有 FLA Triton RMSNorm | 两个真实行宽的 FP32 数值/梯度及单项耗时，再测完整模型；慢则弃用 |
| 激活重算缓存 | 仅在前述结果明确后评估；可利用 FF 节省的显存缓存不可变几何或 UT | 按实际剩余热点和峰值显存判断，避免先增加复杂性 |

开发首次脚本替换误改相似代码段导致 IndentationError，已停止队列、静态检查全文件、恢复不相关 chunk_bwd，compileall 全树通过再恢复。首次失败与修后结果分别保存。

最终从新暂停 checkpoint 恢复 model、Adam、32 rank RNG/loader；固定父来源和 SHA，仅在完整验证及有用整步收益后迁移。不能把开发单卡结果写成正式 32 卡吞吐。

2026-11-09 后续证据：

- 用户明确接受所见约0.1%的 dq 差异后，UT 候选验收改为相对 RMS≤0.001；但完整检查发现 tf32x3 的 dk=0.3503%，bf16x6 的 dk=0.2958%，均未满足该门限。未将只见到 dq 的单项结果冒充全部梯度验收，未写入正式训练。
- IEEE+decay 同卡独立 sweep：base38.0547ms，ut_slot8=40.5368（弃），compose8=35.8702，proj_bwd8=37.4577，ut_bwd/out_bwd8=32.5397，keygrad8=36.7862；keyblock32/16和mapgrad64没有收益。组合仍需再测，不相加宣称整步收益。
- slot prefix/diagonal 简化候选38.4103ms，比decay38.1193慢，暂不纳入最终组合。
- 增加唯一一个替代 UT 方向：同一非重入 checkpoint 内保留最初计算的 immutable TT，保留全部 IEEE 计算，通过省掉反向重算 UT 提速；预计保存张量+6GiB。只在 learned 路径复用，FIFO 的UT还会写GCP，因此不跳过它。复用现有 context 生命周期，不全局缓存，不跨层/微批/调用共享。正在测实际显存和整步耗时。

- 用户后续明确“0.3%可以接受；对齐GDN”，因此按上游GDN BF16分项门限复验，025 UT production+tail全部passed，dk0.35028%处于.8%梯度门限内。旧门限失败证据保留；验收标准变更来自用户明确授权。
- 016 FF整模型首次检查loss三次均2.704592466，首个超限参数attn_norm.weight梯度RMS.009688 vs baseline repeat.001643。该比较同时改动slot/decay及FF，先停止FF分支并静态检查，改为完整汇总及相同算子下隔离FF对照，不改变原梯度门限。
