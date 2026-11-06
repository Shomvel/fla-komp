# Exact KOMP Triton 验收结果

已实现每 token 更新、每 token 压缩的 FIFO / learned KOMP，包含解析反向。计算块长 16/32 不改变算法 C=1 语义。

## 数值与训练验收

H100 80GB；首轮 22 项、完整 293 项，共 315 项通过。基准为原冻结逐 token 实现以 FP64 执行。

| 输入精度 | 检查中的最大 RMS 相对误差 | 最大归一化绝对误差 |
|---|---:|---:|
| torch.float32 | 2.822869e-06 | 4.1261785e-06 |
| torch.bfloat16 | 0.0026063541 | 0.0027503069 |

上述误差包含输出、最终状态/字典与全部输入梯度；归一化绝对误差的分母为 max(1,max(abs(reference)))。投影归一化、Gram 和 solve 使用 FP64，状态与块内乘法使用 FP32/tf32x3；BF16 表示 q/k/v 输入及输出精度。

覆盖 M32/M64、非整块/非二次幂小尺寸、近重复 key、强遗忘、scalar/per-slot decay、learned 同分/拒绝新原子、两种计算块长、流式续算与因果干预。未来输入干预的前缀差严格为零。

优化器对拍使用含 GDN-slot、learned score 门、可训练字典和 q/k/v 的小模型；AdamW 两组各 12 步。

| 规则 | 初始 loss | 最终 loss | 最大参数轨迹差 |
|---|---:|---:|---:|
| FIFO | 0.38645625 | 0.31559625 | 8.3684921e-05 |
| learned | 0.44374764 | 0.36476746 | 5.9604645e-08 |

## 单卡算子测速

B=1、H=2、M=32、K=V=64；输入 FP32、投影 FP64。CUDA event，排除编译，2 次预热后 5 次测量取中位数。包括接口布局转换、分配和 autograd 调用，不是完整模型训练吞吐。

| 规则 | T | 计算块 | 前向 ms | 前向＋反向 ms | 增量峰值 MiB |
|---|---:|---:|---:|---:|---:|
| FIFO | 128 | 16 | 0.578 | 1.561 | 7.81 |
| FIFO | 128 | 32 | 0.576 | 1.510 | 7.81 |
| FIFO | 512 | 16 | 0.642 | 2.043 | 30.82 |
| FIFO | 512 | 32 | 0.712 | 2.021 | 30.82 |
| learned | 128 | 16 | 0.598 | 1.526 | 7.81 |
| learned | 128 | 32 | 0.566 | 1.484 | 7.81 |
| learned | 512 | 16 | 0.714 | 1.999 | 30.82 |
| learned | 512 | 32 | 0.770 | 1.991 | 30.82 |

## 使用与限制

现有项目入口：`from komp_slot.exact_komp import exact_komp`。独立验收包入口：`from exact_komp import exact_komp`。两份源文件相同，不修改原有 operator。

这版证明了完整规则及训练梯度可运行。选择仍有独立时间预处理，反向系数仍有逐 token 扫描；当前还保存 O(BHT M²) 的 FP64 几何量与 O(BHT MV) 的状态，不应直接当作大模型最优实现。没有与 GDN / C128 在相同形状下测量速度，也没有进行新一轮 PhysicsLM 质量训练。

完整环境、逐项数值结果、训练轨迹及源代码哈希在 evidence/。

现有项目包的 GPU 导入验收通过，包含输出、最终字典、最终状态及有限反向梯度；见 evidence/integration.json。项目副本与 GPU 验收源码的哈希一致。

测试作业 komp-exact-triton-20261105-r1（job-53ea38bb9434-20261105140137）已记录 Completed 终态并删除，精确 alias 复查返回空列表。收据：evidence/job-terminal.json、evidence/job-delete.json、evidence/job-absent.json。
