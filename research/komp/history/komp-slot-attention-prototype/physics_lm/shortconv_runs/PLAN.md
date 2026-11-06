# Short conv 配对实验

预定 12 runs：KOMP GSA-slot / GSA / GDN × plain / conv × seeds 731/732。

- 使用上一轮冻结训练源码，KOMP 原 IEEE geometry 和串行 route；不混入后续性能优化。
- conv 组在三条 q/k/v 线性投影后分别加 FLA Triton causal depthwise convolution，宽度 4、无 bias、SiLU。plain 组保持原投影，所以本实验测量标准 conv+SiLU 模块的整体收益，不能将收益全部归于卷积。
- 新卷积使用 FLA 默认初始化，按 seed/layer/projection 命名随机种子；三种架构的卷积初值完全相同。有/无卷积组全部原参数初值相同，共同骨干初值跨架构相同。每模型增加 6144 个参数，以及每层三条投影的卷积缓存；不宣称总状态预算仍完全相等。
- 沿用各自门公式：KOMP GSA-slot 为 logsigmoid/8、独立 beta；GSA 为上一轮 logsigmoid/1 slot 门；GDN 为上一轮 sigmoid scalar gate/beta。不是原生完整 GSA/GDN 网络复现，也不同时改成原生 gate 配方。
- 统一使用上一轮 decay 实验 optimizer：decay.bias/A_log/dt_bias 不施加 weight decay，其余 WD=.1。因此 GSA/GDN 的 plain 组必须重跑；历史结果只作背景。
- 四层 hidden128、H2、D64、M32、FFN256，10,001,108 非padding唯一输入token；每 run 一次完整 epoch、2231 steps、batch64、padding192。数据顺序 seed12345，固定 warmup5%+cosine、peakLR5e-4 和残差输出 LR .1→.3。
- 最后一步 checkpoint，完整 val/test/confirm 评估；固定 confirm 每任务128例自由生成。Brevo 合法性重新从 prompt 图边求祖先闭包与拓扑序，不依赖 gold 答案集合。
- 预检 FLA 卷积 FP32/BF16 前反向对 Torch 左padding reference，未来输入梯度为0、后缀干预、流式缓存对齐；六个模型因果前缀、16步学习、卷积梯度和初始化一致性。12个最终 checkpoint 再验13个因果截断点与安全回读。
- 最多三张 H100，每架构同GPU串行执行四个run；停止于首次失败，不自动重训。隔离 job/root，完成保存证据后精确清理本次 job。
- 两seed仅为探索；不按确认集选超参数、不扩大扫描、不用训练耗时替代同卡算子benchmark。
