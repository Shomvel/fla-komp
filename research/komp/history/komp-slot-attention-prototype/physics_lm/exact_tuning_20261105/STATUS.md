# Exact-KOMP 首轮调参

用户已授权执行。18 个候选任务已上传并原子接入现有两组 8 H100 队列，不增加 GPU 分配。

- Depo2：job-0f8389cb5b94-20261105175129，9 × 10,001,214 训练位置。
- Brevo2：job-f0df20da162b-20261105175132，9 × 10,000,577 训练位置。
- 合计 180,016,119 个训练位置（不包括独立预检/验证计算）。
- 顺序：七模型 DDP 预检 → 每组 FP64 数值门禁 → 9 个 Exact 候选 → 原七模型 100M 对照。
- 运行目录：`/gpfs/users/linlizhi/physics100m-exact-20261105/exact_tuning_20261105/`。
- 训练候选：baseline，alpha=.995/.999，beta=.1/.4，temperature=.25/1，ridge=1e-5/1e-3。
- 共同配置：100,526,392 参数、M32、FIFO、逐 slot 衰减、conv4、FP64 geometry / FP32 state、算法 C1 / 计算 C16。
- AdamW、WD=.1、任务 LR 与原配方相同；每次使用独立10M horizon的5% warmup + cosine schedule。
- 全局 batch=128，固定模型初始随机种子731和数据顺序12345；8卡分片损失按全局监督位置归一化。
- 每20%进度评估完整验证集，按最终验证CE在任务内排序；不看test/confirm选择。
- 每100step或验证点保存模型、optimizer、各rank RNG；STOP在optimizer边界保存后退出。

本文件记录接入状态，不代表数值验收或训练已经通过。完整结果由 `report.py` 根据实际 summary/metrics 生成。

进度：7/7 DDP预检通过；首次数值门禁FP32近重复key梯度失败，已暂停作业。现统一使用已有FP64 geometry重验，不降低阈值。尚无完成的短训练。
