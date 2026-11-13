# Learned+123 1B / 50B 32 H100

当前作业：`komp-learned123-1b-50b-20261108-r1` / `job-874b71e8ade5-20261108232310`，2026-11-08 23:23:10提交，4节点已启动。

源码SHA：`3750920234e52a036297491de0958eb52ee5701a98216929ae4c4fb8d9b69aab`。配置：D2048 L24 H32 K64 M32 FF3072，实际参数1,003,181,056；Flame global512 seq2048 lr3e-4 warmup4000，mb4 accum4，47,684步精确50B有效token。

首轮作业已Aborted：save空间检查的shutil局部导入遮蔽，32rank同一UnboundLocalError，正式0step。r1移除重复局部导入，实际save逻辑回归复现旧错且修复通过，32rank文件和保留两个checkpoint已检验；全部任务Python局部导入静态排查通过。GPU全链路重新验收中。

首轮 trial full loss10.79470681399107，63.8343s，peak24.5525GiB；tail loss10.16086279646599，15.8932s，peak32.0266GiB。算子/8开关检查通过、replicas_identical=true，trial不计入50B。首轮恢复验收未完成，故不能用首轮代替r1完整preflight。

实时证据：evidence/monitor-*.json，GPU回执 evidence/preflight-block.json 与 restore-rank*.json，正式指标 runs/block/metrics.jsonl。旧340M评测未修改。

## 正式启动成功

- 2026-11-08 23:35：完整preflight passed，32/32模型/Adam/RNG/loader恢复回执及源码SHA一致。
- 当前 step 20，有效token 20,971,520/50,000,000,000，loss 9.167139，每步 2.863s。
- peak allocated 32.026 GiB/卡；global512，mb4×accum4，32 H100。
- 初期吞吐不含后续validation/checkpoint开销，完工时间需以稳态速度校正。
- 项目WeaverRSI、queue proj-32。首轮失败作业已保留终态和32份证据后精确清理，当前唯一新训练为r1。

训练观测：{"captured_utc": "2026-11-08T16:11:51.029869+00:00", "jobs": [{"name": "job-874b71e8ade5-20261108232310", "user_def_name": "komp-learned123-1b-50b-20261108-r1", "status": "Running", "ready_status": "4/4", "namespace": "qiji", "job_creator": "linlizhi", "user_gpu": 32}], "step": 747, "tokens": 783286272, "target": 50000000000, "percent": 1.566572544, "loss": 4.458953868597746, "loss_mean100": 4.603385394457728, "step_seconds_mean100": 2.970177818201482, "delta_steps": null, "delta_tokens": null, "validation_last": null, "checkpoints": [], "node_exits": {}, "restore_passed": 32}

训练观测：{"captured_utc": "2026-11-08T16:17:58.524259+00:00", "jobs": [{"name": "job-874b71e8ade5-20261108232310", "user_def_name": "komp-learned123-1b-50b-20261108-r1", "status": "Running", "ready_status": "4/4", "namespace": "qiji", "job_creator": "linlizhi", "user_gpu": 32}], "step": 871, "tokens": 913309696, "target": 50000000000, "percent": 1.826619392, "loss": 4.190258672460914, "loss_mean100": 4.336264564990997, "step_seconds_mean100": 2.963245465233922, "delta_steps": 124, "delta_tokens": 130023424, "validation_last": null, "checkpoints": [], "node_exits": {}, "restore_passed": 32}

性能对照：历史Flame GDN 1B 32卡实测约44k tokens/s/卡（参考路径、真实配置与10条metric见evidence/flame-speed-comparison.json）；当前KOMP约11k/卡，约4倍吞吐差。40小时是当前实现的实测估计，不能解释为1B/50B不可避免的耗时。mb4×accum4、mixer重算、无torch.compile、自定义KOMP算子与GDN的计算差异均存在，但尚未profile，不能断言各自耗时占比；未变更运行中的训练。

训练观测：{"captured_utc": "2026-11-08T16:23:32.555285+00:00", "jobs": [{"name": "job-874b71e8ade5-20261108232310", "user_def_name": "komp-learned123-1b-50b-20261108-r1", "status": "Running", "ready_status": "4/4", "namespace": "qiji", "job_creator": "linlizhi", "user_gpu": 32}], "step": 983, "tokens": 1030750208, "target": 50000000000, "percent": 2.061500416, "loss": 4.069405760616064, "loss_mean100": 4.138643392212689, "step_seconds_mean100": 2.9657001188211143, "delta_steps": 112, "delta_tokens": 117440512, "validation_last": null, "checkpoints": [], "node_exits": {}, "restore_passed": 32}
