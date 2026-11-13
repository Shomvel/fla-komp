# Learned+123 1B / 50B 每卡batch16恢复

用户要求每卡batch16重启；global512=32卡×16×累积1，seq2048和模型/优化器/LR/50B总预算不变。

- 原作业 job-874b71e8ade5-20261108232310 已安全暂停并Completed，四节点exit0。
- 父checkpoint step1068 /1,119,879,168有效token，模型/Adam及32rank输入/RNG文件完整，固定哈希见configs/resume-parent.json。
- 新alias komp-learned123-1b-50b-20261109-mb16，identity job-b054fc9711a2-20261109003121，2026-11-09 00:31:21提交32H100，WeaverRSI/proj-32。
- 源码SHA5d822987b3cfd39e27aa98923cb49217e86b1fa4b32bcef6ac99e011f148c1b9，模型/data/operator源码与父版本逐文件一致。
- 恢复契约回归执行实际resume函数，拒绝不匹配checkpoint路径、source/data、step/token、world、模型/每rank文件哈希或model源码变动。
- batch16全32rank预验收进行中；通过后丢弃trial权重并恢复父checkpoint继续正式训练。
- 最新状态由monitoring/collect_training.py采集；新正式steps从1069开始，不从0重训。

训练观测：{"captured_utc": "2026-11-08T16:34:25.517786+00:00", "phase": "preflight", "jobs": [{"name": "job-b054fc9711a2-20261109003121", "user_def_name": "komp-learned123-1b-50b-20261109-mb16", "status": "Running", "ready_status": "4/4", "namespace": "qiji", "job_creator": "linlizhi", "user_gpu": 32}], "step": 1068, "tokens": 1119879168, "target": 50000000000, "percent": 2.239758336, "loss": null, "loss_mean100": null, "step_seconds_mean100": 0, "global_tokens_per_second": null, "delta_steps": null, "delta_tokens": null, "validation_last": null, "checkpoints": [], "node_exits": {}, "restore_passed": 0, "parent_resume_passed": 0, "resume_parent_step": 1068, "resume_parent_tokens": 1119879168}
