# 2026-11-08 KOMP eval 断点恢复 r4g

用户授权重启三个中断作业。通过 Northjob 平台确认旧 s0/s1/s2 均 Aborted、0/1，残留实例仅 Failed，没有存活实例。项目 WeaverRSI、队列 proj-32 已核对；旧日志无 Traceback、OOM、CUDA/NCCL 错误或终止信号，平台事件为空，旧作业中断原因尚未确认。

恢复前保存 26,731 个完整 batch 的 SHA-256，共 132,969 条结果。仅迁移 Block、Block+123、learned、learned+123 的 12 个 worker claim；旧 claim 和证据归档在本恢复目录。已完成 KDA、Value、Exact+123、Exact partial 跳过。沿用原评测源码、固定分片、checkpoint、生成协议、镜像和每作业 8 GPU 配置。

| 分片 | Alias | Identity |
|---|---|---|
| s0 | komp340m-downstream-eval-20261108-r4g-s0 | job-555935e78289-20261108183216 |
| s1 | komp340m-downstream-eval-20261108-r4g-s1 | job-2f01a6119448-20261108183228 |
| s2 | komp340m-downstream-eval-20261108-r4g-s2 | job-573d9d01ad05-20261108183240 |

三个作业已 Running 1/1。运行验收、新 batch 与旧结果哈希核对见 verification-summary.json / recovery-audit.json；恢复前指纹见 before.json。collect.py 已切换至新作业。

18:36:54 验证快照：12/12 worker acceptance passed，无 failure；5/12 worker 新增 138 条结果，其余 7 个在首批长上下文生成中。26,731 个旧 batch 哈希和已完成模型证据全部一致。
