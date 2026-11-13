# 2026-11-08 评测断点恢复 r4e

用户授权重启评测。旧三个作业经平台核实均 Aborted、0/1、无实例；对应旧日志未见异常堆栈或 OOM，平台事件为空，中断原因尚未确定。

恢复前保存 23,191 个完整 batch 的 SHA-256，共 113,573 条结果。仅迁移四个未完成模型的 worker claim，旧 claim 和证据归档保留。已完成 KDA、Value、Exact123、Exact partial 跳过。评测代码、分片计划、模型 checkpoint 和生成协议保持原指纹。

项目 WeaverRSI，队列 proj-32，沿用固定镜像与原 GPU 索引，每个作业 8 GPU，仅启动 index 1、3、4、5。

| 分片 | Alias | Identity |
|---|---|---|
| s0 | komp340m-downstream-eval-20261108-r4e-s0 | job-e36dbf1dc596-20261108074146 |
| s1 | komp340m-downstream-eval-20261108-r4e-s1 | job-443f62846225-20261108074154 |
| s2 | komp340m-downstream-eval-20261108-r4e-s2 | job-ab960a56ab90-20261108074203 |

三个作业已进入 Running；运行验收见 recovery-audit.json，要求 12 worker 全部验收通过并各有新增完整 batch，所有旧 batch 与已完成模型证据哈希不变。

现有每小时跟踪已切换到 r4e 三个作业；训练保持原任务运行。

07:52 运行快照：三个作业均 Running 1/1，12/12 worker owner 匹配且 acceptance passed，无 failure。9/12 worker 新增 28 个完整 batch、52 条样本；learned-s1、block123-s2、block-s2 仍等待首批，审计保留 pending_runtime_progress。23,191 个旧 batch、冻结源码及已完成模型证据哈希全部不变。详见 verification-summary.json 与 recovery-audit.json。
