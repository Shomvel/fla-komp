# 2026-11-08 s1 断点恢复 r4f

用户授权“s1 重启”。旧 r4e-s1 已核实 Aborted、0/1、无实例；平台事件为空，旧日志未见错误堆栈/OOM，中断原因未知。

新 alias：komp340m-downstream-eval-20261108-r4f-s1
新 identity：job-7551a320be07-20261108145549
项目/队列：WeaverRSI / proj-32

恢复前保存 8047 个完整 s1 batch 的 SHA-256，共 39395 条结果。只恢复 block123、learned、learned123、block 四个 s1 worker；已完成模型跳过，s0/s2 claims 纳入不变性核对。固定镜像、评测代码、计划、checkpoint 和生成协议沿用。

最终状态及新增结果见 verification-summary.json；旧文件哈希与各 worker 验收见 recovery-audit.json。collector 已使用新 s1 alias。原每小时自动化当前 PAUSED，已更新作业信息并保留暂停状态。
