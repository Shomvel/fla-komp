# Learned+123 1B 下游评测

用户最终要求先续训到50B再评测。当前仅准备，未提交任何GPU作业。训练 step31744 的 33,285,996,544 token checkpoint 候选已归档到 evidence/partial-candidate-not-launched.json，未用于评测。执行前必须由 prepare_final.py 固定真正 completed 的精确50B checkpoint；runner 和 plan 生成也强制此预算。

范围为 NIAH single1/2/3 × 1024/2048/4096/8192，各500条；PhoneBook32/64/100/128/150/200/256，各1000条；SWDE1111、FDA1102、NQ3157、DROP2087。总计23分项20457条。保留旧340M评测已审计的请求、batch形状、tokenizer、生成及评分函数；只改变新模型路径/固定checkpoint及8分片。AST核对见 evidence/inference-parity-static.json。

FP32权重+BF16 autocast，greedy，完整前缀重算，无padding、无cache。同组严格相同有效长度。NIAH生成上限128，PhoneBook20，BASED48；EOS与原任务stop保持一致。训练context2048，4096/8192为外推。PhoneBook使用历史single_break，prompt约80–244 tokens，不视为长上下文测试。

单Northjob作业8 H100，WeaverRSI/proj-32，8个worker各持有一份模型，分配固定互斥batch。不重跑旧340M评测，不修改训练源码/checkpoint或旧数据。所有结果位于本独立目录。先验证跨GPFS客户端atomic mkdir/link，再加载模型执行代表性真实prompt数值与生成验收；所有worker通过后以实际batch落盘确认启动。

本机重新采集：`python3 collect.py`。各worker日志在远端 logs/worker-N.log；每worker acceptance/failure/progress/complete位于 sharded/evidence-v2。checkpoint、source、requests、protocol与plan哈希随结果记录；只对完整分项报告分数。

## 训练完成后的自动衔接

`watch_after_training.py` 在 BP SSH 主机持久运行，认证仅在远端内存读取。绑定 watcher/config.json 中的续训 alias+identity；每60秒检查。只有平台 Completed、4节点exit0及精确50B/47684完整checkpoint同时成立，才调用 prepare_final.py 固定所有哈希。

提交前独占创建 watcher/submission-attempt.json 并fsync。API响应不明时只查询确切alias，不重发POST；已有alias必须匹配本目录job.json，否则停止。8 GPU worker先写互斥claim/publication probes，由SSH客户端验证后释放模型加载门槛。随后持续采集acceptance、failure、完整batch和平台状态；任何失败留存watcher/failure.json，不自动重启或重复提交。

CPU模拟验证见 evidence/handoff-guard-tests.json：拒绝partial、训练失败、错identity和非零节点退出；重复提交与模糊intent不产生第二次POST。真实GPU数值验收将在50B后触发，当前尚未执行。

持久 watcher 已于北京时间2026-11-10 17:22启动，BP SSH PID 53835；绑定训练 job-35a0f599ed70-20261110172046 / komp-learned123-1b-50b-20261110-resume1。首次独立复查为 waiting_for_50b_training，训练平台 Running 4/4；评测尚未提交，没有submission intent或job receipt。启动证据见 evidence/watcher-started.json。
