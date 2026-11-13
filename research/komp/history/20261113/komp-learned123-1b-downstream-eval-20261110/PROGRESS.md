# Learned+123 1B 下游评测

采集时间 UTC: 2026-11-10T09:27:45.539082+00:00
进度: 0/20457 条，0/23 完整分项。

独立 8 H100 / 8 分片，WeaverRSI / proj-32。具体 checkpoint 和 token 预算固定于 evidence/checkpoint.json。
沿用历史同一批 20457 条请求、相同 batch 形状与生成/评分函数。NIAH single1/2/3 × 1024/2048/4096/8192；PhoneBook 7 档；SWDE/FDA/NQ/DROP。
PhoneBook 为历史 single_break，prompt 约80–244 token；BASED cap2048。FP32 权重+BF16 autocast，全前缀重算，无 cache。

| 分项 | 进度 | 完整分数 |
|---|---:|---:|

自动衔接状态：waiting_for_50b_training

作业:
