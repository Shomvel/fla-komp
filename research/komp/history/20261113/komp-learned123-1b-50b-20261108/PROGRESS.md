# 32 H100 Learned+123 1B / 50B

- 作业：`komp-learned123-1b-50b-20261108`
- ID：`job-e72852fdf8d4-20261108230824`
- 提交：2026-11-08 23:08:24 Asia/Shanghai；平台接收成功，Running 4/4 / 32 GPU。
- 新运行目录：`/gpfs/users/linlizhi/komp-learned123-1b-50b-20261108`。
- 源码 manifest SHA256：`5df4b9a9f7d0e2d42730e4c8e26214af3ac7e606b099403f431f0078fe935a27`。
- 当前阶段：32rank preflight。待取得 preflight passed、restore32/32 与 fresh 正式训练指标后更新。
- 配方和相对 Flame 的差异见 README.md / configs/experiment.json；Flame 原始参考副本及哈希在 evidence/。
- 旧340M下游评测保持原样；本作业不读取340M模型权重。

- 预检推进：实际实例化参数 1,003,181,056；32张H100 80GB与四节点源码哈希一致；tokenizer、算子FP64对照、全部8组options性质已通过；32/32 MODEL_READY。真实full/tail更新与完整恢复仍在执行。

- 首次预检失败：32rank均在save新增空间检查中触发shutil局部导入遮蔽的UnboundLocalError；正式训练0步。旧包冻结不改，修订在同级-r1目录，已完成作用域静态排查与实际save文件回归。
