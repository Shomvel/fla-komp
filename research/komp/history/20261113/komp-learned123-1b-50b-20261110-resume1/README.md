# Learned+123 1B：续训至50B后评测

2026-11-10用户明确要求：先续训到50B，再评测NIAH1/2/3、PhoneBook、SWDE、FDA等。

旧opt1在2026-11-09 19:42（北京时间）保存step32768前因GPFS可用空间不足20GiB而退出。最后完整checkpoint为step31744/33,285,996,544 tokens。新作业恢复该完整模型、Adam、32rank loader/RNG状态；重算未保存的1024步，保持50B总有效token预算。

- Northjob alias：`komp-learned123-1b-50b-20261110-resume1`
- identity：`job-35a0f599ed70-20261110172046`
- 项目/队列：WeaverRSI / proj-32
- 4节点 × 8 H100，microbatch16/accum1/global512，seq2048
- 源码SHA：`1970918eed171dd8956297b3545df550a13a995864706f5793d1ecc6be098add`
- 原模型、优化器、训练循环、算子、数据、超参数保持一致；仅更新启动路径、作业名称及父checkpoint固定回执。
- 远端根目录：`/gpfs/users/linlizhi/komp-learned123-1b-50b-20261110-resume1`

本机采集入口：`python3 monitoring/collect_training.py`。状态快照保存在evidence/training-latest.json，真实恢复以32个parent-resume回执和step31745起的新metrics为准，平台Running不能代替数值推进。

后续评测目录：`/Users/shom/Downloads/komp-learned123-1b-downstream-eval-20261110`，远端使用同名目录。该目录watcher只在本训练精确50B、final checkpoint完整、4节点exit0、平台Completed后提交8GPU评测，覆盖23分项20,457条。禁止使用partial checkpoint。查看评测目录README和watcher/status.json获取衔接状态。
