# 24 GPU 评测并行度诊断

结论：并行worker数足够，GPU实际利用不足；当前不建议继续加GPU。

采集时间：2026-11-07T15:35:08.018755+08:00

三个作业均Running、各8GPU，分布node050/node118/node119。24卡当前利用率平均18.35%，近15分钟60点/卡平均19.90%。GPU0和GPU6的近15分钟均值分别约66–72%和77–81%，其余18卡只有0.15–5.8%。GPU index与逻辑worker的对应依据容器CUDA_VISIBLE_DEVICES，未独立查询GPU UUID映射。已验证API没有提供显存数据，本次未测显存。

24个worker近15分钟全部有新增结果，无失败，也没有因任务完成而闲置。GPFS日志最后进展年龄为2–364秒；长间隔集中在KOMP系列。

| 模型 | 三分片15分钟新增样本 | 当前阶段 |
|---|---|---|
| kda | 359 / 430 / 361 | niah_single_3_2048 |
| value | 187 / 183 / 195 | niah_single_1_2048 |
| block | 653 / 678 / 669 | phonebook_64 |
| block123 | 653 / 678 / 669 | phonebook_64 |
| learned | 348 / 342 / 675 | phonebook_32, phonebook_64 |
| learned123 | 348 / 357 / 392 | phonebook_32 |
| exact123 | 28 / 34 / 23 | based_swde |
| exact_partial | 26 / 34 / 32 | based_swde |

样本数跨任务不可直接比较；同模型分片差异显示Learned有约2倍不均，其余多在4%–48%区间。尚未出现完整分片提前结束造成的GPU闲置。

| 实际任务 | Batch形状 | 完成batch耗时 |
|---|---|---|
| KDA NIAH8192 | 114batch中104个batch4 | 中位13.81秒，最大13.88秒 |
| Value NIAH2048 | 49batch中46个batch8 | 中位41.11秒，最大41.84秒 |
| Exact123 SWDE | 61batch中41个batch1、16个batch2 | 中位1.51秒，最大346.25秒 |
| 原Exact SWDE | 61batch中37个batch1、18个batch2 | 中位36.41秒，最大210.88秒 |
| Block PhoneBook32 | 137batch中118个batch8 | 中位0.71秒，最大236.14秒 |

这些耗时包含冷编译和整个生成过程，不能全部解释为GPU计算时间。低util、短prompt batch8常态不足1秒而偶发数百秒，与新shape冷编译/主机侧等待一致；仅凭监控无法细分CPU编译、GPFS缓存IO和Python调度各自占比。

不加GPU的改进优先级：

1. 保留原batch与结果，减少相同KOMP kernel在不同worker私有cache内的重复编译。需核实Triton cache跨节点并发安全后才能共享；可先只读统计重复cache key。
2. 保留batch形状，按预计耗时分配或整batch动态领取，改善Learned等分片不均；更改前验证结果与恢复契约。
3. 查明Exact SWDE batch1/2路径的长编译间隔，再优化主机侧准备和编译。直接再加GPU会复制更多冷编译。
4. 较长期再验证可缓存decode或安全的变长批处理。当前每token重算完整prefix；原padding曾数值不一致，不能直接开启padding或盲增batch。

本次仅只读观测，未提交、停止、重启或修改运行源。
