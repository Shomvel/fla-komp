# Exact-KOMP：第一阶段优化结果

2026-11-06。在单张 **H100 80GB** 上，使用真实 **339,966,848 参数模型、B16×2、T2048、step3072 checkpoint 和固定真实训练输入**，组合优化把暖更新耗时从 **5.2414 s 降到 4.4673 s**：时间减少 **14.77%**，吞吐提高 **17.33%**。峰值 allocated 从 **30.948 GiB 增至 35.242 GiB**，reserved 从33.557增至37.068 GiB。

这是完整模型单卡更新的受控 A/B，不是算子微测的外推，也不是8卡实测。原正式训练、batch、checkpoint、指标和监控未修改。8卡 DDP 与长期训练质量尚未验证；没有部署到正式训练。

**具体改动**

1. `candidate/geometry_cache.py` 与 `candidate/models.py`：每次 mixer checkpoint 独立缓存首次前向的 FP32 `p/h/r/n`；重放直接复用。缓存不跨层、microbatch、step共享，不绕过原 custom autograd backward；图释放后回收。形状、device、dtype和相关参数不符时拒绝重放。仅支持 eager Exact FIFO FP32；learned/FP64不缓存，torch.compile明确拒绝。默认关闭，研究runner用`--cache`打开。
2. `candidate/snapshot/komp_slot/exact_komp/recurrence.py` 与 `operator.py`：通过独立`--prune-gc`开关去掉未被 backward 读取的全局`gc` tape；kernel内仍计算和使用局部累计量。每次前向少分配/写入64 MiB，当前每卡更新的96次前向/重放合计少约6 GiB逻辑写入。没有改变cum-sum算术。

没有改变 FIFO、每token压缩、compute_chunk16、ridge、temperature、投影/状态精度、TF32x3、原解析backward、BF16兼容舍入、loss scaling或optimizer。完整差异见`candidate.patch`。保持默认关闭是为了保留对照和避免意外改变训练入口；启用后的实际路径已经通过本轮验证。

**同卡 ABBA 暖步结果**

所有进程恢复同一模型、Adam和RNG，先单列首步和3次warmup，再各测10次更新。输入每rank32×2049 IDs，两个B16微批，每更新65536有效token。每轮独立进程，顺序A1→B1→B2→A2；不把编译、数值抓取或profiler步骤放进速度统计。

| 轮次 | 路径 | 暖步中位数 | token/s，单卡 |
|---|---|---:|---:|
| A1 | 冻结baseline |5.250739 s|12,481|
| B1 | 几何缓存＋去gc tape |4.467429 s|14,670|
| B2 | 同上重复 |4.467114 s|14,671|
| A2 | 冻结baseline回测 |5.232038 s|12,526|

原始逐步数据在`gpu-results/bench-{a1,b1,b2,a2}/result-rank0.json`，汇总为`evidence/abba-summary.json`。A1/A2相差约0.36%，B1/B2约0.007%；17.3%收益显著大于本轮观测漂移，但这不是多机器或多次独立实验的统计置信区间。

整步包含真实网络forward/backward、checkpoint重放、阻塞H2D、clip和Adam；单卡没有DDP通信。数据使用真实固定batch反复更新，不计入正式训练token预算；不包含逐步stream/tokenization、周期validation/checkpoint I/O和整个6.8B训练过程。Adam保持checkpoint学习率，不模拟后续scheduler。不可直接把单卡速度乘8作为已测吞吐。

**收益来自哪里**

真实第0层B16T2048暖测：forward为28.070→28.051 ms，backward（含重放）为81.706→65.531 ms，层总时间109.826→93.612 ms。`gpu-results/layer-{baseline,cache}/kernels.txt`显示：

- geometry forward调用从2次减到1次，合计32.651→16.316 ms。省下约16.3 ms/层/微批，乘24层×2微批约0.783 s，与整步约0.774 s节约吻合。
- geometry backward仍约28.5 ms；`local_gradients_scan`仍约15.8–15.9 ms；maps两次合计约9.4 ms。它们是后续优先方向，不是本轮已消除的成本。
- 单层profiler包含CPU算子聚合和CUDA kernel事件，两类行有重叠，不能把整表Self CUDA逐行相加当作总时间。上述分项引用具体kernel行。

静态成本模型保留在`cost_model.py` / `evidence/cost-model.json`：FP32几何缓存194 MiB/层、24层4.546875 GiB；每卡更新省25,165,824个重复几何系统求解。实测整模型峰值增4.293 GiB，说明简单的逐层字节加法不能代替峰值测量。

独立开关探针（各一次10暖步，未做额外ABBA）：只开缓存 **4.46506 s**；只去gc **5.23864 s**。因此可观测速度收益来自缓存，未测出gc独立速度收益。只去gc的峰值allocated为30.886 GiB，较baseline少约64 MiB；缓存＋去gc比只缓存也少约64 MiB。不能把少6 GiB逻辑写入宣称为同等HBM流量或显存峰值节约。

完整模型profiler进一步确认geometry forward调用 **96→48**，CUDA时间 **1.554 s→0.783 s**；geometry backward约1.357 s、local_gradients_scan约0.765 s、maps约0.454 s是下一步主要kernel目标。clip+Adam约9.5–9.7 ms，当前不是首要瓶颈。单rank真实CPU loader重复读取同一cursor约0.110–0.115 s，首遍0.168 s，OS缓存未清空、8rank竞争未测。原始结果为`gpu-results/profile-*/kernels-rank0.txt`与`evidence/remote/input-profile-rank0.json`。

**数值结果：门槛未变**

20个CUDA算子用例通过：FP32/BF16，T=1/15/16/17/496/497/511/512/513/2048，包含输出、最终keys/state及全部输入梯度，覆盖partial chunk和NC=31/32路径切换。所有用例最差相对RMS约1.36e-5。门槛仍为FP32 .005/.005、BF16 .035/.04和原nearzero allowance；没有据此宣称解决近共线压力限制。

真实340M模型，对同一native FLA checkpoint做B16×2对B16×2；full每rank65536监督位置，tail为63616（32768+30848），两次loss都按总local valid归一化。逐tensor检查clip前gradient、clip+Adam后update，固定原RMS .035、normalized max .04或absolute1e-7，loss差<.005。

| 对照 | loss差 | 整模型grad相对L2 | 整模型update相对L2 | strict失败tensor |
|---|---:|---:|---:|---:|
| full：关闭优化的控制组 |0|0.4395%|0.4850%|0|
| full：组合优化 |0|0.4403%|0.4858%|0|
| tail：关闭优化的控制组 |0|0.4408%|0.4846%|0|
| tail：组合优化 |0|0.4394%|0.4841%|0|

组合优化full最差tensor RMS为grad2.656%、update2.815%；tail为2.543%/2.706%，均低于既定3.5%门槛。控制组也存在相近偏差，因此不能把全部误差归因于缓存；这些结果不是逐位一致证明。没有把既有旧batch/legacy轨迹失败重标为通过。

全部检查finite、模型/每个Adam状态slot/RNG恢复通过，eval前缀因果差为0。fixture生成时从每rank原loader cursor重放并验证IDs；所有恢复与临时更新都在私有目录。8rank replica检查尚未运行，单卡hash不能替代它。原始完整逐tensor结果见`gpu-results/*comparison.json`，摘要见`evidence/gpu-summary.json`。

另外，ABBA中的A1/B1从同一状态连续进行14次固定batch更新，最大loss绝对差为1.7619e-4。该观测保存在`evidence/repeated-input-loss.json`，它没有使用新的训练数据或heldout，不是长期质量验收。

**来源、可复现入口与执行边界**

冻结runtime manifest：`1dde0fbb5668f1105425de6774a4d188535b278819d3af3f1c569a987359f5ea`，658个文件原样复制到`baseline/`。原始静态研究与未经GPU验证的预期已存`archive-v1/REPORT.md`；当前结论以本报告和GPU证据为准。

固定输入fixture SHA：`f985ed5a9ab3d6ecd90ae73177df2a20341a3f07b5ecbdd088c391b10d2f7188`，从Exact已完成step3072、1,610,612,736有效token checkpoint复制。data manifest仍为`2c6f0fa997a87c85d2b7940c99051523751fb725d3d221b775ed6d894262aa0f`；tokenizer与排除集hash核验通过。每个result记录来源源码SHA、GPU、torch、CUDA与Triton版本。

镜像为原训练pinned digest，实际为H10080GB、Python3.12.3、torch2.10.0a0+b558c986e8.nv25.11、CUDA13、Triton3.7.1。远端独立目录`/gpfs/users/linlizhi/exact-komp-efficiency-research-20261106`；原始grad/delta二进制、trace与私有fixture保留在那里，本地同步结构化结果与profile摘要。

`README.md`给出复现和集成入口。不要直接运行复制的`train.py`/`run_fla.py`/migration脚本；它们只作为源码证据，原manifest也不是候选的生产放行回执。原有两项正式训练完全独立。

**首次失败与剩余限制**

首次作业在准备fixture时因缺少数据manifest停止，未进入数值验证。先静态检查所有入口依赖，再把data manifest/audit作为显式输入，增加无副作用的CPU依赖检查及回归；远端CPU检查确认253训练文件、tokenizer/排除集和全部checkpoint文件后才恢复验证。失败日志、部分fixture和分析保留在`evidence/startup-failure-analysis.md`。未放宽任何数值门槛。

本轮由用户明确授权新增1张H100、累计最多45分钟。首个失败作业记账1分钟，重试deadline缩为44分钟；没有提交8卡阶段。后续要推广到正式训练，仍需8卡同形状DDP验收与更长轨迹/heldout质量检查。本次通过只覆盖指定checkpoint、输入、形状和单卡链路。

**执行已结束**

单卡作业`job-272b90ec468f-20261106110439`已Completed，所有预定检查、ABBA、profile与独立开关探针结束。失败启动和成功作业的平台报告时长分别33秒、23分29秒，合计 **24分2秒，约0.401单卡小时**，含启动、不是SM活跃时间计量。两个研究作业均已按精确alias清理并确认不存在；原两个8卡正式训练保持同一identity且仍Running。见`evidence/resource-ledger.json`、`stage1-r2-terminal.json`和`formal-jobs-after.json`。

最后逐文件核对了实际GPU结果记录的baseline/candidate源码hash，以及原冻结runtime的658个文件；全部一致。当前可交付的是已通过单卡验证的隔离候选，下一步仍是单独授权的8卡验收；没有申请或启动该阶段。
