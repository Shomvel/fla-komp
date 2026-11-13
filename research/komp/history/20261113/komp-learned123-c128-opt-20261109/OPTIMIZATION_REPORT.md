# Learned+123 1B：C128 / mb16 前后向优化

当前状态：优化版已正式在32张H100上接续，32/32 rank恢复通过，实际训练连续推进。最终数值/Adam/图隔离检查和同卡ABBA测速已完成。

## 固定条件

D2048 L24 H32 K64 M32 FF3072，1,003,181,056 参数；seq2048，microbatch16，accum1，global512。模型、数据流、AdamW、LR/warmup 和精确 50,000,000,000 有效监督 token 预算保持一致。固定已训练 checkpoint010240、输入 ID SHA `da6016e4cd3b3100d0ec566288289b2ee530e6bd57222bc9837c8548132a2abe`。同一 H100 / 固定镜像，profile 和计时分开。

## 验收口径

用户明确授权对齐上游 GDN BF16 测试：输出/最终状态相对 RMS≤0.005；dq/dk/dv/dh0≤0.008；dg/dbeta≤0.02；绝对最大误差≤1e-6 也可通过。必须全部有限。整模型 CE 绝对差≤0.001，检查全部 579 参数梯度、原版重复运行噪声，以及父 checkpoint Adam 更新。早期更严门限下的失败完整保留，不能写成从未失败。

## 已完成候选

| 候选 | 实测 | 结论 |
|---|---|---|
| 原版 | operator64.202ms；完整训练步2.59250s | 固定基线 |
| 槽位衰减反向直接求和 | operator38.119ms；完整步1.96921s；ABBA原版2.5922s/候选1.9681s | 保留；前向逐位一致，8组算子和全部参数梯度检查通过 |
| UT tf32x3 | operator37.410ms；dq0.09749%、dk0.35028%、out0.02417% | 按用户GDN口径算子通过；包含此项的全模型组合失败，与RMSNorm一起撤下，未部署 |
| 线程布局 | 单项compose8=35.870ms、ut_bwd8=32.540ms、keygrad8=36.786ms、proj_bwd8=37.458ms | 有界组合实测，不能相加声称收益 |
| 两阶段UT反向 | operator35.569ms，最大梯度RMS2.38e-5 | 与8warps组合对照后选择 |
| 槽位prefix简化 | operator38.410ms；整模型compose回退 | 排除 |
| FF融合 | 宽值域/真实BF16大shape通过；同一算子下loss一致，全部579参数梯度通过；省4.63GB | 保留以降低激活显存；原017含慢slot的2.01344s不能归因于FF本身 |
| FLA RMSNorm | 两个真实shape最大梯度RMS6.58e-7；宽64前后向1.652→0.659ms，宽2048为0.874→0.687ms | 包含此项的整模型组合失败，最终保留PyTorch RMSNorm |
| 分离UT前向 | 4warps37.637ms、2warps35.870ms | 独立备选；无进一步扩展 |
| 重算缓存 | 完整步1.76280s；UT48→24次，loss逐位相同；独立使用peak72.9455GB | 双图/保留图重复backward通过，8轮释放后1,100,800bytes恒定；与FF省显存组合 |

016首次全模型对照同时改变slot/decay/FF，loss三次相同，某参数梯度超过原版一次重复噪声推导的门限。先停止分支并静态检查，补完整结果汇总和相同算子下的FF隔离对照。029隔离对照通过：global梯度RMS0.0048544，对照重复0.0046656；没有提高该检查门限。

## 正式迁移要求

最终组合通过后，当前作业在步边界保存完整model+Adam与32rank RNG/loader checkpoint，4节点exit0后再续训。核对父checkpoint、来源SHA、每rank SHA与剩余token预算；新作业必须从父step+1推进，实际32卡速度与单卡probe分开报告。

## 最终组合的整模型拒绝证据

第一组合包含tf32x3 UT与FLA RMSNorm，算子GDN通过、整模型CE差-0.00014043通过，但全梯度RMS3.457%明显高于原版重复0.450%。560/579梯度、445参数更新、572 Adam delta超出门限。所有JSON保留在evidence/final-model-adam.json，未部署。未把单算子门限推广为整模型保证；静态一次性排除这两项改变前向算术的替换，v2保留原IEEE UT和PyTorch RMSNorm，不单独认定哪一项是根因。其余主要优化收益仍保留，按相同门限重新检查新组合。

生命周期检查037实际证明：16.78MB缓存forward后仍在，backward完成后即释放到1536B，即使loss变量仍保留；释放loss后1024B。无需额外修改训练循环，正式train.py保持原样，以未改动源码SHA核对。

## 正式采用候选v2

保留IEEE UT与原PyTorch RMSNorm；采用槽位衰减反向直接求和、C128实测8warp布局、BF16中间舍入兼容的FF融合、单checkpoint图内不可变UT/dictionary缓存。生产/尾块算子最大RMS1.4023e-6/4.0414e-7；完整1B CE与原版完全相同2.704592466；全部579梯度与strict checkpoint Adam参数更新/真实delta全部通过（全梯度RMS0.00515245，原版重复0.00468493）。源码固定，完整测速与32卡迁移待完成。

源码静态差异核对：snapshot相对父包仅block_fast.py、modeling_komp.py及新增rounded_swiglu.py三项；adapter models.py增加图内cache上下文。正式train.py（SHA c5d915d3546467f8fd0996670503b5ef125d7f82f30eb898ebf5c8e50fd85b12）保持不变，未引入额外loss图释放改动。

独立FP64参考：随机FP32、随机BF16与近共线BF16原版/候选都通过；FP32近共线有两个字段原版和候选都未达旧绝对参考门限。raw final-v2-reference.json 保持passed=false。配对逐字段检查确认没有新增失败，全部absolute/RMS误差不超过原版*1.02+1e-6，ranking 3/3通过；单独final-v2-reference-acceptance.json记录无回归passed，不把既有局限写成全参考通过。

最终同卡完整训练步ABBA：原版2.59485868/2.59509472秒，候选1.46921275/1.46952212秒，吞吐1.76605倍（耗时减少43.38%）；peak68,313,894,912bytes（63.62GiB），单卡不是32卡实测。源算子SHA9ab491fcbc7b356aeb74d90f95dd1f85bcfac74b2fe52f33dd68f85ef98a456b。正式候选configs/optimization.json固定全部差异源码和验收证据SHA。

已提交正式opt1：job-757679e41f6c-20261109101556，4×8H100，source manifest SHA f513fda5a095ade64f330175bf51c14b93a3ad821f7ae7efcaf0c4505adb34ce。父12922/13,549,699,072 tokens完整checkpoint，4节点exit0后才提交，不并行占64训练卡。额外1H100开发job已exit0、delete、再次查absent。

Adam比较中的早层k参数原版原子加法重复运行自身有较高相对噪声：最大实际Adam delta RMS候选0.22606，对照原版重复0.21593；按原先固定的噪声感知门限通过。不能把“全部通过”解释为每参数绝对RMS都低于0.1%。整体loss逐位一致；原版重复噪声的相邻风险与完整逐参数数据均保留。

## 正式32卡结果

2026-11-09 10:21:44，opt1已从12923正式跑到12995，73步，13,626,245,120有效tokens；32rank严格model/Adam/RNG/loader恢复通过，4/4 Running且无退出/非有限metrics。去前5步后68步平均1.70102秒，中位1.65240秒，相对父100步2.85798秒约1.68倍吞吐；剩余纯训练估计16.4小时（另加validation/checkpoint开销）。显存allocated约67.03GiB，reserved70.46GiB。下一正式validation/checkpoint为13312，父12288验证CE2.404324不冒充新作业验证。

小时跟踪：Codex automation_update接口持续Transport closed，未手工修改automation.toml；原每小时计划保持。已在原collector设置明确successor-run.json并实际运行转发验证到opt1，原evidence与PROGRESS记录新identity及实际metrics，避免继续只报父Completed。转发首次执行缺shutil import已修复并真实重跑通过，新采集入口为opt1/monitoring/collect_training.py。

最终新采样10:21:58：step13004 /13,635,682,304 tokens，连续82步；去首步81步平均1.691709秒，相对父2.857984秒为1.689406倍吞吐（耗时减少40.81%），619,832有效tokens/s；剩余纯训练16.30小时。loss当前2.34939、82步均2.38600。正式32卡source SHA和回执均核对，小时采集入口转发已实际执行成功。
