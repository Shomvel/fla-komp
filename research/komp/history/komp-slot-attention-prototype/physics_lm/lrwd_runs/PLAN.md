# LR/WD 搜索与十倍扩容

三个架构均开启 short conv：GSA-slot KOMP、GSA、GDN。保持上一轮门公式、冻结 IEEE geometry、q/k/v 宽度4 depthwise causal conv+SiLU，不同时改变门参数化或 KOMP 算法。

## 搜索与选择

- 每个架构使用 seed731，搜索峰值 LR={2.5e-4,5e-4,1e-3} × WD={0,0.01,0.1}，共27个完整10M-token训练。三者搜索预算相同。
- 采用最后一步完整验证集 mean content CE（Depo/Brevo各占一半）最低的配置，平局按较低LR、较低WD排序。选择文件写入后锁定。
- 搜索代码只读取train/val，不调用test/confirm。选定配置后训练seed732，并且完成选择后才做test/confirm评估，不据此改选。
- 保留原来的warmup5%、cosine到峰值10%、残差输出LR倍率.1→.3、AdamW beta=(.9,.95)、clip1、batch64。只变峰值LR与非豁免参数WD；decay.bias/A_log/dt_bias始终WD=0。
- 每次完整1epoch、10,001,108唯一输入token；同seed初值、数据顺序一致。不按中途最低val挑checkpoint。

## 十倍模型与数据

- 由4层hidden128/H2/FFN256扩到15层hidden512/H8/FFN1536，head维度64、slot32、词表32000不变，untied embedding/head。
- 含卷积的参数量：KOMP 90,204,472；GSA 89,897,152；GDN 88,050,352。这里“十倍”按总参数量定义；embedding占比变化，非embedding参数不是精确十倍。
- 使用各架构在小模型验证集选定的LR/WD直接迁移，先各跑seed731、100M唯一token完整1epoch，共3个大模型run。这不是大模型自身调参后的最优结果，也不是单独模型或单独数据扩容消融。
- 100M语料包含原10M训练集，再追加约90M新token；按完整输入token序列SHA256去重，排除原始旧实验语料以及固定val/test/confirm。保留相同任务分布、题目难度、完整文档、padding192、不packing。
- 维持原val/test/confirm的内容与SHA以便比较。它们已在历史探索中使用过，不能称为从未查看的新测试集；本轮选参仍只使用val。
- 相同head维度/slot数保证每head记忆预算可比；更深更宽后的总记忆与计算成本另报。

## 验证和资源

复用已通过的卷积前反向/缓存检查和原kernel因果审计。新预检核对冻结源码、初值一致性、LR缩放与WD分组契约、小batch有限梯度。每个训练前后做13个截断点因果检查，并检查checkpoint安全回读、实际token数、所有梯度有限。

最多3张H100，每架构在固定GPU顺序执行候选。数据生成为同一作业内CPU进程，与小模型搜索并行。保留独立源码包、环境、参数、逐步日志、checkpoint与校验和。首次错误暂停该类验证做静态排查，不自动重试、不按失败结果追加搜索点。
