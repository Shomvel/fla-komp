# 原生 FLA GSA2 补充对照

用户要求加入“查找 gated Oja rule 实现”会话里的GSA2，并直接使用FLA layer。固定FLA commit `035f54b64e22abdb640725b3cf13db47d33692c5`，调用未修改的`fla.layers.gsa2.GatedSlotAttention2`；适配器只提取返回tuple的输出，不重写投影、门、卷积或递归规则。

- 骨干与此前三组相同，非mixer参数按同样命名种子初始化。mixer保留FLA layer构造器的原生初始化，用独立的每层种子；不宣称其投影/门初值与此前手写mixer匹配。
- 显式设置hidden/head数/head维度与骨干一致、expand_v=1、chunk模式；使用最小支持的64slots。其余默认：level2_axis=head、w_rank=head_dim、q/k/v及w四条short conv、宽度4、SiLU、原生Oja2/GDN2 decay及erase/write门。
- 每head双记忆共8192个状态元素，约为此前三组4096的两倍；不宣称等状态预算。GSA2使用较新FLA快照，其余三组继续使用各自冻结源码。
- AdamW使用同样的残差输出LR倍率，并尊重layer标记的`_no_weight_decay`（A1_log/A2_log/dt1_bias/dt2_bias）；其余参数使用搜索WD。
- LR={2.5e-4,5e-4,1e-3} × WD={0,0.01,0.1}，seed731跑9个完整10M候选；只按最终完整验证集mean content CE选参，锁定后复验seed732和评估固定test/confirm。
- 选定配置直接迁移到15层hidden512/H8/FFN1536，先seed731完整训练相同100,000,690唯一token。复用主实验生成的数据只读目录，不重新抽样。
- 新增独立1GPU作业，原三组运行不变。预检核对原生层接口、共同骨干初值、WD豁免/残差分组、有限梯度、小batch学习与因果干预。每个训练前后按相同协议校验。
