# KOMP decay 与几何优化，2026-10-31

用户授权三个subagent分别实现GDN scalar decay、GSA逐slot decay和KOMP加速。训练结果与几何优化分别取证，不把算法变化混入速度收益。

## 预先固定的四组

1. sigmoid：原scalar gate，使用Triton gate激活。
2. gdn_scalar：alpha=exp(-exp(A_log)*softplus(a+dt_bias))，A_log=0、dt_bias使alpha=.98。
3. gsa_scalar：alpha=exp(logsigmoid(a)/8)，scalar参数化对照。
4. gsa_slot：相同/8参数化，alpha[B,T,H,M]作用旧系数，独立beta。

所有组零gate权重、初始alpha=.98、beta=.2，共同参数（包括初始keys）逐参数hash一致。这里KOMP的不同key和循环路由已破坏slot对称，因此slot门可以零权重起步而精确匹配scalar初始函数；预检必须确认首步不同slot的gate梯度不相同。它与先前GSA两个零状态/对称slot的退化不同。

四组共同规定decay bias、A_log、dt_bias不做weight decay，其余保持.1，显式optimizer分组；为此重跑sigmoid对照，不把旧模型当严格匹配baseline。两seed731/732，共8runs，每组同样10,001,108唯一非padding输入token、2231步、batch64，沿用原4层hidden128及warmup/cosine+输出投影倍率LR。固定最后checkpoint。geometry使用原IEEE路径，核函数/ridge/循环route均不改。

报告完整val/test/confirm内容CE、内容准确率、每任务固定128例greedy生成及难度桶、门值分布/学习速率、训练吞吐与显存。两seed仅作探索，不声称统计显著性。独立beta始终保留。

## 验收与加速

先验收门输出/梯度、slot递推输出/终态/全梯度与FP64 oracle、退化scalar、零门、尾块、重复route、streaming及slot置换；再验证完整模型因果性、初始函数匹配、小batch可学习性、checkpoint回读。

几何优化保留baseline API，新增gather_ieee及gather_tf32x3候选。相同ridge与kernel，不改路由；先普通case及近重复key的数值验收，再同卡eager/CUDA Graph计时。未通过的候选不用于默认路径。若快路径没有加速，如实保留负结果，不把减少工作量当加速。

GPU使用WeaverRSI独立alias、固定镜像及Triton3.7.1 overlay；每类首次失败先静态排查再重跑，所有作业有期限且完成后清理。源码和证据保存在本轮独立目录。

## 实施审计

第一轮门107项、slot182项验收通过。模型初始值预检发现BF16 Linear先量化bias导致alpha偏离.98，统一改为FP32 gate和beta投影，四层显式检查alpha=.98/beta=.2后重跑预检。预检票据绑定完整实验/核心源码hash，后续训练必须一致。slot梯度检查逐层逐head执行。日志里的门统计取最后一个验证minibatch且包含padding，仅作诊断，不是全验证集分布。

第二轮预检四组因果与小batch学习均通过，但BF16初始hidden跨模型最大误差.0183，超过FP32级等价阈值。静态核对公共初始化一致且无随机forward；scalar与slot块系数算术顺序不同、不同gate亦有ULP级差异。第三轮将初始数学函数等价性在禁用autocast的FP32下按原3e-4门槛检查，同时单独记录BF16最大误差/RMS，不将其写成BF16逐元素等价。
