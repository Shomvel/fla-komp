# Depo/Brevo：KOMP、GSA、GDN 匹配小模型验证

沿用“精美LR Schedule”后续的去重 10M-token 数据与评分口径。旧训练快照存在已记录的门换底缺陷（见旧项目 `MASK_AUDIT.md`），本研究不复用旧模型或 FLA 工作树，不将旧 181M 成绩混入比较。

## 预先固定的设计

- 三个序列算子：KOMP 全 Triton、干净 FLA GSA、干净 FLA GDN；两个 seed 731/732，共六次训练。
- 骨干一致：4 层、hidden128、2 heads、每 head key/value64、FFN256、词表32000、untied embedding/head、RMSNorm、SwiGLU、输出 gate。无卷积、无 attention 混合；架构特有的门/初始 key 参数不同，记录实际参数量。
- GSA/KOMP 每 head32 slots，与 GDN 的64×64状态拥有相同的4096个状态元素。KOMP 初始 key 正交并可训练，value 零初态；替换位置为 t mod32，硬路由不训练、不执行完整 KOMP 删除代价搜索。
- 初态/输入投影等共同模块按名字固定随机初始化，模型特有参数不改变公共模块的随机数序列。数据顺序使用独立 seed12345。
- 原数据 `/gpfs/users/linlizhi/softmax-10m-20261023/data` 只读。train142784完整文档、10001108去重非padding token，Depo/Brevo各半。每次完整1epoch，batch64，共2231 optimizer steps。固定右padding192、不packing、不截断；所有原文档最长171。
- 相同 AdamW，beta=(.9,.95)、WD=.1、clip1、peakLR5e-4；warmup5%+cosine至10%。输出残差投影的 LR 倍率从 .1 平滑释放到 .3（step25%–75%），其余倍率1。这里复用旧研究的策略形式，不沿用只对旧181M GDN调出的最优参数，也不声称此配方最适合小模型。没有按新测试结果选 LR。
- FP32参数，BF16 autocast。共享输出门/逐head RMSNorm。KOMP内部为FP32；GSA/GDN使用原生计算精度。
- 固定使用 FLA commit97874af2400916e2885c33ea30559a70ff686b66；H100的GDN反向要求Triton>=3.7.1，所以三者统一使用隔离overlay Triton3.7.1，不绕过上游编译器保护。

## 验证与评测

先1GPU预检：四个split独立检查映射/拓扑答案、完整文档去重及SHA-256，三个完整模型的因果前缀、16步小batch可学习性和梯度有限性。它只是链路检查，不是正式训练结论。

正式训练每500step与终点报告验证集内容CE/准确率；loss对答案区域监督，内容指标排除特殊分隔符。每模型固定使用最后一步checkpoint，不按test或confirm挑选。最终完整val/test/confirm各2048条，confirm另取固定每任务128例自由greedy生成。Depo评估首个问题答案；Brevo评估祖先集合与合法拓扑序，允许多种有效顺序，另报gold序列exact。没有约束解码，没有把gold答案放进prompt。记录节点数/跳数用于分桶。

正式预定6runs均完成才作两seed对照；不能从两seed给出稳健显著性结论。报告训练吞吐、显存、内容CE、内容准确率、自由生成准确率。不同模型的gate设计、初始化记忆和硬路由仍是架构差异，等状态大小不等同于等效果。

最多同时3张H100，每个architecture串行跑两seed。独立路径/alias和有界超时；失败不自动重训或扩大扫描。已有LR研究、源checkout和其他作业保持不动。

## 实施审计与修复

- 首次训练在安全checkpoint回读时遇到TorchVersion对象，修复为普通字符串，并把保存/回读契约检查移到训练前。首轮失败日志保留，未混入正式结果。
- r2的GSA门权重全零、各slot同bias且零记忆，保留slot置换对称，构成退化基线。静态复核发现后，原两seed移入excluded-symmetric-gsa，仅作为失败对照。r3用独立normal(0,.02)门投影打破对称，保留共同初始化、门bias、LR和全部预算；训练开始检查每head的slot门方差。此修复不改变GDN/KOMP，正式表使用r2的GDN/KOMP和r3的GSA。
- 这是统一骨干中的三种mixer比较，不是FLA原生完整GSA/GDN网络复现。GSA使用共同的L2-normalized q/k、输出norm/gate，门logsigmoid未除以原生GSA的8；这些设计限定结论。

门的范围说明：本轮GDN与KOMP都使用sigmoid(linear)的scalar decay和sigmoid beta，初始alpha=.98/beta=.2；没有使用FLA原生GDN的A_log/dt_bias参数化。后续门研究需作为独立消融，不把本轮结果称为原生GatedDeltaNet完整复现。
