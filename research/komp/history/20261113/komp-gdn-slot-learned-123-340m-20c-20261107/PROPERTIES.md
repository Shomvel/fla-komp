# GDN slot + learned Block + 123 性质记录

## 结构与可比性

这是一组新的组合实验：341,159,168 参数，FF2272，三开关全部开启。与原组合 123 的同 FF2272 共享骨干初始化哈希在 GPU 预检中核验；与 FF1824 learned 组存在 FFN 形状差异。更换衰减参数化、淘汰策略、score 写门、输出激活、投影秩和时间尺度后，单组收益不能归因于某一个开关。

## 可直接由定义得到的性质

| 性质 | 含义与边界 |
|---|---|
| GDN 衰减 | 有限参数下，数学上 g=-exp(A_log)softplus(z+dt_bias)<0，alpha=exp(g)在(0,1)；浮点极端值仍可能下溢或溢出，训练检查finite。 |
| 多时间尺度 | 初始16个head的半衰期约48.55–776.82 token，同head的32个物理slot相同。只匹配平均logalpha=log(.995)，不匹配所有head alpha。半衰期不等于实际token保留时长。 |
| sigmoid 输出门 | 门值位于[0,1]，而SiLU可负且正向无界；整层还有RMSNorm、输出投影和残差，不能据此声称整层输出有界。 |
| 低秩投影 | gate和decay线性映射rank<=64，没有中间非线性；softplus/sigmoid后的门矩阵不受同一线性rank上界约束。 |
| learned eviction | 每chunk结束保留最高分M个旧/新key，同分取新；高分key可以保留超过FIFO窗口，也可能被后来更高分token淘汰。 |
| score训练路径 | 硬排序与索引选择detach。score通过beta乘2sigmoid(score)学习写入强度，不是对离散淘汰动作的直接梯度。 |
| beta范围 | beta基础sigmoid再乘2sigmoid(score)，数学范围(0,2)，初始0.2；不是始终限制在(0,1)。不据此宣称整个记忆更新严格收缩。 |
| 初始低秩遗忘 | random down / zero up保持初始衰减与输入无关。首步down数据梯度0，但AdamW WD=.1造成参数衰减；up更新后第二步应有非零down数据梯度。 |
| timescale优化 | A_log与dt_bias为可学习参数、WD0；训练后head/slot时间尺度可偏离初值。 |
| 物理slot语义 | 衰减参数属于物理slot；learned路由改变某slot中的key，并不把某个token的专属衰减参数一同迁移。 |

## 实测验收范围

`evidence/properties-block.json`：8种123开关组合的真实mixer输出路径、初始门值、独立FP64门公式和梯度、低秩结构以及score梯度路径。

`evidence/operator-block.json`：随机FP32/BF16与真实语料操作数，对独立FP64参考比对输出、keys、state、7组连续操作数梯度；retain-early、retain-initial、ties-newest三种routing。

`evidence/preflight-block.json`：真实8rank、B16x2 full/tail更新，24层192个门参数的梯度及更新，初始/更新后prefix1024和1031因果差，replica一致性，完整model/Adam/RNG/loader恢复。所有门限预先冻结；旧PASS不用于新源。

`runs/block/metrics.jsonl` 是正式训练进度。预检trial不计6.8B预算，正式模型从fresh seed42和原始数据cursor开始。速度、显存以正式暖步为准，验收耗时包含编译、hash、诊断不能当训练吞吐。

GPU验收状态以对应新source证据为准；本文的结构说明不是GPU通过证明。所有长训比较按同有效token的val CE/PPL，不用不同进度的train loss排名，不承诺bitwise。

## 本次新源已取得的GPU证据

源指纹：`b5828e0262452ac518840be596e26406620e19b2b68bc5e8d590d8037699ecd7`。

8种开关组合状态：passed。独立FP64门公式及梯度对照最大RMS相对误差为 0.000416490066。

- random-learned-tail-torch.float32：passed=True，最大RMSrel=0.00222368557。
- random-learned-tail-torch.bfloat16：passed=True，最大RMSrel=0.00400212225。
- real-first-layer-initial-ties-tail：passed=True，最大RMSrel=0.00449728954。

完整8rank更新与恢复均已通过，新源正式训练已从fresh开始。

完整模型验收：

- 初始完整模型SHA `b963326927545c1d7b440f05598b305a96fe00118b11a2332ed97a4d29e34132`；共享FF2272骨干SHA `bb75bbc333c113ed94243df3df38086bd5ce49e229a8096bee1717b5e3133955` 与原组合一致。
- full/tail各192个门参数检查通过；首次24个decay down数据梯度为0且只有WD更新，第二次全部有非零数据梯度。
- 初始及更新后的prefix1024/1031因果绝对差均0；full/tail 8rank replicas一致。
- 8份恢复receipt均通过，579个Adam状态，model/Adam SHA一致，RNG与loader一致。
- 预检跨rank峰值 35.5916 GiB；此值包含诊断与比对，不作为正式训练显存。
- 验收trial loss不用于质量结论，正式首步已记录在runs/block/metrics.jsonl；首次val在512步。

正式启动快照（北京时间2026-11-07 03:03）：step 55，28,835,840有效token（0.4241%），loss 8.2457964。最近20步中位2.683914秒，吞吐195345token/s，rank0峰allocated 35.58747 GiB。步数连续唯一、loss/梯度有限；尚无checkpoint256或val512。只作启动性能记录，不是质量收益证据。
