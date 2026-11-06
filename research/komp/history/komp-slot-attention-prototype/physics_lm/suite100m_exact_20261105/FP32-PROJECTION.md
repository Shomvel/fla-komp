# FP32 投影测试：2026-11-05

结论：已训练的 alpha0=.995 模型切换 FP32 geometry 后，两个任务的完整验证 CE 基本不变，监督位置 hidden 相对 RMS 误差低于0.18%。反向梯度存在可观测偏差，Depo2关键参数最大3.66%、Brevo2最大1.68%；合成近重复key压力检查仍有一项不满足原0.5%门槛。因此本次支持固定checkpoint评估近似一致，不证明FP32从头训练或续训轨迹等价，不计为严格数值验收全过。

## 方法与范围

- 两任务各使用已训练满约10M token的100,526,392参数Exact模型；alpha0=.995、beta0=.2、temperature=.5、ridge=1e-4、M32、FIFO。
- 这是当前已完成候选中最优配置，筛选尚未全部完成。
- 同一checkpoint分别使用FP64和FP32投影，递推状态均FP32、模型投影BF16；不执行optimizer.step，不增加训练token。
- 完整验证集8卡分片；固定前16篇真实验证文档做监督位置hidden及9组关键参数梯度比较。
- 检查checkpoint SHA256、源码指纹和权重未改变。FP64验证CE精确复现原筛选记录。

| 指标 | Depo2 | Brevo2 |
|---|---:|---:|
| 完整验证监督位置 | 65,627 | 36,619 |
| FP64 验证 CE | 1.088958904850 | 1.257567770904 |
| FP32 验证 CE | 1.088950344843 | 1.257536485834 |
| CE差：FP32−FP64 | −0.000008560007 | −0.000031285070 |
| 监督 hidden 相对 RMS | 0.125065% | 0.174442% |
| 固定梯度 batch 监督位置 | 1,052 | 625 |
| k projection 梯度相对 RMS | 3.663661% | 1.683284% |
| embedding 梯度相对 RMS | 1.292887% | 0.843857% |
| out projection 梯度相对 RMS | 1.076993% | 1.023483% |
| decay weight 梯度相对 RMS | 0.829107% | 1.176225% |
| decay bias 梯度相对 RMS | 0.745641% | 1.111840% |
| beta weight 梯度相对 RMS | 0.675737% | 0.492267% |
| LM head 梯度相对 RMS | 0.482749% | 0.552661% |

所有实测输出/梯度均有限。第0与14层初始字典keys梯度在这批文档上均为0，不能据此推断初始字典梯度普遍稳定。CE微小降低属于精度扰动，不作为性能提升。

## 压力测试

沿用seed917、B1/T65/H1/M32/K64/V64，random、near、duplicate、strong四种输入，FP32/BF16两种输入精度，分别核对输出、最终keys/state、七项输入梯度，共80项。参照是FP64 PyTorch逐token实现；阈值没有更改。

两组独立执行均79/80通过。唯一失败为baseline/near/torch.float32/dkeys0：相对RMS约1.30098%，原门槛0.5%；归一化最大误差约0.18478%，在0.5%门槛内。此前FP64 geometry同例相对RMS约0.0003407%。这说明FP32近奇异Gram投影反向仍有明确精度限制。

## 耗时解释与后续

本次记录的单次验证耗时包含FP32新shape编译，不能用于比较稳态吞吐。调参在安全checkpoint边界暂停后已排队自动恢复，原候选预算和模型配方不变。正式FP32训练效果仍需独立训练证据；本次固定checkpoint的CE接近不能替代该证据。

原始证据：`evidence/geometry-transfer-depo2.json`、`evidence/geometry-transfer-brevo2.json`、`evidence/fp32-projection-diagnostic-depo2.json`、`evidence/fp32-projection-diagnostic-brevo2.json`。
