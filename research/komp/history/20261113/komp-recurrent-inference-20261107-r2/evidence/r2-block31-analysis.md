# Block T31 诊断结果：完成采集，未通过模型验收

2026-11-07 18:38 北京时间结束，退出码0，1216条比较。root复用原评测已空闲GPU执行，没有新增资源。原始`attempts/r2-diagnostic-block31`及launcher回执已同步本地；comparisons SHA核对为`0969f8e1c1c08d51cb50b695ede8e530650a4ac96eeb2a1bef9260798d49dc72`。

## 已定位范围

24层同输入控制：72项conv final flag、72项conv缓存、168项operands、24项double-normalization输出、24项norm、24项postprocess和24项手工完整layer控制均逐元素相同。该样本可以排除上述路径导致首个偏差；不外推到其他长度/模型。

首个偏差出现在layer00算子输出：Triton-vs-native RMSrel .00263506006、abs .00390625，Torch-vs-native .00263496371。Triton-vs-Torch的开放state系数最大RMSrel仅5.62e-7（全24层），输出最大9.36e-5。该结果把主要差异定位到原生Block分块数值与两种token recurrence的共同差别，尚未分别量化BF16各中间舍入的贡献。

原r1级联logits重现RMSrel .00300777050、abs .0625的门禁失败。只换成原native整个prefill输出后，24层及最终prefill logits与baseline逐元素一致。这个probe的缓存仍开放（63原子，其中31pending），未提前压缩。

## Decode

以下teacher forcing均使用baseline token，保持双方输入相同：

| 步骤 | r1 logits RMSrel | native-output probe RMSrel |
|---|---:|---:|
| 第1步decode | .00289759260 | .00253125484 |
| 第2步decode | .00270005586 | .00278867705 |
| 第3步decode | .00246531363 | .00291557322 |
| 第4步decode | .00379114708 | .00426382395 |

全部abs .0625。第4步双方均超过冻结.003门限；因此只改prefill输出不足以修复。baseline延长到T32..35时，已存在prefix hidden逐元素稳定，该样本不是baseline前缀自身漂移造成。

## 诊断top1字段存在bug，不能使用原回执的零mismatch结论

`logits_details`把`topk(2)`首项当作top1。PyTorch topk平局排序不保证与argmax一致，真实生成使用argmax。r1第1步decode保存了actual `[5344,272]` 两个值均9.5625；baseline唯一最大5344=9.5625、第二272=9.5。argmax选择最小最大值索引，因此actual argmax至多272，不可能等于5344；无需重跑即可证明该步greedy已经不一致。原始`top1_equal=true`错误，原回执保持不改，在本分析纠正。未保存全logits，不能断言actual确切argmax是272（可能还有更小的并列项）。

其他已存比较的双方top2 gap均正且最大ID相同，能够确认这些点argmax相同。native-output probe的prefill及4步decode均为正gap，所以这些点greedy一致，但数值门禁仍失败，不能称接受。

后续新revision必须直接记录`actual.argmax(-1)`和`expected.argmax(-1)`，保留topk仅作margin诊断，并补充平局且较小ID获胜的回归。未修改冻结r2源、门限或原始证据，未自动启动新GPU。下一实现需要解决Block decode数值路径的共同差异，不能通过只改prefill或放宽门限绕过。
