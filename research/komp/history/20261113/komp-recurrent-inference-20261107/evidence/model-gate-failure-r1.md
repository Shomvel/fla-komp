# r1 真实模型门禁失败：保留现场，未重试

候选 source-manifest：f0ebd0c1b3375e671d849a99e70c2019b3cf20be89d588982e4e5fe86dd050f8。八种算子共 120 项 GPU 对拍 passed；这不等于完整模型验收。

首次真实 checkpoint 检查为原 Block final checkpoint-012970。模型阶段 attempts/r1-models 在首个 model-prefill-31 失败：logits RMSrel 0.003007770501277965，冻结门限 0.003；maxabs 0.0625，冻结绝对门限 0.0003。两项均未通过。此处抛错发生在 greedy top1 检查之前，不能声称该样本 top1 一致或不一致。其余模型、decode、golden fixture 和性能均未进入验证。

## 已确认的静态路径

- komp_inference/prefill.py 对 Block 计算 count=T//128*128。因此 T31 的 count=0，整个前缀走 recurrent_komp；冻结原模型在 T31 仍调用 block_komp_fast(chunk_size=128)。两边是不同的数值计算路径。
- 该分支同时覆盖所有 Block 变体的短 prompt；非整块 prompt 的尾段也采用 recurrent 路径。检查或修复不能只针对原 Block/T31。
- 外层模型使用同一份 checkpoint 模块。核对后，unfused attention norm、两段 residual 加法、MLP 和最终 norm 的顺序与冻结 FLA block 一致。没有在此轮重新初始化模型参数。
- 卷积候选开启 output_final_state，原始调用未要求最终状态；候选持久状态与原生 block 的归约和 BF16 舍入路径也不同。这些是待逐层测量的候选因素，尚未证明哪一个造成此次超限。

## 后续诊断边界

先在独立新 revision 准备逐层差分诊断，固定相同权重、输入及 q/k/v/g/beta，分别记录卷积、mixer output、norm、residual 与最终 logits 的首次偏差。覆盖小于128、整128以及带尾段的边界，并检查 learned slot/value 共用分支。任何修复都必须保留开放 chunk，不能为匹配 prefill 而提前压缩后续 decode 的状态。

本轮没有放宽门限，没有修改冻结 r1 或正式训练/评测源，没有重跑失败门禁，也没有启动后续 fixtures/benchmark。完整失败日志归档在 evidence/model-failure-20261107T0944Z/r1-models。
