"""Immediate report for the completed original56 runs, separate from the ongoing fine scan."""
from pathlib import Path
import json
P=Path(__file__).resolve().parents[1];R=P/'physics_lm/block_runs_coh';s=json.loads((R/'original-statistics.json').read_text())
assert s['complete'] and len(s['arms'])==8 and all(x['content CE']['n']==7 for x in s['arms'].values())
labels={'coh_fifo_off':'FIFO / off','coh_fifo_99':'FIFO / 0.99','coh_fifo_95':'FIFO / 0.95','coh_fifo_80':'FIFO / 0.80','coh_gate_off':'Gate / off','coh_gate_99':'Gate / 0.99','coh_learned_off':'Learned / off','coh_learned_99':'Learned / 0.99'}
lines=['| 配置 | Content CE ↓ | Depo生成 ↑ | Brevo合法率 ↑ |','|---|---:|---:|---:|']
for a,label in labels.items():
 m=s['arms'][a];lines.append(f"| {label} | {m['content CE']['mean']:.4f} | {m['Depo gen %']['mean']:.2f}% | {m['Brevo valid %']['mean']:.2f}% |")
text='''# 原始 coherence 实验：56次训练已完成

原始8臂各7seeds已全部完成；本报告不包含正在进行的0.995/0.999细扫。

**结论：在原始网格中，开启coherence没有优于关闭；0.99是开启项里最好的。与gate/learned叠加均损害质量，但Gram条件数确实大幅降低。**

'''+ '\n'.join(lines)+'''

- FIFO+0.99相对关闭：CE +0.0321，95%CI[+0.0009,+0.0692]，仅略离零且未做多重比较校正；Brevo -2.46pp，CI[-6.14,+0.78]跨零。没有质量收益证据。
- 0.95与0.80分别损失Brevo约24.00pp与27.68pp，区间均远离零。0.80改善Depo CE，但Depo生成提升不稳定，不能抵消Brevo损失。
- Gate+0.99相对gate关闭：CE +0.0815，CI[+0.0348,+0.1323]；Brevo -6.58pp，CI[-10.83,-3.68]。
- Learned+0.99相对learned关闭：CE +0.0993，CI[+0.0336,+0.1664]；Depo -1.34pp，CI[-1.90,-0.89]；Brevo -4.58pp，CI[-8.71,-0.22]。原始0.99与两种配置都没有表现出正向叠加。
- E3 seed731：FIFO关闭的四层Gram条件数中位数为95207/73596/23059/14369；0.99降到105/1622/2198/2176。0.95/0.80下降更多，却损害质量。条件数改善不能直接证明TF32投影安全。

## 方法与可信度

全部臂均重新跑同一冻结coherence快照的FP32 flex；外围BF16 autocast；确定性算法开启；相同10,001,108token、2231steps、LR5e-4/WD0.1。没有混入旧Triton结果。preflight覆盖未来干预、训练后因果性、未来梯度、统计窗口隔离和E3只读捕获一致性。56次全部重载通过，14,336条生成记录独立审计一致。

最初非确定性scatter_add造成相同输入重复前向的末位差异，被BF16放大；开启确定性模式后因果检查严格为零。修正发生在正式训练之前；未改kernel，未放宽容差。

统计是seed配对20,000次bootstrap，固定数据与生成样本，无多重比较校正。阈值按7seed confirm CE选择，存在重复使用confirm与选参偏差；E3只覆盖每臂seed731。0.95/0.80在用户要求细扫之前已经跑完，作为负结果保留，不再追加。

## 脚本生成完整表

'''+(R/'ORIGINAL_TABLE.md').read_text()+ '\n## E3完整表\n\n'+(R/'ORIGINAL_DIAGNOSTICS.md').read_text()
(P/'BLOCK_KOMP_COHERENCE_ORIGINAL.md').write_text(text)
print(P/'BLOCK_KOMP_COHERENCE_ORIGINAL.md')
