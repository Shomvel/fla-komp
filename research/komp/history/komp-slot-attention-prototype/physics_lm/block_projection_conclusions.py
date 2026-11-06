"""Render Chinese conclusions using script-computed means and paired intervals."""
import json
from pathlib import Path
R=Path(__file__).parent/'block_runs_proj'
s=json.loads((R/'statistics.json').read_text())
assert s['integrity']=='complete'

def mean(a,m):return s['arms'][a][m]['mean']
def val(a,m,d=2):return f'{mean(a,m):.{d}f}'
def diff(a,b,m,reverse=False):
 p=next(p for p in s['paired'] if (p['a'],p['b'],p['metric'])==(a,b,m))
 mu,lo,hi=p['mean'],*p['ci95']
 if reverse:mu,lo,hi=-mu,-hi,-lo
 if mu==0:mu=0.
 d=2 if m.endswith('%') else 4
 return f'{mu:+.{d}f}（95% CI [{lo:+.{d}f}, {hi:+.{d}f}]）'
def benefit(f,m,c=64):return diff(f'{f}_noproj_c{c}',f'{f}_proj_c{c}',m,True)
text=f'''## 投影值多少

以下“开启投影的变化”都以无投影为基准，按相同 seed 配对。CE 越低越好；生成差异单位为百分点。

- **FIFO：投影稳定改善 CE，但没有稳定改善 Depo 生成，并小幅损害 Brevo 合法率。** 总 CE 变化 {benefit('fifo','content CE')}；Depo CE {benefit('fifo','Depo CE')}，Brevo CE {benefit('fifo','Brevo CE')}。Depo 生成 {benefit('fifo','Depo gen %')}，区间跨零；Brevo 生成 {benefit('fifo','Brevo valid %')}，区间不跨零。不能把 CE 收益直接替换成完整答案成功率收益。
- **learned：投影的平均 CE 收益更大，并稳定提升 Depo 生成；Brevo 生成没有明确变化。** 总 CE 变化 {benefit('learned','content CE')}；Depo CE {benefit('learned','Depo CE')}，Brevo CE {benefit('learned','Brevo CE')}。Depo 生成 {benefit('learned','Depo gen %')}；Brevo 生成 {benefit('learned','Brevo valid %')}。两侧都保留相同的 learned score 写入门控，因此这是 learned 配置内部的投影消融，不是门控或淘汰选择的额外消融。

**按 CE 目标，投影有明确价值；按 Brevo 合法生成目标，不能声称投影必要。** FIFO 无投影的 Brevo 合法率为 {val('fifo_noproj_c64','Brevo valid %')}%，learned 无投影为 {val('learned_noproj_c64','Brevo valid %')}%，都保留了较强的 Brevo 表现。本轮没有实现或测量无投影 Triton kernel，不能把 flex 训练用时减少直接换算成生产 kernel 加速。

## 同等状态预算与 SWA

**按用户指定的状态元素数对齐，Block 的优势主要在 Brevo。** FIFO 有投影与 SWA W32 都为每 head 4096 个状态元素。前者相对后者，总 CE {diff('fifo_proj_c64','swa_w32','content CE')}；Depo CE {diff('fifo_proj_c64','swa_w32','Depo CE')}，Brevo CE {diff('fifo_proj_c64','swa_w32','Brevo CE')}。Depo 生成 {diff('fifo_proj_c64','swa_w32','Depo gen %')}，区间跨零；Brevo 合法率 {diff('fifo_proj_c64','swa_w32','Brevo valid %')}。Depo CE 的区间较接近零，且未做多重比较校正，其证据强度低于 Brevo 的大幅差异。

扩大到 W96、达到 Block 块内最大原子数后，SWA 的平均 content CE 为 {val('swa_w96','content CE',4)}。Block FIFO 相对它的总 CE 为 {diff('fifo_proj_c64','swa_w96','content CE')}，Brevo 合法率为 {diff('fifo_proj_c64','swa_w96','Brevo valid %')}。Depo CE 对比 {diff('fifo_proj_c64','swa_w96','Depo CE')}、Depo 生成对比 {diff('fifo_proj_c64','swa_w96','Depo gen %')}，均不能得出 Block 稳定更好的结论。

RoPE96 相对 NoPE96 的总 CE 为 {diff('swa_w96','swa_w96_nope','content CE')}；Depo/Brevo 生成分别为 {diff('swa_w96','swa_w96_nope','Depo gen %')}、{diff('swa_w96','swa_w96_nope','Brevo valid %')}。在本配方中 RoPE 有稳定收益，SWA 的较差表现不能简单归因于没有位置编码。

这里对齐的是元素数；实际状态字节数、瞬时字典、读写规则和有效参数量并不完全相同，详见方法。所有 SWA 使用 KOMP 的门控 backbone、初始化与统一 LR 配方，没有单独调参。

## 全注意力参考

全因果 RoPE192 的 content CE 为 **{val('full_attn','content CE',4)}**，Depo CE {val('full_attn','Depo CE',4)}、Brevo CE {val('full_attn','Brevo CE',4)}；Depo/Brevo 合法生成分别为 **{val('full_attn','Depo gen %')}% / {val('full_attn','Brevo valid %')}%**。W96 的对应总 CE 为 {val('swa_w96','content CE',4)}、生成率为 {val('swa_w96','Depo gen %')}% / {val('swa_w96','Brevo valid %')}%。

全注意力在这份固定训练配方下，均值没有展示出相对 W96 的大幅提升。它给出的是本次训练达到的参考水平，**不是 softmax attention 的能力上限**。尤其 Depo 的 seed 波动较大；没有更长训练或专门调参，不能据此宣称 Block 普遍优于 Transformer。

## C128 补充实验

C128 开启投影相对关闭投影，总 CE {benefit('fifo','content CE',128)}；Depo CE {benefit('fifo','Depo CE',128)}，Brevo CE {benefit('fifo','Brevo CE',128)}；Depo/Brevo 生成分别为 {benefit('fifo','Depo gen %',128)}、{benefit('fifo','Brevo valid %',128)}。

这组结果必须结合序列长度解读：confirm 中只有约 4.5% 的 Depo、1.5% 的 Brevo 内容预测位置跨过第一次 C128 块末。C 的改变也改变压缩频率和块内可见字典，并非纯计算分块优化。因此不能把 C128 的较小投影效应外推为长上下文中投影无用。

## Block 是否只是“带 delta rule 的 sliding window”

**有投影的 Block-KOMP 包含显式的窗口外系数转移，不能等同于普通的滑动窗口。** FIFO 限定字典 key 来自最近的原子，但被淘汰原子的系数通过联合投影加到保留原子上，历史信息可以跨过字典 key 的窗口继续传递。learned 淘汰则连保留 key 的时间范围也不局限于最近 M 个。

关闭投影后，称其为“按块淘汰、在保留 key 上做 delta rule 的窗口模型”是合理的，但它仍不等同于 softmax SWA：块内字典大小随相位变化，写入系数是相对旧记忆预测的残差，留下的系数仍可能间接编码更早历史；多层堆叠也扩大模型的感受野。

实验支持“投影改善固定训练配方下的预测质量”，尤其 learned 的 Depo；它也显示 Brevo 的大部分生成优势在关闭投影后仍保留。**不能把 Block 相对 SWA 的全部收益归于联合投影**，也不能仅凭这次短序列消融证明改进全部来自长期信息保真。Gaussian 读出、delta 写入、逐 slot 衰减、初始原子与分块机制尚未分别消融。
'''
(R/'evidence/conclusions.md').write_text(text)
print(text)
