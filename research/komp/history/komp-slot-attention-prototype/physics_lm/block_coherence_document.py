"""Render the final coherence report from audited, complete statistics."""
from pathlib import Path
import json

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'physics_lm/block_runs_coh'


def main():
    stats=json.loads((ROOT/'statistics.json').read_text());assert stats['complete']
    audit=json.loads((ROOT/'evidence/final-source-audit.json').read_text());assert audit['status']=='passed'
    generation=json.loads((ROOT/'evidence/independent-generation-audit.json').read_text())
    assert generation['status']=='passed' and generation['checked_generations']==14336
    assert json.loads((ROOT/'evidence/result-audit.json').read_text())['status']=='passed'
    for i in range(1,17):assert json.loads((ROOT/f'evidence/final-cleanup/r{i}-cleanup.json').read_text())==[]
    fine=stats.get('fine_selection')
    tag=(fine or stats['selection'])['selected_tag'];theta={'99':.99,'95':.95,'80':.8,'995':.995,'999':.999}[tag]
    total=sum(a['content CE']['n'] for a in stats['arms'].values())
    if fine:
        assert json.loads((ROOT/'evidence/final-fine-audit.json').read_text())['status']=='passed'
        for i in range(17,31):assert json.loads((ROOT/f'evidence/final-cleanup/r{i}-cleanup.json').read_text())==[]
    selected='coh_fifo_'+tag
    def diff(a,b,metric):
        r=next(r for r in stats['paired'] if r['terms']==[a,b] and r['metric']==metric)
        d=2 if metric.endswith('%') else 4;lo,hi=r['ci95'];unit=' 个百分点' if metric.endswith('%') else ''
        status='区间跨零' if lo<=0<=hi else '区间不跨零'
        return f"{r['mean']:+.{d}f}{unit}，95% CI [{lo:+.{d}f}, {hi:+.{d}f}]（{status}）"
    def decision(a,b,metric):
        r=next(r for r in stats['paired'] if r['terms']==[a,b] and r['metric']==metric)
        lo,hi=r['ci95'];lower=not metric.endswith('%')
        if lo<=0<=hi:return '本轮不能确认稳定差异'
        return '改善' if (hi<0 if lower else lo>0) else '变差'
    ranking=sorted(stats['selection']['ranking'],key=lambda x:x['mean_content_ce'])
    off_status='关闭组的 CE 均值仍更低，因此不能把 θ* 称为胜过关闭组的方案。' if stats['arms']['coh_fifo_off']['content CE']['mean'] <= stats['arms'][selected]['content CE']['mean'] else f'包含关闭组在内，{selected} 的 CE 样本均值最低；统计稳定性仍需看配对区间。'
    diagnostics=json.loads((ROOT/'diagnostics-statistics.json').read_text())
    condition=[]
    for layer in range(4):
        b=diagnostics['coh_fifo_off'][str(layer)];s=diagnostics[selected][str(layer)]
        condition.append(f"第{layer}层中位数 {b['condition_median']:.2f} → {s['condition_median']:.2f}，最大值 {b['condition_max']:.2f} → {s['condition_max']:.2f}")
    quality=[]
    for a in ['coh_fifo_99','coh_fifo_95','coh_fifo_80'] + [f'coh_fifo_{t}' for t in (stats.get('fine_screening') or {}).get('expand_tags',[])]:
        quality.append(f"- `{a}` 相对关闭：content CE {diff(a,'coh_fifo_off','content CE')}；Depo 生成 {diff(a,'coh_fifo_off','Depo gen %')}；Brevo 合法生成 {diff(a,'coh_fifo_off','Brevo valid %')}。")
    combo=[]
    for family in ('gate','learned'):
        a=f'coh_{family}_{tag}';b=f'coh_{family}_off'
        combo.append(f"- **{family} 叠加 coherence**：content CE {diff(a,b,'content CE')}；Depo CE {diff(a,b,'Depo CE')}；Brevo CE {diff(a,b,'Brevo CE')}；Depo 生成 {diff(a,b,'Depo gen %')}；Brevo 合法生成 {diff(a,b,'Brevo valid %')}。")
    text=f'''# Block-KOMP coherence 准入：Depo/Brevo 7-seed 实验

原始56次训练已完成；按用户追加的近1阈值细扫完成后，当前共 **{total}次正式训练、{len(stats['arms'])}个臂**。全部使用冻结coherence kernel的FP32 PyTorch flex路径，最大并发16个单卡H100作业。每轮阈值选择均在相应7seeds完成后锁定，再跑组合实验。preflight、源码、预算、重载和生成审计通过；每臂seed731完成E3，随后删除权重文件。kernel源码未改。

## coherence 是否优于关闭

按完成的阈值扫描选出的 θ*={theta:g} 相对关闭：content CE **{decision(selected,'coh_fifo_off','content CE')}**，Depo生成 **{decision(selected,'coh_fifo_off','Depo gen %')}**，Brevo合法生成 **{decision(selected,'coh_fifo_off','Brevo valid %')}**。

{chr(10).join(quality)}

CE和生成率是不同指标，区间跨零不能解释为统计等效。这里的置信区间只反映固定语料与评测样本下的seed波动。

## 阈值选择

**θ*={theta:g}**，初始扫描是0.99/0.95/0.80，锁定0.99后开始组合实验；随后用户要求向1附近细扫，新增0.995/0.999，先按预先声明的seed731止损线筛选，再给通过者补齐7seeds。最终选择以0.99与通过筛选的细阈值的7seed confirm content CE均值为准。{off_status} 最终近1候选排序另见 `evidence/fine-selection.json`（如有）；原始排序仍保留。

“最优”仅指本次网格与当前训练预算，不是连续阈值的全局最优。沿用上一轮报告的content CE字段，即confirm均值；同一个confirm还用于选参和报告，因此存在选择偏差。没有把扫描获胜者的普通bootstrap区间伪装成选参校正后的独立测试区间。原始选择记录和所有候选的validation均值保存在 `evidence/selection.json`。

## 与 gate、learned 的叠加

{chr(10).join(combo)}

learned臂始终包含 `beta *= 2*sigmoid(score)` 与块末score排序；gate臂仅有该写入门控、淘汰仍为FIFO。coherence是更早的准入步骤：拒绝成为新原子的token将残差并到最近原子，不更新其key、slot、birth或保留score。硬准入和top-M均不传梯度，score通过写入门控获得梯度。

下面同时给出“组合−相同gate/learned关闭coherence”的主对比、“组合−FIFO+θ*”以及差中差。差中差是补充交互诊断，不能只凭两个独立对比的显著性判断有无协同。

## 是否改善 Gram 条件数

在seed731、相同64条confirm完整样本上，θ*与FIFO关闭组的条件数对比如下：

{chr(10).join('- '+s+'。' for s in condition)}

各阈值和组合的完整逐层结果见E3表，同时列出并入率、近重复原子对和系数范数。E3只用每臂一个seed，尚不能推断7seed的条件数分布。较低条件数只能支持进一步做TF32数值验收，**不能据此宣称投影可以安全改成TF32**；本轮没有改kernel或运行TF32。

## 方法与验收

{(ROOT/'evidence/METHOD.md').read_text()}

本轮实际冻结文件：`block_flex.py` SHA256 `04e049363c3adcc4e675f2076b9977e585c3ce655e8566ffaad14519964c3144`；`block_reference.py` SHA256 `c36873c0ec4a9987bb1a8c3760289091a93c8ac47ce0bca4191c6a1d30cd5a24`。全部komp_slot文件来自指定快照，不从持续变化的主树拷贝。

GPU首次preflight发现默认scatter_add累加顺序造成的非确定性；相同输入重复前向也出现差异。开启PyTorch确定性模式后重复运行及未来干预均严格为零，再对所有臂统一重新preflight。这个修正只改变实验运行设置，没有改冻结kernel，也没有放宽因果性容差。初次失败、定位、修正验收及各次作业清理证据均保留。

生成审计独立执行Depo输入字典跳转和Brevo输入DAG祖先闭包；原始56次训练的14,336条生成记录与细扫新增记录分别审计，均与summary一致。对照全部重跑在本轮flex路径，不混入旧Triton结果，也不混入Claude的3seed预实验。

## 质量与条件数图

![Coherence comparison]({ROOT/'COMPARISON.png'})

上图误差棒是seed样本标准差，不是置信区间；下图条件数只来自seed731。

## 脚本生成的质量与并入表

以下逐字插入 `physics_lm/block_runs_coh/TABLE.md`，由 `block_report_coherence.py` 读取summary与并入json生成。

{(ROOT/'TABLE.md').read_text()}
## 脚本生成的E3诊断表

以下逐字插入 `physics_lm/block_runs_coh/DIAGNOSTICS.md`。原始逐样本/head/块末记录在每臂 `*-seed731-diagnostic.json`。

{(ROOT/'DIAGNOSTICS.md').read_text()}
## 产物与清理

结果目录：`physics_lm/block_runs_coh/`，不含 `.pt`。`statistics.json`与`diagnostics-statistics.json`为机器可读统计；`evidence/`保存冻结快照、命令、16个作业配置、计划、选参、环境、错误定位与最终审计。

独立远端目录 `/gpfs/users/linlizhi/komp-coh-trials-20261101`，原始16个alias为 `komp-coh-trials-20261101-r1` 至 `-r16`；细扫使用 `-r17`/`-r18` 试跑及 `-r19` 至 `-r30` 扩充，并保持总同时占卡不超过16。保存终态后逐一按精确alias删除，并重新查询为空；证据在 `evidence/final-cleanup/`。未操作Claude的 `komp-block-coh-20261101`。

重建：`python3 physics_lm/block_report_coherence.py`，再运行 `python3 physics_lm/block_coherence_document.py`。本轮没有修改 `BLOCK_KOMP.md` 或 `README.md`。
'''
    (PROJECT/'BLOCK_KOMP_COHERENCE.md').write_text(text)
    print(PROJECT/'BLOCK_KOMP_COHERENCE.md')


if __name__=='__main__':main()
