"""Reproduce descriptive tables from frozen, verified 340M evaluation results."""
import csv,datetime,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
evidence=ROOT/'evidence/final-verification.json'
verified=json.loads(evidence.read_text())
rows=list(csv.DictReader((ROOT/'FINAL_SCORES.csv').open()))
assert verified['status']=='passed' and verified['total_results']==163656 and verified['total_items']==184
assert len(rows)==184 and len({(r['model'],r['case']) for r in rows})==184
for r in rows:
 v=verified['models'][r['model']];c=v['cases'][r['case']]
 assert int(r['count'])==c['count'] and abs(float(r['score'])-c['score'])<1e-14
 assert r['checkpoint_sha']==v['checkpoint_sha'] and r['plan_sha']==verified['plan_sha']
ORDER=['kda','block','block123','learned','learned123','value','exact123','exact_partial']
LABEL=dict(zip(ORDER,['KDA','Block','Block+123','Learned','Learned+123','Value','Exact+123','Exact partial¹']))
M=verified['models']
def score(m,case):return M[m]['cases'][case]['score']*100
def mean(m,prefix):
 xs=[v['score']*100 for k,v in M[m]['cases'].items() if k.startswith(prefix)]
 return sum(xs)/len(xs)
def table(headers, body):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']+['---:']*(len(headers)-1))+'|']+['| '+' | '.join(map(str,r))+' |' for r in body])
def task_table(cases,headers):
 return table(['模型',*headers],[[LABEL[m],*[f'{score(m,c):.2f}' for c in cases]] for m in ORDER])
summary={m:{p:mean(m,p) for p in ['niah_single_1_','niah_single_2_','niah_single_3_','phonebook_']} for m in ORDER}
case_winners={}
for case in M['kda']['cases']:
 best=max(score(m,case) for m in ORDER if m!='exact_partial')
 case_winners[case]=dict(score=best,models=[LABEL[m] for m in ORDER if m!='exact_partial' and score(m,case)==best])
report='''# KOMP 340M 下游评测整理

这批8个checkpoint的评测已全部完成：**163,656/163,656条结果，184/184分项，34,000个完整batch**。最终核验status=passed；每模型23分项、20,457条结果，唯一ID覆盖、固定plan/checkpoint内容哈希和恢复前26,731个旧batch哈希均通过。

这是340M级别的8模型比较。除Exact partial仅训练3,758,096,384 tokens，其余均训练6.8B。参数量约338–341M。这批结果不包含当前正在训练的1B/50B，也不包含新训练的Value+learned+123 checkpoint。不同模型并非严格等参，当前单checkpoint结果不提供多seed方差或显著性检验。

**主要结论：Learned+123的优势集中在NIAH single1，以及single2的1K–4K；自然任务尚未全面超过KDA，PhoneBook没有因+123继续改善。不能据此称为所有检索任务或整体语言能力全面领先。**

## 1. NIAH：最清楚的改进，也有明确失效区

下表为同一NIAH变体的1K/2K/4K/8K四个分项等权均值，单位%；只做描述性汇总，不与其他任务合成总分。

'''
report+=table(['模型','single1四长度均值','single2四长度均值','single3四长度均值'],[[LABEL[m],*[f'{summary[m][p]:.2f}' for p in ['niah_single_1_','niah_single_2_','niah_single_3_']]] for m in ORDER])
report+='\n\n- single1：Learned+123均值68.60%，Learned36.35%，KDA19.35%；相对Learned提高32.25个百分点，相对KDA提高49.25个百分点。四个长度都领先KDA和其余模型。\n- single2：Learned+123在1K/2K/4K分别94.6/78.4/26.6%，均为本组最高；8K降到3.2%，低于Learned8.6%、KDA5.0%。不能把single1的8K表现推广到所有NIAH变体。\n- single3：所有模型基本失败，全部分项最高仅2.2%，不存在可用的解决方案。\n- +123的收益依赖底座：Block→Block+123的single1均值2.85→15.95%，但single2仅4.30→5.90%；Learned→Learned+123改善更大。当前实验无法把收益拆分到三个选项各自的因果贡献。\n- single1从4K的44.6%回升到8K的55.2%，而single2在8K突降，值得按needle深度、位置和失败输出分析；目前没有这些变化原因的验证证据。\n\nNIAH single1完整表：\n\n'
report+=task_table([f'niah_single_1_{n}' for n in [1024,2048,4096,8192]],['1K','2K','4K','8K'])
report+='\n\nNIAH single2完整表：\n\n'+task_table([f'niah_single_2_{n}' for n in [1024,2048,4096,8192]],['1K','2K','4K','8K'])
report+='\n\n## 2. BASED：KDA仍有优势，Learned+123主要追平SWDE\n\n'
report+=task_table(['based_swde','based_fda','based_nq_1024','based_drop'],['SWDE','FDA','NQ','DROP'])
report+='''

- SWDE：Learned+123 8.10% vs KDA 8.01%，差0.09个百分点，可描述为接近；没有显著性依据支持可靠超越。
- FDA：Block+123最高3.81%；Learned+123 2.09%，低于KDA 2.72%。
- NQ：KDA 10.23% vs Learned+123 8.24%，KDA领先1.996个百分点。
- DROP：KDA 15.52% vs Learned+123 14.14%，KDA领先1.390个百分点；Exact+123为14.66%。
- Learned+123相对Learned的四个BASED分项都提高，但尚未转化成对KDA的全面优势。BASED各分项保留自身原指标，不合成跨任务总分。

## 3. PhoneBook：Value/Learned接近，+123没有稳定增益

'''
report+=task_table([f'phonebook_{n}' for n in [32,64,100,128,150,200,256]],['32','64','100','128','150','200','256'])
report+='''

七档等权均值：Value24.53%，Learned24.39%，Learned+12323.84%，KDA21.46%；Value与Learned仅差0.14个百分点，不据此断言稳定优胜。Learned+123相对Learned在200/256档分别少1.1/1.4个百分点。

**协议限制：这里沿用历史single_break，实际prompt仅约80–244 tokens，不能用此表证明长上下文能力。** 不同档位的key/value分布与难度也需按原数据协议理解。

## 4. 对下一阶段的判断

1. 当前继续验证Learned+123的规模扩展有依据：NIAH single1优势大，single2在1K–4K也领先。1B结果仍需实测，不能由340M直接外推。
2. 下一次同口径评测应优先看single2 8K、single3及BASED NQ/DROP，检验短板是否随训练预算/模型规模改善。
3. Block/Value/Exact+123在这组NIAH上整体较弱；“数值更精确”本身没有在这些checkpoint上转化为更高分。Exact partial预算不足，单列参考，不用于公平排名。
4. 新Value+learned+123未纳入本plan；若要比较其贡献，需补齐同一23分项及相同推理协议。目前不能从Value或Learned+123单独推断组合结果。

## 5. 口径与证据

- NIAH每个分项500例，PhoneBook每档1000例；BASED SWDE1111、FDA1102、NQ3157、DROP2087例。
- 推理：FP32权重、BF16 autocast、无padding/无cache、完整前缀重算，无chat template/BOS，与固定原HFLM协议对齐。
- BASED按cap2048左截断到2000；NIAH的1K–8K是本次实际评测长度，不把配置上限当已测覆盖。
- 所有表格分数为百分比；差值为百分点。原数据及184行CSV保留完整精度。
- 最终核验快照UTC 2026-11-08 19:56:36，即北京时间2026-11-09 03:56:36；本次整理直接读取已完成的固定结果，未重跑评测。

文件：`FINAL_RESULTS.md`完整分项表；`FINAL_SCORES.csv`原精度得分；`evidence/final-verification.json`覆盖与哈希核验；`analysis/descriptive-summary.json`本次计算的描述性均值及分项最高分。\n
¹ Exact partial仅训练3.758B，其余模型6.8B。
'''
(ROOT/'RESULTS_ANALYSIS.md').write_text(report)
out=dict(verified_total_results=verified['total_results'],verified_items=verified['total_items'],within_task_equal_weight_means_percent=summary,case_winners_percent=case_winners,
 provenance={'final_scores_sha256':hashlib.sha256((ROOT/'FINAL_SCORES.csv').read_bytes()).hexdigest(),'verification_sha256':hashlib.sha256(evidence.read_bytes()).hexdigest(),'plan_sha':verified['plan_sha']})
(ROOT/'analysis/descriptive-summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('Wrote RESULTS_ANALYSIS.md and analysis/descriptive-summary.json; CSV matches final verification for all 184 rows')
