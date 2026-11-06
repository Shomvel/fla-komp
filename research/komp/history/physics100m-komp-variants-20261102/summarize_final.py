"""Final descriptive tables from phase=final JSON only; two-seed paired rankings."""
import json,statistics,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'results/latest-all/summary.json').read_text())
assert len(rows)==90 and all(r['test_ce'] is not None and r['tokens']>=100000000 for r in rows)
tasks=['mano','lano','depo2','brevo2','capo'];kinds=['gdn','gsa','gsa2','komp','full','swa','komp_learned','komp_gdn_slot','komp_learned_gdn_slot']
labels=dict(zip(kinds,['GDN','GSA','GSA2','KOMP C128','Full','SWA','KOMP learned','KOMP GDN-slot','KOMP learned＋GDN-slot']))
groups={}
for task in tasks:
 for kind in kinds:
  group=sorted([r for r in rows if r['task']==task and r['kind']==kind],key=lambda r:r['seed'])
  assert [r['seed'] for r in group]==[731,732]
  groups[task,kind]={m:statistics.mean(r[m] for r in group) for m in ['val_ce','test_ce','generation_percent']}
means=[dict(task=t,kind=k,**v) for (t,k),v in groups.items()]
ranks=[]
for metric in ['val_ce','test_ce','generation_percent']:
 for task in tasks:
  vs=[groups[task,k][metric] for k in kinds]
  for kind,v in zip(kinds,vs):
   better=sum(x>v+1e-12 if metric=='generation_percent' else x<v-1e-12 for x in vs);tie=sum(abs(x-v)<=1e-12 for x in vs)
   ranks.append(dict(task=task,kind=kind,metric=metric,mean=v,rank=1+better+(tie-1)/2,n=9))
(ROOT/'results/latest-all/means.json').write_text(json.dumps(means,indent=2))
(ROOT/'results/latest-all/ranks.json').write_text(json.dumps(ranks,indent=2))
lines=['# PhysicsLM：六架构基线与 KOMP 三变体最终结果','',
'90/90 个正式训练及完整评估已完成：基线 60 个，新增变体 30 个。以下为每个任务 100M token、seed 731/732 的均值；每个 seed 使用 256 条生成探针。10M 调参成绩不进入表格。','',
'KOMP learned 在 Mano、Brevo2 的平均生成正确率领先；原 KOMP 在 Lano 领先；GDN 在 Depo2 领先；KOMP GDN-slot 在 Capo 领先。两个 seed 只作描述性比较，不代表统计显著性。','']
for metric,title in [('generation_percent','生成正确率（%，越高越好）'),('test_ce','测试 CE（越低越好）'),('val_ce','验证 CE（越低越好）')]:
 lines += ['## '+title,'','| 架构 | Mano | Lano | Depo2 | Brevo2 | Capo |','|---|---:|---:|---:|---:|---:|']
 for kind in kinds:
  values=[groups[t,kind][metric] for t in tasks];fmt='.2f' if metric=='generation_percent' else '.6f'
  lines.append('| '+labels[kind]+' | '+' | '.join(format(v,fmt) for v in values)+' |')
 lines.append('')
lines += ['## 两个 seed 的完整分数','', '[逐 run 的验证 CE、测试 CE、生成正确率、LR/WD、训练耗时和配对差值](results/latest-all/TABLE.md)。JSON：`results/latest-all/summary.json`、`means.json`、`ranks.json`。排名按未舍入均值计算，并列采用平均名次。','',
'## 审计与可比性','',
'- 原 60 个 run 的 15360 条生成评分已独立复核；新增 30 个 run 的 7680 条也使用独立评分实现复核。两者合计 23040 条，无差异。Capo 审计采用保存的目标 token；新增实验修复路径时另核对了 256 条 Capo 探针的重新生成一致性。测试 CE 未从 checkpoint 独立重算。',
'- 三变体各使用 9 组 LR/WD，在每任务 10M 验证集选参后重新进行 100M 正式训练；seed 对齐。完整选择记录保存在归档中。',
'- 原 KOMP 已有输入相关 GSA per-slot decay。learned 变体同时增加 hard top-M 淘汰和写入门控，因此提升不能单独归因于淘汰；GDN-slot 替换衰减参数化，并将 timescale 参数排除 WD。',
'- 共享 backbone、卷积和字典初始化保持一致；固定 FFN 宽度，附加参数未用缩小 FFN 抵消。原冻结训练/kernel 源码未修改。',
'- Mano 最长 38 token，低于 SWA 窗口 96 和 KOMP 分块 128；此任务无法检验跨分块保留能力。',
'- 评估因四小时作业期限曾重启；随后增加每 8 条生成的原子断点保存。生成批量、停止规则、评分保持不变。耗时字段不含旧实现丢失的部分尝试，不宜用于完整成本比较。','',
'## 证据','',
'- 最终原始 JSON：`evidence/monitor-20261102T1049Z.json`。',
'- 独立评分：`evidence/final-generation-audit.json`；可复用脚本在原实验 `audit_final.py`。',
'- 日志、165 个训练 summary、15 个选参文件、30 组 heldout/generations：`final_archive/`，逐文件哈希见 `ARCHIVE_MANIFEST.json`；不含 checkpoint。',
'- 作业终态、精确删除及消失证明：`evidence/cleanup-final-20261102T1049Z/`。最终完成收据：`evidence/VARIANTS_COMPLETE.json`。']
(ROOT/'FINAL_RESULTS.md').write_text('\n'.join(lines)+'\n')
print('Generation means:')
for kind in kinds:print(labels[kind],*[f'{groups[t,kind]["generation_percent"]:.4f}' for t in tasks])
