"""Summarize only completed matched runs; do not rank partial trials."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
runs=[]
for p in sorted((ROOT/'results').glob('*/summary.json')):
    r=json.loads(p.read_text())
    if r.get('status')=='completed':runs.append(r)
lines=['# Depo/Brevo 匹配小模型结果','',f'已完成 {len(runs)}/6 次有效训练；每个run完整遍历同一10M-token数据。GSA使用修复slot初始化后的r3，GDN/KOMP使用r2；失败对照不计入此表。',
       '', '| 模型 | seed | 参数量 | confirm内容CE | Depo内容准确率 | Brevo内容准确率 | Depo生成 | Brevo合法生成 | 秒 |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in runs:
    c=r['config'];e=r['confirm'];g=r['generation']
    lines.append(f"| {c['architecture']} | {c['seed']} | {c['params']:,} | {e['mean_content_ce']:.4f} | {e['depo']['content_accuracy']:.2%} | {e['brevo']['content_accuracy']:.2%} | {g['depo']['valid']:.2%} | {g['brevo']['valid']:.2%} | {r['elapsed_seconds']:.1f} |")
if len(runs)==6:
    assert {(r['config']['architecture'],r['config']['seed']) for r in runs}=={(k,s) for k in ('gdn','gsa','komp') for s in (731,732)}
    assert all(r['unique_tokens']==10001108 and r['reload_validation_ce_error']<1e-4 for r in runs)
    assert all(r['config'].get('gsa_slot_initialization','').startswith('independent') for r in runs if r['config']['architecture']=='gsa')
    lines+=['','## 两个 seed 的平均值','', '| 模型 | 内容CE mean±SD | Depo生成均值 | Brevo合法生成均值 |', '|---|---:|---:|---:|']
    for kind in ('gdn','gsa','komp'):
        rr=[r for r in runs if r['config']['architecture']==kind]
        ce=[r['confirm']['mean_content_ce'] for r in rr]
        lines.append(f"| {kind} | {statistics.mean(ce):.4f} ± {statistics.stdev(ce):.4f} | {statistics.mean(r['generation']['depo']['valid'] for r in rr):.2%} | {statistics.mean(r['generation']['brevo']['valid'] for r in rr):.2%} |")
    for seed in (731,732):
        hashes=[r['config']['common_init_sha256'] for r in runs if r['config']['seed']==seed]
        assert all(x==hashes[0] for x in hashes)
    lines+=['','## 完整模型训练吞吐','',
            '| 模型 | seed | 常态 step 毫秒 | 非padding token/s | 峰值 allocated GiB | 回读CE误差 |',
            '|---|---:|---:|---:|---:|---:|']
    for r in runs:
        c=r['config'];kind=c['architecture'];seed=c['seed']
        records=[json.loads(x) for x in (ROOT/'results'/f'{kind}-seed{seed}'/'metrics.jsonl').read_text().splitlines()]
        records=[x for x in records if x['step']>0]
        assert records[-1]['step']==2231 and records[-1]['unique_tokens']==10001108
        intervals=[(b['seconds']-a['seconds'],b['step']-a['step'],b['unique_tokens']-a['unique_tokens'])
                   for a,b in zip(records,records[1:]) if 'val' not in a]
        step_ms=statistics.median(1000*t/s for t,s,n in intervals)
        throughput=statistics.median(n/t for t,s,n in intervals)
        lines.append(f"| {kind} | {seed} | {step_ms:.2f} | {throughput:,.0f} | {r['peak_allocated_bytes']/2**30:.2f} | {r['reload_validation_ce_error']:.2g} |")
    lines+=['','吞吐取连续日志区间的中位数，排除夹有验证的区间及首次编译；包含数据组batch、前反向和optimizer，未包含周期验证及生成。峰值allocated覆盖训练和评估，不是专门的训练显存microbenchmark。三个架构分别占一张H100，端到端吞吐不是同卡串行算子速度比较。']
    lines+=['','记录的共同模块初始化SHA-256逐seed一致。所有run固定同一LR配方、数据顺序、token预算；未按确认集选模型。两seed仅供探索，不足以证明稳定统计优势。',
            '', 'KOMP采用循环硬路由，并非完整最小删除代价KOMP。小模型、特定训练配方和合成任务结论不能直接推广到语言模型容量或大规模预训练。',
            '', '旧LR研究的181M结果不参与此表；旧FLA快照有已记录的换底接口问题，本研究使用新的干净commit。',
            '', '三种mixer嵌入统一骨干，GSA也采用L2-normalized q/k；GDN/KOMP均为sigmoid scalar gate，没有原生GDN的A_log/dt_bias；并非上游原生完整网络复现。先前GSA零门权重导致slot对称，原结果移至excluded-symmetric-gsa。详见PLAN.md的实施审计。',
            '', '![Validation curves](validation-curves.png)']
    # Break down free generation by actual first-query hops or graph node count.
    lines+=['','## 生成分桶','', '| 模型 | 任务 | 难度桶 | 正确/总数（两seed合并） |', '|---|---|---|---:|']
    for kind in ('gdn','gsa','komp'):
        bins={}
        for p in (ROOT/'results').glob(kind+'-seed*/generations.json'):
            for rec in json.loads(p.read_text()):
                key=(rec['task'],rec['meta'].get('query_k') if rec['task']=='depo' else rec['meta'].get('n','unknown'))
                a,b=bins.get(key,(0,0));bins[key]=(a+int(rec['valid']),b+1)
        for (task,difficulty),(a,b) in sorted(bins.items(),key=lambda x:str(x[0])):
            lines.append(f'| {kind} | {task} | {difficulty} | {a}/{b} |')
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:18]))
