"""Report fixed-checkpoint KOMP decay trials, including negative results."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent/'decay_runs'
KINDS=('sigmoid','gdn_scalar','gsa_scalar','gsa_slot')
runs=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('*/summary.json'))]
lines=['# KOMP decay 消融结果','',f'完成 {len(runs)}/8 次预定训练。四组都使用原IEEE geometry、相同初始alpha=.98、独立beta=.2。',
       '每组相同10M唯一输入token、4层hidden128、LR配方、公共初始化；decay bias/时间尺度参数无weight decay。',
       '', '| variant | seed | confirm内容CE | Depo生成 | Brevo合法生成 | 常态step ms | 峰值GiB |',
       '|---|---:|---:|---:|---:|---:|---:|']
for r in runs:
    c=r['config'];kind=c['architecture'];seed=c['seed']
    assert r['unique_tokens']==10001108 and r['reload_validation_ce_error']<1e-4
    records=[json.loads(x) for x in (ROOT/'results'/f'{kind}-seed{seed}'/'metrics.jsonl').read_text().splitlines()]
    records=[x for x in records if x['step']>0]
    assert records[-1]['step']==2231
    ms=statistics.median(1000*(b['seconds']-a['seconds'])/(b['step']-a['step']) for a,b in zip(records,records[1:]) if 'val' not in a)
    lines.append(f"| {kind} | {seed} | {r['confirm']['mean_content_ce']:.4f} | {r['generation']['depo']['valid']:.2%} | {r['generation']['brevo']['valid']:.2%} | {ms:.2f} | {r['peak_allocated_bytes']/2**30:.2f} |")
if len(runs)==8:
    assert {(r['config']['architecture'],r['config']['seed']) for r in runs}=={(k,s) for k in KINDS for s in (731,732)}
    for seed in (731,732):
        shared=[r['config']['shared_parameter_sha256'] for r in runs if r['config']['seed']==seed]
        assert all(x==shared[0] for x in shared)
    lines+=['','## 两seed均值','', '| variant | 内容CE mean±SD | Depo生成 | Brevo合法生成 |','|---|---:|---:|---:|']
    for kind in KINDS:
        rr=[r for r in runs if r['config']['architecture']==kind]
        ce=[r['confirm']['mean_content_ce'] for r in rr]
        lines.append(f"| {kind} | {statistics.mean(ce):.4f} ± {statistics.stdev(ce):.4f} | {statistics.mean(r['generation']['depo']['valid'] for r in rr):.2%} | {statistics.mean(r['generation']['brevo']['valid'] for r in rr):.2%} |")
    lines+=['','## 分任务内容CE','', '| variant | Depo CE | Brevo CE |', '|---|---:|---:|']
    for kind in KINDS:
        rr=[r for r in runs if r['config']['architecture']==kind]
        values=[statistics.mean(r['confirm'][task]['content_ce'] for r in rr) for task in ('depo','brevo')]
        lines.append(f'| {kind} | {values[0]:.4f} | {values[1]:.4f} |')
    lines+=['','全部非decay参数的初始化逐seed hash一致；checkpoint回读验证通过。两seed是探索性证据，不足以证明统计显著性。',
            '常态step排除夹有验证的日志区间，包含数据准备、前反向和optimizer。峰值allocated包括训练和评估。',
            'gsa_scalar与gsa_slot用于分离门参数化和粒度；与旧GSA完整mixer结果不可混淆。',
            'slot与scalar初始数学函数在FP32通过原3e-4门槛；BF16下初始隐藏状态RMS相对差异约0.171%，未声称逐元素完全相同。',
            '', '![Validation curves](validation-curves.png)',
            '', '## 生成分桶','', '| variant | 任务 | hop或节点数 | 正确/总数（两个seed） |','|---|---|---:|---:|']
    for kind in KINDS:
        bins={}
        for p in (ROOT/'results').glob(kind+'-seed*/generations.json'):
            for r in json.loads(p.read_text()):
                key=(r['task'],r['meta']['query_k'] if r['task']=='depo' else r['meta']['n'])
                a,b=bins.get(key,(0,0));bins[key]=(a+int(r['valid']),b+1)
        for (task,n),(a,b) in sorted(bins.items()):lines.append(f'| {kind} | {task} | {n} | {a}/{b} |')
    lines+=['','## 最后一步门诊断','', '取最后一个验证minibatch，含padding；不是全验证集分布。',
            '', '| variant | seed | 参数量 | 各层平均alpha | 各层slot标准差 |', '|---|---:|---:|---|---|']
    for r in runs:
        c=r['config'];kind=c['architecture'];seed=c['seed']
        records=[json.loads(x) for x in (ROOT/'results'/f'{kind}-seed{seed}'/'metrics.jsonl').read_text().splitlines()]
        diag=[x['decay'] for x in records if 'decay' in x][-1]
        means=', '.join(f"{x['alpha_mean']:.3f}" for x in diag)
        std=', '.join(f"{x['slot_std']:.4f}" for x in diag) if kind=='gsa_slot' else '—'
        lines.append(f"| {kind} | {seed} | {c['params']:,} | {means} | {std} |")
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:24]))
