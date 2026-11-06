"""Summarize the complete prespecified paired trials and verify their receipts."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent / 'shortconv_runs'
KINDS = ('komp_slot', 'gsa', 'gdn')
LABELS = {'komp_slot': 'GSA-slot KOMP', 'gsa': 'GSA', 'gdn': 'GDN'}
runs = {}
for path in sorted((ROOT / 'results').glob('*/summary.json')):
    summary = json.loads(path.read_text())
    config = summary['config']
    key = (config['mixer'], config['use_short_conv'], config['seed'])
    assert key not in runs
    assert summary['status'] == 'completed' and summary['unique_tokens'] == 10001108
    assert summary['reload_validation_ce_error'] < 1e-4
    causal = json.loads((path.parent / 'trained-causality.json').read_text())
    assert causal['status'] == 'passed' and all(c['prefix_max_abs'] == 0 for c in causal['checks'])
    generated = json.loads((path.parent / 'generations.json').read_text())
    for task in ('depo', 'brevo'):
        rows = [r for r in generated if r['task'] == task]
        assert len(rows) == 128
        assert sum(r['valid'] for r in rows) / 128 == summary['generation'][task]['valid']
    metrics = [json.loads(line) for line in (path.parent / 'metrics.jsonl').read_text().splitlines()]
    metrics = [m for m in metrics if m['step'] > 0]
    assert metrics[-1]['step'] == 2231
    summary['step_ms'] = statistics.median(1000 * (b['seconds'] - a['seconds']) / (b['step'] - a['step'])
                                           for a, b in zip(metrics, metrics[1:]) if 'val' not in a)
    summary['generations'] = generated
    runs[key] = summary
assert set(runs) == {(k, c, s) for k in KINDS for c in (False, True) for s in (731, 732)}, 'Incomplete experiment'
for seed in (731, 732):
    conv_hashes = []
    for kind in KINDS:
        plain, conv = (runs[kind, setting, seed]['config'] for setting in (False, True))
        assert plain['nonconv_initial_sha256'] == conv['nonconv_initial_sha256']
        assert conv['params'] - plain['params'] == 6144
        conv_hashes.append(conv['conv_initial_sha256'])
    assert all(h == conv_hashes[0] for h in conv_hashes)


def mean(kind, conv, fn):
    return statistics.mean(fn(runs[kind, conv, seed]) for seed in (731, 732))


lines = ['# Short conv 配对实验结果', '',
         '完成12/12次预定训练：三种mixer × 有/无卷积 × 两个seed。每run相同10,001,108唯一输入token、2231steps、固定LR与数据顺序，使用最后一步checkpoint。', '',
         '卷积组在q/k/v投影后加宽度4的FLA causal depthwise convolution + SiLU，无bias。每模型增加6144参数；卷积初值跨架构相同，配对组全部非卷积参数初值相同。', '',
         '| 模型 | short conv | 内容CE ↓ | Depo生成 ↑ | Brevo合法生成 ↑ | 常态step ms |',
         '|---|---|---:|---:|---:|---:|']
aggregate = []
for kind in KINDS:
    for conv in (False, True):
        ce = mean(kind, conv, lambda r: r['confirm']['mean_content_ce'])
        depo = mean(kind, conv, lambda r: r['generation']['depo']['valid'])
        brevo = mean(kind, conv, lambda r: r['generation']['brevo']['valid'])
        ms = mean(kind, conv, lambda r: r['step_ms'])
        lines.append(f'| {LABELS[kind]} | {"有" if conv else "无"} | {ce:.4f} | {depo:.2%} | {brevo:.2%} | {ms:.2f} |')
        aggregate.append({'mixer': kind, 'conv': conv, 'content_ce': ce, 'depo': depo, 'brevo': brevo, 'step_ms': ms})
lines += ['', '## 配对变化', '', '| 模型 | 内容CE变化 | Depo变化（百分点） | Brevo变化（百分点） |', '|---|---:|---:|---:|']
for kind in KINDS:
    values = []
    for fn in (lambda r: r['confirm']['mean_content_ce'], lambda r: r['generation']['depo']['valid'] * 100,
               lambda r: r['generation']['brevo']['valid'] * 100):
        values.append(mean(kind, True, fn) - mean(kind, False, fn))
    lines.append(f'| {LABELS[kind]} | {values[0]:+.4f} | {values[1]:+.2f} | {values[2]:+.2f} |')
lines += ['', '## 每个seed', '', '| 模型 | conv | seed | 参数量 | 内容CE | Depo生成 | Brevo合法生成 |', '|---|---|---:|---:|---:|---:|---:|']
for (kind, conv, seed), r in runs.items():
    lines.append(f"| {LABELS[kind]} | {conv} | {seed} | {r['config']['params']:,} | {r['confirm']['mean_content_ce']:.4f} | {r['generation']['depo']['valid']:.2%} | {r['generation']['brevo']['valid']:.2%} |")
lines += ['', '## 分任务CE', '', '| 模型 | conv | Depo CE | Brevo CE |', '|---|---|---:|---:|']
for kind in KINDS:
    for conv in (False, True):
        depo = mean(kind, conv, lambda r: r['confirm']['depo']['content_ce'])
        brevo = mean(kind, conv, lambda r: r['confirm']['brevo']['content_ce'])
        lines.append(f'| {LABELS[kind]} | {conv} | {depo:.4f} | {brevo:.4f} |')
lines += ['', '## 验证与结论范围', '',
          '- 卷积FP32/BF16前反向通过独立Torch左padding reference；过去输出对未来输入梯度为0，后缀干预不影响前缀；流式缓存对齐通过。',
          '- 六个初始模型及12个最终checkpoint各13个截断点的未来token干预差异为0；所有checkpoint回读CE一致性通过。',
          '- Brevo自由生成只以prompt初始化，合法性从图边求祖先闭包与拓扑序；没有用gold答案集合判定。',
          '- 因果审计验证的是本次测试范围，不宣称任意shape与输入均无bug。',
          '- 保持既有门公式：KOMP GSA-slot使用logsigmoid/8；GSA使用前轮logsigmoid/1；GDN使用前轮sigmoid scalar gate/beta。本实验仍不是原生完整GSA/GDN网络复现。',
          '- 所有组统一decay.bias无weight decay，所以GSA/GDN无卷积组也重新训练，直接与本轮配对组比较。',
          '- 测量的是conv+SiLU整体增益，不能单独归因于卷积；只有两个seed，不证明稳定统计优势或大规模泛化。',
          '- 常态step包含数据准备、前反向和optimizer，排除周期验证区间；同一架构配对运行在同一GPU，跨架构并行运行，不能当作同卡算子benchmark。',
          '- KOMP使用冻结的训练geometry，不包括后续性能优化。checkpoint保存在远端独立实验目录，报告/日志/逐样本生成与源码指纹保存在本地证据目录。']
(ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
(ROOT / 'aggregate.json').write_text(json.dumps(aggregate, indent=2))
print('\n'.join(lines[:27]))
