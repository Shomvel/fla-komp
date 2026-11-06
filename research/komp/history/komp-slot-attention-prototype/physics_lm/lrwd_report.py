"""Read completed search/scale-up evidence without selecting on held-out scores."""
import json
import os
from pathlib import Path
import statistics

ROOT = Path(os.environ.get('STUDY_REPORT_ROOT', Path(__file__).resolve().parent / 'lrwd_runs'))
KINDS = tuple(os.environ.get('STUDY_REPORT_KINDS', 'komp_slot,gsa,gdn').split(','))
LABELS = {'komp_slot': 'GSA-slot KOMP', 'gsa': 'GSA', 'gdn': 'GDN', 'gsa2': 'GSA2 (native FLA)'}
LRS = (2.5e-4, 5e-4, 1e-3)
WDS = (0., .01, .1)


def name(kind, seed, lr, wd, scale='small'):
    return f'{kind}-{scale}-lr{lr:g}-wd{wd:g}-seed{seed}'


def read_trial(trial):
    dest = ROOT / 'results' / trial
    result = json.loads((dest / 'summary.json').read_text())
    assert result['status'] == 'completed' and result['reload_validation_ce_error'] == 0
    assert result['unique_tokens'] == result['config']['data']['splits']['train']['tokens']
    assert all(c['prefix_max_abs'] == 0 for c in result['trained_causality'])
    path = dest / 'heldout.json'
    if path.exists():
        result['heldout'] = json.loads(path.read_text())
    records = [json.loads(line) for line in (dest / 'metrics.jsonl').read_text().splitlines()]
    records = [r for r in records if r['step'] > 0]
    assert records[-1]['step'] == result['config']['steps']
    result['step_ms'] = statistics.median(1000 * (b['seconds'] - a['seconds']) / (b['step'] - a['step'])
                                         for a, b in zip(records, records[1:]) if 'val' not in a)
    return result


def main():
    selection = json.loads((ROOT / 'selection.json').read_text())
    lines = ['# LR/WD 搜索与十倍模型/数据实验', '',
             f'所有模型开启q/k/v宽度4的causal depthwise conv+SiLU。小模型每架构9点搜索，共{len(KINDS)*9}runs；只用seed731最后一步完整验证集mean content CE选参，然后seed732复验。', '',
             '## 小模型验证集搜索', '',
             '| 峰值LR | WD | ' + ' | '.join(LABELS[k]+'验证CE' for k in KINDS) + ' |',
             '|' + '---:|' * (2 + len(KINDS))]
    for lr in LRS:
        for wd in WDS:
            values = [read_trial(name(k, 731, lr, wd))['val']['mean_content_ce'] for k in KINDS]
            lines.append(f'| {lr:g} | {wd:g} | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
    lines += ['', '## 锁定配置', '', '| 模型 | LR | WD | 验证CE：原配置→选定配置（seed731） |', '|---|---:|---:|---|']
    for kind in KINDS:
        selected = selection['models'][kind]['selected']
        candidates = selection['models'][kind]['candidates']
        assert selected == min(candidates, key=lambda r: (r['val_ce'], r['lr'], r['wd']))
        original = read_trial(name(kind, 731, 5e-4, .1))['val']['mean_content_ce']
        lines.append(f"| {LABELS[kind]} | {selected['lr']:g} | {selected['wd']:g} | {original:.4f} → {selected['val_ce']:.4f} |")
    lines += ['', '## 小模型选定配置：两seed确认集', '',
              '| 模型 | 内容CE | Depo生成 | Brevo合法生成 |', '|---|---:|---:|---:|']
    selected_runs = {}
    for kind in KINDS:
        selected = selection['models'][kind]['selected']
        runs = [read_trial(name(kind, seed, selected['lr'], selected['wd'])) for seed in (731, 732)]
        selected_runs[kind] = runs
        if all('heldout' in r for r in runs):
            ce = statistics.mean(r['heldout']['confirm']['mean_content_ce'] for r in runs)
            depo = statistics.mean(r['heldout']['generation']['depo']['valid'] for r in runs)
            brevo = statistics.mean(r['heldout']['generation']['brevo']['valid'] for r in runs)
            lines.append(f'| {LABELS[kind]} | {ce:.4f} | {depo:.2%} | {brevo:.2%} |')
        else:
            lines.append(f'| {LABELS[kind]} | 评估未完成 | — | — |')
    lines += ['', '## 扩容：15层hidden512、100M唯一token（seed731）', '',
              '| 模型 | 参数量 | 唯一token | 内容CE | Depo生成 | Brevo合法生成 | step ms | 峰值GiB |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    large = {}
    for kind in KINDS:
        selected = selection['models'][kind]['selected']
        trial = name(kind, 731, selected['lr'], selected['wd'], 'large')
        if not (ROOT / 'results' / trial / 'summary.json').exists():
            lines.append(f'| {LABELS[kind]} | 未完成 | — | — | — | — | — | — |')
            continue
        r = read_trial(trial)
        large[kind] = r
        if 'heldout' in r:
            held = r['heldout']
            lines.append(f"| {LABELS[kind]} | {r['config']['params']:,} | {r['unique_tokens']:,} | {held['confirm']['mean_content_ce']:.4f} | {held['generation']['depo']['valid']:.2%} | {held['generation']['brevo']['valid']:.2%} | {r['step_ms']:.2f} | {r['peak_allocated_bytes']/2**30:.2f} |")
        else:
            lines.append(f"| {LABELS[kind]} | {r['config']['params']:,} | {r['unique_tokens']:,} | 评估未完成 | — | — | {r['step_ms']:.2f} | {r['peak_allocated_bytes']/2**30:.2f} |")
    lines += ['', '## 大模型确认集：分任务 teacher-forcing 内容CE', '',
              '| 模型 | Depo CE | Brevo CE | 两任务等权均值 |',
              '|---|---:|---:|---:|']
    for kind, result in large.items():
        if 'heldout' not in result:
            continue
        confirm = result['heldout']['confirm']
        lines.append(f"| {LABELS[kind]} | {confirm['depo']['content_ce']:.4f} | {confirm['brevo']['content_ce']:.4f} | {confirm['mean_content_ce']:.4f} |")
    lines += ['', '内容CE在给定真实前缀时计算，生成正确率要求模型从题目自主生成完整答案，二者度量不同。',
              '', '## 同seed731扩容前后', '',
              '| 模型 | 内容CE：小→大 | Depo生成：小→大 | Brevo合法生成：小→大 |',
              '|---|---|---|---|']
    for kind, big in large.items():
        small = selected_runs[kind][0]
        if 'heldout' not in big or 'heldout' not in small:
            continue
        a, b = small['heldout'], big['heldout']
        lines.append(f"| {LABELS[kind]} | {a['confirm']['mean_content_ce']:.4f} → {b['confirm']['mean_content_ce']:.4f} | "
                     f"{a['generation']['depo']['valid']:.2%} → {b['generation']['depo']['valid']:.2%} | "
                     f"{a['generation']['brevo']['valid']:.2%} → {b['generation']['brevo']['valid']:.2%} |")
    lines += ['', '## 大模型生成正确例数与难度分桶', '',
              '| 模型 | Depo正确/128 | Brevo合法/128 | Brevo 3–5节点 | 6–10节点 | 11–24节点 |',
              '|---|---:|---:|---:|---:|---:|']
    for kind, result in large.items():
        if 'heldout' not in result:
            continue
        selected = selection['models'][kind]['selected']
        path = ROOT / 'results' / name(kind, 731, selected['lr'], selected['wd'], 'large') / 'generations.json'
        records = json.loads(path.read_text())
        counts = [sum(r['valid'] for r in records if r['task'] == task) for task in ('depo', 'brevo')]
        bins = []
        for low, high in ((3, 5), (6, 10), (11, 24)):
            rows = [r for r in records if r['task'] == 'brevo' and low <= r['meta']['n'] <= high]
            bins.append(f"{sum(r['valid'] for r in rows)}/{len(rows)}")
        lines.append(f'| {LABELS[kind]} | {counts[0]}/128 | {counts[1]}/128 | ' + ' | '.join(bins) + ' |')
    lines += ['', '## Brevo 失败类型（事后分析）', '',
              '| 模型 | 未正常结束 | 节点集合错误或重复 | 节点完整但顺序错误 |',
              '|---|---:|---:|---:|']
    for kind, result in large.items():
        if 'heldout' not in result:
            continue
        selected = selection['models'][kind]['selected']
        path = ROOT / 'results' / name(kind, 731, selected['lr'], selected['wd'], 'large') / 'generations.json'
        records = json.loads(path.read_text())
        failures = [0, 0, 0]
        for record in records:
            if record['task'] != 'brevo' or record['valid']:
                continue
            guess, gold = record['guess'], record['gold']
            assert gold[-1] == 9998
            if not guess or guess[-1] != 9998:
                failures[0] += 1
            elif set(guess[:-1]) != set(gold[:-1]) or len(guess[:-1]) != len(set(gold[:-1])):
                failures[1] += 1
            else:
                failures[2] += 1
        assert sum(failures) == sum(r['task'] == 'brevo' and not r['valid'] for r in records)
        lines.append(f'| {LABELS[kind]} | ' + ' | '.join(map(str, failures)) + ' |')
    lines += ['', '按未结束、节点集合/重复、仅顺序错误依次归类，三类互斥。合法性已由prompt图的祖先闭包和拓扑约束独立复核；此表利用保存的答案节点集合细分失败，不用于选参，也不能单独定位模型内部原因。']
    if all(k in large and 'heldout' in large[k] for k in ('komp_slot', 'gdn')):
        paired = []
        for kind in ('komp_slot', 'gdn'):
            selected = selection['models'][kind]['selected']
            path = ROOT / 'results' / name(kind, 731, selected['lr'], selected['wd'], 'large') / 'generations.json'
            paired.append(json.loads(path.read_text()))
        assert len(paired[0]) == len(paired[1])
        assert all(a['task'] == b['task'] and a['gold'] == b['gold'] and a['meta'] == b['meta']
                   for a, b in zip(*paired))
        lines += ['', '## KOMP 与 GDN 逐题比较（seed731）', '',
                  '| 任务 | 两者都正确 | 仅KOMP正确 | 仅GDN正确 | 两者都错 |',
                  '|---|---:|---:|---:|---:|']
        for task in ('depo', 'brevo'):
            pairs = [(a, b) for a, b in zip(*paired) if a['task'] == task]
            counts = [sum(bool(a['valid']) == x and bool(b['valid']) == y for a, b in pairs)
                      for x, y in ((True, True), (True, False), (False, True), (False, False))]
            lines.append(f'| {task} | ' + ' | '.join(map(str, counts)) + ' |')
    lines += ['', '## 解释范围', '',
              '- 搜索与选择阶段只读取train/val；test/confirm在配置锁定后评估。选参标准是内容CE，不是生成成功率，因此选定配置不保证每个任务生成率都提高。',
              '- 每架构只做9点有限网格；若最优点落在边界，不代表全局最优，不自动扩展网格。',
              '- 大模型采用小模型选定配置直接迁移，单seed。模型和训练数据同时扩十倍，不能分离两者各自贡献。',
              '- “十倍模型”按总参数量定义，词表和embedding占比不同导致非embedding参数增幅不同；head维度64。GSA2用原生64slots，其余三组32slots（GDN直接矩阵状态）。',
              '- 100M训练集包含原10M并追加去重样本；排除固定评测集及更早实验文档。任务难度分布不变，未通过重复epoch凑token数。',
              '- 仍是既有统一骨干mixer对照，GDN/GSA门参数化没有改成完整上游网络默认配方。',
              '- GSA2是用户要求直接加入的原生FLA layer，保留原生初始化、四条卷积、门公式与WD豁免。每head8192个递归状态元素，是前三组4096的两倍；骨干尺寸一致但参数量不同，不是等状态/等参数消融。',
              '- 验证、测试和确认集沿用历史固定split，已用于此前探索，不属于从未查看的新测试集。',
              '- 所有已完成run均检查token预算、有限梯度、最终checkpoint回读及13个前缀因果干预；实际记录见各run的manifest/metrics/summary。']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    (ROOT / 'report-status.json').write_text(json.dumps({'small_selected': {k: all('heldout' in r for r in rr) for k, rr in selected_runs.items()},
                                                       'large': {k: 'heldout' in r for k, r in large.items()}}, indent=2))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
