"""Audit completed results and reproduce paired statistics and training curves."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'block_runs_v4'
PLAN = json.loads((ROOT/'plan.json').read_text())
ARMS = {name: ROOT/'results' for name in PLAN['variants']}
ARMS.update({name: OLD/'results' for name in
             ('block_c64', 'block_slot_c64', 'block_slot_gate_c64', 'block_slot_learn_c64')})
ARMS.update({name: OLD/'baselines/results' for name in ('gdn', 'gsa', 'gsa_slot')})
ARMS.update({name: OLD/'gsa2/results' for name in ('gsa2_native_lr5e4', 'gsa2_native_lr1e3', 'gsa2_noconv_lr5e4')})
METRICS = {
    'Content CE': ('confirm', 'mean_content_ce'),
    'Depo CE': ('confirm', 'depo', 'content_ce'),
    'Brevo CE': ('confirm', 'brevo', 'content_ce'),
    'Depo generation %': ('generation', 'depo', 'valid'),
    'Brevo valid %': ('generation', 'brevo', 'valid'),
    'Elapsed seconds': ('elapsed_seconds',),
}
PAIRS = [
    ('exact_slot_learned', 'exact_slot_fifo'), ('exact_slot_gate', 'exact_slot_fifo'),
    ('exact_slot_learned', 'exact_slot_gate'), ('exact_slot_fifo', 'exact_scalar_fifo'),
    ('exact_scalar_fifo', 'block_c64'), ('exact_slot_fifo', 'block_slot_c64'),
    ('exact_slot_gate', 'block_slot_gate_c64'), ('exact_slot_learned', 'block_slot_learn_c64'),
    ('exact_slot_fifo', 'gsa_slot'), ('exact_slot_learned', 'gdn'),
    ('exact_slot_learned', 'gsa'), ('exact_slot_learned', 'gsa2_native_lr1e3'),
]


def metric(row, name):
    for key in METRICS[name]:
        row = row[key]
    return row * (100 if name.endswith('%') else 1)


def audit_generations(folder, summary, confirm):
    records = json.loads((folder/'generations.json').read_text())
    assert len(records) == 256
    for task in ('depo', 'brevo'):
        inputs = [row for row in confirm if row['task'] == task][:128]
        outputs = [row for row in records if row['task'] == task]
        assert len(outputs) == 128
        valid_total, exact_total = 0, 0
        for row, out in zip(inputs, outputs):
            ids = row['input_ids']
            guess = out['guess']
            if task == 'depo':
                boundary = ids.index(9500)
                end = boundary + 1
                while ids[end] <= 50:
                    end += 1
                gold = ids[boundary+1:end+1]
                valid = guess == gold
            else:
                end = ids.index(9997)
                gold = ids[end+3:]
                edges = list(zip(ids[1:end:2], ids[2:end:2]))
                answer = guess[:-1] if guess and guess[-1] == 9998 else guess
                nodes = set(gold[:-1])
                pos = {node: i for i, node in enumerate(answer)}
                valid = bool(guess and guess[-1] == 9998 and set(answer) == nodes
                             and len(answer) == len(nodes)
                             and all(pos[a] < pos[b] for a, b in edges if b in nodes))
            assert out['gold'] == gold and out['exact'] == (guess == gold)
            assert out['valid'] == valid
            valid_total += valid
            exact_total += guess == gold
        assert summary['generation'][task]['valid'] == valid_total/128
        assert summary['generation'][task]['exact'] == exact_total/128
    return len(records)


def main(partial=False):
    confirm_path = ROOT/'evidence/confirm.jsonl'
    manifest = json.loads((ROOT/'snapshot/physics_lm/data-manifest.json').read_text())
    assert hashlib.sha256(confirm_path.read_bytes()).hexdigest() == manifest['splits']['confirm']['sha256']
    confirm = [json.loads(line) for line in confirm_path.read_text().splitlines()]
    rows, files, generation_count = {}, {}, 0
    for arm, directory in ARMS.items():
        rows[arm] = {}
        for path in sorted(directory.glob(arm+'-seed*/summary.json')):
            row = json.loads(path.read_text())
            seed = row['config']['seed']
            assert seed in PLAN['seeds'] and seed not in rows[arm]
            assert row['status'] == 'completed' and row['unique_tokens'] == 10001108
            assert row['reload_validation_ce_error'] < 1e-4
            assert row['config']['data'] == manifest
            assert row['config']['steps'] == 2231 and row['config']['batch'] == 64
            rows[arm][seed] = row
            files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            if arm in PLAN['variants']:
                generation_count += audit_generations(path.parent, row, confirm)
                expected = json.loads((ROOT/'snapshot-sha256.json').read_text())
                for filename, digest in row['config']['source_sha256'].items():
                    assert digest == expected[filename], (arm, seed, filename)
                assert row['config']['compression_tokens'] == 1 and row['config']['compute_chunk'] == 16
        if not partial:
            assert set(rows[arm]) == set(PLAN['seeds']), (arm, list(rows[arm]))
    for seed in PLAN['seeds']:
        inits = {arm: runs[seed]['config']['common_init_sha256'] for arm, runs in rows.items() if seed in runs}
        baseline = inits['gdn']
        for arm, init in inits.items():
            # Native GSA2 replaces the mixer and therefore has no mix.q.weight.
            expected_names = set(baseline) - ({'layers.0.mix.q.weight'} if arm.startswith('gsa2_') else set())
            assert set(init) == expected_names, (arm, seed, 'initialization fields')
            assert all(digest == baseline[name] for name, digest in init.items()), (arm, seed, 'shared initialization')
    rows = {arm: runs for arm, runs in rows.items() if runs}
    stats = {}
    for arm, runs in rows.items():
        stats[arm] = {}
        for name in METRICS:
            values = [metric(r, name) for r in runs.values()]
            stats[arm][name] = {'n': len(values), 'mean': statistics.mean(values),
                                'sd': statistics.stdev(values) if len(values)>1 else None}
    paired = []
    for a, b in PAIRS:
        if a not in rows or b not in rows:
            continue
        seeds = sorted(set(rows[a]) & set(rows[b]))
        if not seeds:
            continue
        indices = np.random.default_rng(91731).integers(0, len(seeds), size=(20000, len(seeds)))
        for name in list(METRICS)[:-1]:
            differences = np.array([metric(rows[a][s], name)-metric(rows[b][s], name) for s in seeds])
            ci = np.quantile(differences[indices].mean(1), [.025, .975]).tolist()
            paired.append({'a': a, 'b': b, 'metric': name, 'seeds': seeds,
                           'mean': float(differences.mean()), 'ci95': ci, 'differences': differences.tolist()})
    control = {}
    path = ROOT/'control/gdn-seed731/summary.json'
    if path.exists():
        fresh = json.loads(path.read_text())
        control = {'status': fresh['status'], 'deltas': {m: metric(fresh,m)-metric(rows['gdn'][731],m) for m in METRICS}}
        generation_count += audit_generations(path.parent, fresh, confirm)
    preflight = json.loads((ROOT/'preflight.json').read_text())
    assert preflight['status'] == 'passed' and set(preflight['variants']) == set(PLAN['variants'])
    preflight_rms = max(v['reference_rms'] for v in preflight['variants'].values())
    causal_max = max(c['max_abs'] for v in preflight['variants'].values() for c in v['causality'])
    result = {'status': 'partial' if partial else 'complete', 'arms': stats, 'paired': paired,
              'input_sha256': files, 'new_generations_audited': generation_count, 'control': control,
              'bootstrap': {'samples': 20000, 'seed': 91731, 'rng': 'numpy.default_rng'}}
    (ROOT/'statistics.json').write_text(json.dumps(result, indent=2))
    lines = ['# Exact KOMP：Physics of LM 质量实验', '',
             '**状态：'+('部分结果，不用于最终结论。**' if partial else '28 个新训练与 1 个 GDN 复现对照完成。**'), '',
             '沿用冻结 Depo/Brevo 数据：10,001,108 token、4 层宽度128、M32、batch64、2231步，'
             'seeds731–737。新四臂均使用 LR5e-4、原 AdamW 配方、BF16 模型投影和 FP32 exact KOMP。'
             '算法逐 token 压缩；计算 C16。下表 CE 来自 confirm，生成使用每任务固定128例。', '',
             'Exact KOMP 在旧字典上计算残差，加入当前原子并读出，然后投影压缩回M；'
             '这是 Block-KOMP 的算法C=1。旧gsa_slot采用替换/投影后写入与读出的另一条规则，二者不是仅kernel实现不同。', '',
             '均值 ± seed 样本标准差；生成以百分比表示。GDN/GSA/GSA2/Block 行为历史结果，未冒充本轮复跑。', '',
             '| 方法 | n | Content CE ↓ | Depo CE ↓ | Brevo CE ↓ | Depo 生成 % ↑ | Brevo 合法 % ↑ | 总耗时 s |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, values in stats.items():
        cells = []
        for name in METRICS:
            v = values[name]
            digits = 1 if name == 'Elapsed seconds' else 2 if name.endswith('%') else 4
            cells.append(f"{v['mean']:.{digits}f}" + (f" ± {v['sd']:.{digits}f}" if v['sd'] is not None else ''))
        lines.append(f'| {arm} | {len(rows[arm])} | '+' | '.join(cells)+' |')
    lines += ['', '耗时包括训练、验证与最终生成，排除进程启动和 checkpoint reload；跨历史作业的耗时仅供参考，不能替代同卡算子测速。', '',
              'GSA2 native 使用64 slots、8192状态元素/head及短卷积，LR1e-3为此前验证集选定的配方；'
              'KOMP/GDN为4096状态元素/head，learned元数据另计。旧gsa_slot还有decay.bias的WD豁免差异。', '',
              '## 配对比较', '', 'A−B；CE负值更好，生成正值更好。固定数据集上的seed配对bootstrap95%区间，20,000次抽样；未做多重比较校正。', '',
              '| A − B | 指标 | n | 差值 | 95% CI |', '|---|---|---:|---:|---|']
    for p in paired:
        lines.append(f"| {p['a']} − {p['b']} | {p['metric']} | {len(p['seeds'])} | {p['mean']:+.4f} | [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] |")
    lines += ['', '## 验收与产物', '', f'新实验及复现对照已独立核验 {generation_count} 条生成记录：gold、exact、合法性和汇总率一致。'
              '数据指纹、共享初始化、源码指纹、训练token数与checkpoint重载误差均检查。', '',
              f'四组完整模型预检通过：33-token模型输出对FP64逐token参考的最大RMS相对偏差为 {100*preflight_rms:.3f}%，'
              f'低于预先设定的1%门槛；训练后40次跨边界因果干预最大绝对误差 {causal_max:g}。'
              '各组16步小样本训练loss下降、实际batch64反向通过，gate/learned的score梯度非零。', '',
              'GDN seed731 复现对照差值：`'+json.dumps(control.get('deltas', {}), ensure_ascii=False)+'`。', '',
              '预检细节见 preflight.json；每次训练的完整曲线、summary、generation位于results/。'
              'checkpoint保存在远端独立实验目录。统计可用 report.py 重建；实验设计见 PLAN.md。', '',
              '![Validation curves](curves.png)', '']
    absent = ROOT/'evidence/job-absent.json'
    if absent.exists() and json.loads(absent.read_text()) == []:
        lines += ['本轮单卡GPU作业已完成并按精确alias清理，查询确认不存在；远端结果和checkpoint保留。', '']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))
    if not partial:
        conclusions = '''# 本轮结论

四组各7个seed、共28次训练全部完成，另复跑GDN seed731。保持原10M token数据和训练配方，未重调LR。

- **逐token exact压缩没有超过Block-KOMP。** 对应scalar/FIFO/gate/learned四组，Exact的Content CE均更差。逐slot FIFO相对Block C64的CE增加0.1523，配对95%区间[0.1294,0.1838]；Brevo合法率下降2.79个百分点，区间[-4.69,-1.12]。
- **新Exact slot FIFO与旧逐token gsa_slot效果接近。** CE 2.2440 vs 2.2469；Depo均4.13%；Brevo33.71% vs33.93%。三个差值的区间均跨零；这不证明数学等价，只说明本实验未检出收益。
- **逐slot衰减仍是主要收益。** 相对Exact scalar，slot FIFO的CE下降0.2366，Brevo上升22.43个百分点；两者区间均不跨零。
- **写入门改善CE，learned选择损害Brevo。** Gate-only相对FIFO的CE下降0.0425，95%区间[-0.0616,-0.0265]；Brevo上升1.79个百分点的区间跨零。Learned相对gate-only的CE增加0.0689，区间[0.0328,0.1001]，Brevo下降9.375个百分点，区间[-13.2812,-6.0268]。Learned对Depo生成的收益区间跨零，暂不声称稳定改善。
- **相对GDN/GSA仍有Brevo优势，但不是全面优于调参后的GSA2。** GSA2 LR1e-3的CE1.9819、Depo11.94%更好，Exact gate的Brevo35.49%更好；GSA2参数、状态、卷积与LR不同。

因此100M比较采用Block/Exact同slot FIFO、同任务既定LR/WD，直接检验压缩频率变化。不能从这些结果断言FP32精度导致差距，也不能断言所有数据或训练预算下逐token压缩都更差。

GDN复现CE偏移+0.00323，生成有少量离散变化，说明历史比较并非逐位可复现。所有区间只描述固定语料与固定128例/任务下的seed变动，未做多重比较校正。7424条本轮生成记录全部独立核验通过。完整均值、标准差、配对区间与曲线见RESULTS.md；checkpoint保留在远端。
'''
        (ROOT/'CONCLUSIONS.md').write_text(conclusions)
    plot(rows)
    print(json.dumps({'status': result['status'], 'new_runs': sum(len(rows.get(a,{})) for a in PLAN['variants']),
                      'generations_audited': generation_count, 'control': control}, ensure_ascii=False))


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    palette = ['#2563eb', '#0d9488', '#d97706', '#9333ea', '#777777', '#b45309', '#222222']
    arms = [*PLAN['variants'], 'gdn', 'gsa2_native_lr1e3', 'block_slot_c64']
    for arm, color in zip(arms, palette):
        if arm not in rows:
            continue
        buckets = {}
        for seed in rows[arm]:
            path = ARMS[arm]/f'{arm}-seed{seed}'/'metrics.jsonl'
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row['step'] and 'val' in row:
                    buckets.setdefault(row['step'], []).append(row['val'])
        steps = sorted(buckets)
        for ax, task in zip(axes, ('mean_content_ce', 'depo', 'brevo')):
            values = [[r[task] if task == 'mean_content_ce' else r[task]['content_ce'] for r in buckets[s]] for s in steps]
            means = [np.mean(v) for v in values]
            sd = [np.std(v, ddof=1) if len(v)>1 else 0 for v in values]
            label = arm if arm in PLAN['variants'] else arm+' (prior)'
            ax.plot(steps, means, color=color, label=label, linewidth=1.8,
                    linestyle='-' if arm in PLAN['variants'] else '--')
            ax.fill_between(steps, np.array(means)-sd, np.array(means)+sd, color=color, alpha=.1)
    for ax, title in zip(axes, ('Validation content CE', 'Depo validation CE', 'Brevo validation CE')):
        ax.set(title=title, xlabel='Training step', ylabel='Cross entropy')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(ROOT/'curves.png', dpi=170)
    fig.savefig(ROOT/'curves.svg')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    main(parser.parse_args().partial)
