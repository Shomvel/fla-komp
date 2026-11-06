"""Report same-GPU geometry candidates and exact parallel route speedups."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
E=ROOT/'evidence/decay-20261031'
first=json.loads((E/'geometry-fast-benchmark.json').read_text())
parallel=json.loads((E/'geometry-parallel-benchmark.json').read_text())
routes=json.loads((E/'routes-parallel.json').read_text())
assert first['status']==parallel['status']==routes['status']=='passed'
lines=['# KOMP decay 与加速：GPU 证据','',
       'H100、Triton3.7.1、BF16输入，geometry保持FP32计算。下列完整算子包含核几何与chunk recurrence，CUDA Graph三次计时中位数，单位ms。不同作业的数据不用于直接计算加速比。',
       '', '## 默认采用：并行整数route scan + 原IEEE求解','',
       '| T | M | 原前向 | 并行route前向 | 原前反向 | 并行route前反向 | 前反向加速 |',
       '|---:|---:|---:|---:|---:|---:|---:|']
for row in parallel['rows']:
    a=row['modes']['baseline'];b=row['modes']['parallel_ieee']
    af=a['full_forward']['graph_ms'];bf=b['full_forward']['graph_ms']
    ab=a['full_forward_backward']['graph_ms'];bb=b['full_forward_backward']['graph_ms']
    lines.append(f"| {row['T']} | {row['M']} | {af:.4f} | {bf:.4f} | {ab:.4f} | {bb:.4f} | {ab/bb:.2f}× |")
lines+=['','默认在T>=192且M>=16时使用并行route，小尺寸保留串行。此阈值是所测shape下的保守选择，不是全shape最优性保证。`triton_geometry_serial`保留固定基线，`triton_geometry_parallel`可显式调用。',
        '并行算法以整数max扫描构造每个slot的最近写入位置，route/removed/final indices与串行逐元素相同；没有替换kernel/ridge或减少几何求解。',
        '', '## 仅route成本','', '| B | T | H | M | 串行ms | 并行ms | 加速 |', '|---:|---:|---:|---:|---:|---:|---:|']
for r in routes['benchmarks']:
    lines.append(f"| {r['B']} | {r['T']} | {r['H']} | {r['M']} | {r['sequential']['graph_ms']:.5f} | {r['parallel']['graph_ms']:.5f} | {r['speedup']:.1f}× |")
lines+=['','route局部加速不能当成完整KOMP加速。Gram/投影及其backward仍占主要成本。',
        '', '## 首轮算术候选：未默认采用','', '| T | M | baseline前反向 | gather IEEE | gather TF32x3 |', '|---:|---:|---:|---:|---:|']
for row in first['rows']:
    vals=[row['modes'][k]['full_forward_backward']['graph_ms'] for k in ('baseline','gather_ieee','gather_tf32x3')]
    lines.append(f"| {row['T']} | {row['M']} | "+' | '.join(f'{v:.4f}' for v in vals)+' |')
lines+=['','gather与TF32x3没有稳定的跨shape收益，所以作为显式实验入口保留，不默认开启。它们通过本轮普通误差门槛与近重复key压力测试，仍不构成任意条件数下的精度保证。',
        '', '## 逐slot recurrence成本','', '| B | T | H | variant | graph前向ms | graph前反向ms |', '|---:|---:|---:|---|---:|---:|']
slot=json.loads((E/'slot-decay-gpu.json').read_text())
for row in slot['benchmark']:
    b,t,h,m,v=row['shape']
    lines.append(f"| {b} | {t} | {h} | {row['variant']} | {row['forward_graph_ms']:.4f} | {row['forward_backward_graph_ms']:.4f} |")
lines+=['','本节仅预计算h/p/r后的recurrence，不能替代完整KOMP。slot系数在每个chunk内部按16/32步组合，chunk间并行；全部Triton。',
        '', '原始证据：`evidence/decay-20261031/`。训练质量及诊断见`physics_lm/decay_runs/RESULTS.md`。']
(ROOT/'DECAY_ACCELERATION_RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:16]))
