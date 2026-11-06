"""Render same-run benchmark evidence without mixing graph and eager timings."""
import json
from pathlib import Path

ROOT=Path(__file__).parent
r=json.loads((ROOT/'evidence/fla-comparison.json').read_text())
lines=['# H100：KOMP、GSA、GDN 算子速度','',
       f"状态：{r.get('status','running')}；{len(r['checks'])} 项 reference 数值检查；{len(r['rows'])} 组配置。",
       '', 'B=2、H=4、Dk=Dv=64；主表M=32，三者每head均4096个状态元素。KOMP主结果包含全部核几何，三者均含终态及训练时的初态/终态梯度。',
       '', '输入BF16；KOMP内部FP32，FLA原生精度。原生chunk GSA/GDN块长64，KOMP块长16。模型输出不等价；等缓存大小不意味着等容量。',
       '', '下表单位为毫秒，三次独立计时取中位数。CUDA Graph replay排除捕获和Python调度；eager单列。']
for mode,label in [('graph_ms','CUDA Graph replay'),('eager_ms','Eager')]:
    lines+=['',f'## {label}，BF16，M=32','',
            '| T | GDN 前向 | GSA 前向 | KOMP 前向 | GDN 前反向 | GSA 前反向 | KOMP 前反向 |',
            '|---:|---:|---:|---:|---:|---:|---:|']
    for row in r['rows']:
        if row['input_dtype']!='torch.bfloat16' or row['M']!=32:continue
        ops=row['operators']
        vals=[ops[k][stage][mode] for stage in ('forward','forward_backward') for k in ('gdn','gsa','komp')]
        lines.append('| '+str(row['T'])+' | '+' | '.join(f'{x:.4f}' for x in vals)+' |')
lines+=['','## 几何成本和补充配置','',
        '| T | dtype | M | 完整KOMP graph前向 | 仅recurrence graph前向 | 完整KOMP graph前反向 | 仅recurrence graph前反向 | 高层API eager前向 |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
for row in r['rows']:
    o=row['operators']
    vals=[o[k][stage]['graph_ms'] for stage in ('forward','forward_backward') for k in ('komp','komp_core')]
    lines.append('| '+f"{row['T']} | {row['input_dtype']} | {row['M']} | "+' | '.join(f'{x:.4f}' for x in vals)+f" | {row['public_komp_forward_eager_ms']:.4f} |")
lines+=['','预计算几何的recurrence数值只用于成本诊断，不能替代完整算子比较。M=16/64时GSA/KOMP的状态预算为GDN的0.5/2倍。',
        '', '完整配置、源码与环境指纹、各轮原始计时、误差：`evidence/fla-comparison.json`；协议见 `FLA_COMPARISON.md`。']
(ROOT/'FLA_RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
