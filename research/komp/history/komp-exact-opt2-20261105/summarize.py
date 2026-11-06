"""Render final recorded evidence, including unsuccessful optimization candidates."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent;ev=root/'evidence'
load=lambda n:json.loads((ev/n).read_text())
r=load('comparison.json');assert r['status']=='passed' and len(r['rows'])==14
rows=r['rows'];bf='torch.bfloat16'
pick=lambda name,dt=bf,c=16:next(x for x in rows if x['operator']==name and x['dtype']==dt and (name=='gdn' or x['compute_chunk']==c))
fb=lambda x:x['forward_backward']['cuda_graph']['ms']
fw=lambda x:x['forward']['cuda_graph']['ms']
mem=lambda x:x['forward_backward']['extra_peak_MiB']
lines=['# 第二轮 Exact KOMP：最终验收与 8K 实测','',
'本轮保留 FP32 geometry/状态与 tf32x3 计算，保持每 token 压缩和原有 learned 选择语义。基准是上一轮已优化实现（约 4 ms 前反向），不是最早的 FP64 版本。','',
'## 同卡性能','',
'H100 80GB，B1/T8192/H2/K=V64，KOMP M32。KOMP keys+coefficients 与 GDN recurrent matrix 各 4096 个 FP32 持久状态元素/head，learned metadata 另计。下表 CUDA Graph 排除编译、capture 和 Python 调度；3 组×20 次调用取中位数。显存是 eager 测得的峰值 allocated 增量，不含常驻输入、上游梯度或 graph pool。','',
'| 输入精度 | 算子 | C | 前向 ms | 前向＋反向 ms | 训练显存增量 MiB |','|---|---|---:|---:|---:|---:|']
for x in rows:
    lines.append(f"| {x['dtype'].replace('torch.','')} | {x['operator']} | {x['compute_chunk']} | {fw(x):.4f} | {fb(x):.4f} | {mem(x):.2f} |")
lines+=['','BF16 C16 与上一轮同卡基准比较：','']
for name in ('fifo','learned'):
    old,new,gdn=pick('baseline_'+name),pick(name),pick('gdn')
    lines.append(f'- {name}：前向加速 {fw(old)/fw(new):.2f}×，前反向加速 {fb(old)/fb(new):.2f}×；当前前反向为 GDN 的 {fb(new)/fb(gdn):.2f}×。显存增量 {mem(old):.2f} → {mem(new):.2f} MiB。')
lines+=['','普通 eager（含 Python 调度）：','','| 输入精度 | 算子 | C | 前向 ms | 前向＋反向 ms |','|---|---|---:|---:|---:|']
for x in rows:
    lines.append(f"| {x['dtype'].replace('torch.','')} | {x['operator']} | {x['compute_chunk']} | {x['forward']['eager']['ms']:.4f} | {x['forward_backward']['eager']['ms']:.4f} |")
lines+=['','## 实现与减少的工作量','',
'- learned：固定 32-token 窗口内按字典版本复用 LDLᵀ 分解，批量解 RHS；反向同样重算每版本 factor，并合并 Gram 梯度。频繁替换窗口、M>32 或 FP64 诊断模式用独立 token 路径。',
'- dense/FIFO：直接 RHS sweep 前后向；反向解 Az=dp，不再形成完整 inverse 后乘 dp。单 RHS LDLᵀ 实测慢，未无条件采用。M32/K64 的该路径选 2 warp，其余保留 4 warp。',
'- 读取保存的 H/R/N；局部状态与 adjoint 用稳定的仿射前后缀扫描，替代稠密三角矩阵重建。',
'- NC>=32 使用分层仿射边界扫描，Z 由独立 chunk kernel 并行计算；短序列保留原串行扫描。增加少量工作区以换取时间并行度。',
'- 移除 chunk maps 的最后一轮零幂矩阵乘。','']
counts=load('version-counts.json')
lines.append(f"本轮 learned 8K 输入共 {counts['accepted_tokens']} 次接受；{counts['sparse_windows']} 个稀疏窗口、{counts['dense_windows']} 个密集窗口。每个前向或反向的分解次数由 {counts['previous_factorizations_per_pass']} 降为 {counts['factorizations_per_pass']}，减少 {100*counts['fraction_saved']:.2f}%。同一版本跨窗口仍会重算；并非无限制全局缓存。")
lines+=['','## 消融与未采用候选','',
'下表是最终 dense RHS 布局优化之前的独立候选，不能与上表混为同一版本。current 使用逐 token LDLᵀ 与多 RHS 版本复用；每一项按标记独立开关。','',
'| 候选 | FIFO 前反向 ms | learned 前反向 ms |','|---|---:|---:|']
ab=load('ablations.json')['rows']
for name in dict.fromkeys(x['variant'] for x in ab):
    cells=[]
    for learned in (False,True):
        x=next(y for y in ab if y['variant']==name and y['learned']==learned)
        cells.append(f"{fb(x):.4f}" if x['accepted'] else '精度不通过')
    lines.append(f'| {name} | {cells[0]} | {cells[1]} |')
lines+=['',
'- solve_only：逐 token LDLᵀ、保存 H/R/N、去除零幂乘法；保留旧边界和局部重建。它比旧版慢，说明减少 FLOPs 不一定缩短 GPU 执行时间。',
'- no_versions / no_hierarchy / dense_local：分别关闭版本复用、分层扫描、局部仿射扫描。',
'- reuse_z：保留串行扫描，采用 S′=ES+LZ；本次比不复用 Z 的串行路径慢，未采用。',
'- mixed_state：只把状态矩阵乘操作数转 BF16，FP32 累加和 geometry 不变。只做了本组 8K 随机输入对照，未作为默认或发布接口。','']
for x in ab:
    if x['variant']=='mixed_state':
        lines.append(f"mixed_state {'learned' if x['learned'] else 'FIFO'} 的最大 RMS 偏差 {max(e['rms_relative'] for e in x['errors']):.6g}；实验门槛 1%，通过不等于完整精度/训练验收。")
checks=[]
for name in ('preflight','full','edge','scan','versions'):
    result=load(name+'.json');assert result['status']=='passed';checks.extend(result['checks'])
train=load('train-smoke.json');assert train['status']=='passed'
lines+=['','## 正确性','',f'最终版本 {len(checks)} 项检查通过（preflight/full/edge/scan/versions）。另外 GDN 16 项参考对拍通过。涵盖输出、最终状态、输入梯度、近重复 keys、M64/K128、零 beta、零/小范数、强遗忘、FP64 诊断模式、分组尾部、流式及 8K 前缀因果性。','',
'完整 suite 对冻结 FP64 逐 token 参考的最大误差（含梯度）：','','| 输入 | 最大 RMS 相对误差 | 最大归一化绝对误差 |','|---|---:|---:|']
for dt in ('torch.float32','torch.bfloat16'):
    ss=[x for x in load('full.json')['checks'] if dt in x['name']]
    lines.append(f"| {dt} | {max(x['rms_relative'] for x in ss):.8g} | {max(x['normalized_max'] for x in ss):.8g} |")
lines+=['','RMS 分母至少 1e-6；归一化绝对误差分母为 max(1,max(abs(reference)))。沿用原 suite 门槛，未为本轮实现放宽。','']
for dt in ('torch.float32','torch.bfloat16'):
    ss=[x for x in load('edge.json')['checks'] if x['name'].startswith('8k/') and dt in x['name']]
    lines.append(f"8K 对上一轮优化 kernel，{dt} 最大 RMS 偏差 {max(x['rms_relative'] for x in ss):.8g}。")
for learned in (False,True):
    ss=[x for x in train['records'] if x['learned']==learned]
    lines.append(f"12 步 AdamW {'learned' if learned else 'FIFO'}：loss {ss[0]['triton']:.8f} → {ss[-1]['triton']:.8f}；对 FP64 参考最大参数轨迹差 {max(x['parameter_max_abs'] for x in ss):.8g}。")
lines+=['','## 剩余成本与边界','']
for x in load('kernel-profile.json'):
    if x['chunk']==16:
        lines.append(f"- {x['operator']} C16 profiler："+'；'.join(f"{k['name']} {k['us_per_call']/1000:.3f} ms" for k in x['kernels'][:6])+'。')
lines+=['','Profiler 含仪器开销，独立于性能表计时。版本复用收益依赖真实接受率；FIFO 或始终接受的 learned 无法共享不同字典的分解。此轮没有声称与 GDN 相同表达能力或完整模型质量，也未进行 PhysicsLM 质量训练。','',
'RESEARCH.md 记录设计、来源和取舍；compare.py、ablations.py 与 evidence/ 保存原始计时、精度、环境、源码 SHA256 和作业回执。编译索引类型问题与测速脚本的捕获冲突在 queue 日志中保留，未计作成功数值结果。']
if (ev/'integration.json').exists():
    assert load('integration.json')['status']=='passed';lines+=['','现有项目包已完成 GPU 导入与反向验收，源码哈希与实测版本一致。']
if (ev/'job-absent.json').exists():
    assert load('job-absent.json')==[];lines+=['','GPU 测试作业已完成，精确 alias 删除后复查为空。']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('Wrote RESULTS.md')
