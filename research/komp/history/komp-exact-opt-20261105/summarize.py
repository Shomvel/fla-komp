"""Render recorded GPU evidence without rerunning any kernels."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
ev=root/'evidence'
load=lambda name:json.loads((ev/name).read_text())
r=load('comparison.json')
assert r['status']=='passed' and len(r['rows'])==14
checks=[]
for name in ('preflight.json','full.json','edge.json'):
    result=load(name);assert result['status']=='passed';checks.extend(result['checks'])
selection=load('selection.json');assert selection['status']=='passed'
training=load('train-smoke.json');assert training['status']=='passed'
rows=r['rows']
pick=lambda name,dtype,c=16:next(x for x in rows if x['operator']==name and x['dtype']==dtype and (name=='gdn' or x['compute_chunk']==c))
bf='torch.bfloat16'
lines=['# Exact KOMP 优化验收与 8K 性能','',
'默认投影改为稳定 FP32，状态保持 FP32；移除每 token M×V 状态和 M×M Gram/inverse tape。系数反向拆成 chunk 边界反扫与块内并行梯度，learned 选择批量跳过拒绝区间。保留 `projection_dtype="float64"` 作为精度对照。','',
'## 相同 H100 上的 8K 结果','',
'B=1、T=8192、H=2、K=V=64，KOMP M=32。每 head 的 keys+coefficients 与 GDN 的 recurrent matrix 均为 4096 个 FP32 元素；learned metadata 另计。表中 CUDA Graph 排除编译、autotune、capture 和 Python 调度开销，峰值显存增量在 eager 路径独立测量。各自包含最终状态及全输入反向。','',
'| 输入精度 | 算子 | 块长 | 前向 ms | 前向＋反向 ms | 训练显存增量 MiB |','|---|---|---:|---:|---:|---:|']
for x in rows:
    lines.append(f"| {x['dtype'].replace('torch.','')} | {x['operator']} | {x['compute_chunk']} | {x['forward']['cuda_graph']['ms']:.4f} | {x['forward_backward']['cuda_graph']['ms']:.4f} | {x['forward_backward']['extra_peak_MiB']:.2f} |")
lines+=['','同一轮 BF16、块长 16 对比：','']
for name in ('fifo','learned'):
    old,new,gdn=pick('baseline_'+name,bf),pick(name,bf),pick('gdn',bf)
    fb=lambda x:x['forward_backward']['cuda_graph']['ms']
    mem=lambda x:x['forward_backward']['extra_peak_MiB']
    lines.append(f"- {name}：前反向加速 {fb(old)/fb(new):.2f}×，显存增量降低 {(1-mem(new)/mem(old))*100:.1f}%；当前前反向仍是 GDN 的 {fb(new)/fb(gdn):.2f}×。")
lines+=['','普通 eager 耗时（含 Python 调度）另列：','','| 输入精度 | 算子 | 块长 | 前向 ms | 前向＋反向 ms |','|---|---|---:|---:|---:|']
for x in rows:
    lines.append(f"| {x['dtype'].replace('torch.','')} | {x['operator']} | {x['compute_chunk']} | {x['forward']['eager']['ms']:.4f} | {x['forward_backward']['eager']['ms']:.4f} |")
lines+=['','## 精度与训练','',f'最终版本：{len(checks)} 项输出/状态/梯度/流式/因果性检查通过；另有 {len(selection["cases"])} 组 selector 的完整索引、slot、分数、birth 轨迹与旧版逐位一致，包含 8K。GDN 本轮参考对拍另计 16 项。','',
'FP64 冻结逐 token 参考的 full suite（最大值包含输出、最终状态和输入梯度）：','',
'| 输入精度 | 最大 RMS 相对误差 | 最大归一化绝对误差 |','|---|---:|---:|']
full=load('full.json')['checks']
for dt in ('torch.float32','torch.bfloat16'):
    ss=[x for x in full if dt in x['name']]
    lines.append(f"| {dt} | {max(x['rms_relative'] for x in ss):.8g} | {max(x['normalized_max'] for x in ss):.8g} |")
lines+=['','相对误差分母为参考 RMS，至少 1e-6；归一化绝对误差分母为 max(1,max(abs(reference)))。保留既有 FP32 0.5% / BF16 2.5% RMS 检查门槛，没有因初版失败放宽。最难用例为近重复 key 的字典梯度。原始 FP32 直接翻译失败证据保存在 fp32-naive-full-failure.json；最终稳定距离形式、直接 RHS 消元与差分梯度通过。','',
'8K 与旧版 FP64 geometry kernel 的全部输出、状态和梯度对拍：','']
for dt in ('torch.float32','torch.bfloat16'):
    ss=[x for x in load('edge.json')['checks'] if x['name'].startswith('8k/') and dt in x['name']]
    lines.append(f"- {dt} 最大 RMS 相对偏差：{max(x['rms_relative'] for x in ss):.8g}。")
lines+=['','优化器级别：FIFO / learned 各 12 步 AdamW，真实 GDN-slot 参数化与 FP64 逐 token 参考对拍；两者 loss 均下降。','']
for learned in (False,True):
    ss=[x for x in training['records'] if x['learned']==learned]
    lines.append(f"- {'learned' if learned else 'FIFO'}：loss {ss[0]['triton']:.8f} → {ss[-1]['triton']:.8f}；最大参数轨迹差 {max(x['parameter_max_abs'] for x in ss):.8g}。")
lines+=['','## 剩余瓶颈','',
'以下是 warmup 后 profiler 的 kernel attribution，每个算子执行 3 次，独立于上面的计时：','']
for x in load('kernel-profile.json'):
    if x['chunk']!=16:continue
    lines.append(f"- {x['operator']} C16："+'；'.join(f"{k['name']} {k['us_per_call']/1000:.3f} ms" for k in x['kernels'][:5])+'。')
lines+=['','仍有 chunk 边界扫描、直接 Gram 求解和局部重算成本；在 B1/H2 下不能充分利用所有 SM。当前不是与 GDN 等速的实现，也未做完整模型质量训练。learned 的批量跳过优化适用于存在拒绝区间；全替换时仍需依次更新物理 slot。','',
'## 证据与复现','',
'- DESIGN.md：GDN/Mesa 源码依据、推导及状态存储说明。',
'- compare.py：3 组 × 20 次 CUDA events，中位数；同卡测旧/新/GDN。',
'- evidence/comparison.json：全部 trial、显存、运行源码 SHA256。',
'- evidence/kernel-profile.json：各 kernel 耗时。',
'- evidence/environment.json：实际 Torch/Triton/CUDA/H100 环境。',
'- job.yaml / runner.sh：WeaverRSI/proj-32，固定镜像 digest，独立任务目录。',
'- evidence/candidate1-*.json：第一次优化后的测量，最终结果未混用。','']
if (ev/'integration.json').exists():
    assert load('integration.json')['status']=='passed'
    lines.append('现有项目包 `komp_slot.exact_komp` 已完成 GPU 导入、输出/状态和反向验收；本地包与实测源码哈希一致。')
if (ev/'job-absent.json').exists():
    assert load('job-absent.json')==[]
    lines.append('测试作业已记录 Completed 终态，按精确 alias 删除并确认消失；回执保存在 evidence/。')
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('Wrote RESULTS.md')
