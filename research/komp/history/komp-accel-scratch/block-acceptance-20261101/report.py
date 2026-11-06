"""Render only measured results from this acceptance run."""
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
ev = root/'work/evidence'
original = json.loads((ev/'original-validation.json').read_text())
extra = json.loads((ev/'extended.json').read_text())
bench = json.loads((ev/'benchmark.json').read_text())
assert original['status'] == extra['status'] == bench['status'] == 'passed'
lines = ['# Block-KOMP 独立验收（2026-11-01）', '',
         '测试对象是当前 `komp_slot/block_fast.py` 的冻结副本；未修改 Opus 的 kernel。使用独立 H100 作业，固定镜像与 Triton 3.7.1，源码 SHA 见 source-sha256.json。', '',
         '## 精度与因果性', '',
         f'- 原有验收：{len(original["checks"])} 项通过。参考为 FP64 逐 token 写入、块末 FIFO 联合投影实现。',
         f'- 新增验收：{len(extra["checks"])} 项通过。']
for dtype in ('float32', 'bfloat16'):
    checks = [r for r in original['checks'] if 'torch.'+dtype in r['name']]
    worst = max(checks, key=lambda r: r['rms_relative'])
    lines.append(f"- {dtype} 普通输入最坏 RMS 相对误差：{worst['rms_relative']:.6g}，门槛 {worst['tolerance']}; `{worst['name']}`。")
for r in original['checks']:
    if 'candidate_rms' in r:
        lines.append(f"- {r['name']}：最坏 RMS {max(r['candidate_rms']):.6g}；该组沿用原脚本容差 `{r['tolerance']}`。M64/C64 有一项超过普通输入的 0.5% 门槛。" if 'M64' in r['name'] else
                     f"- {r['name']}：最坏 RMS {max(r['candidate_rms']):.6g}；该组沿用原脚本容差 `{r['tolerance']}`。")
interventions = [r for r in extra['checks'] if r['test'] == 'suffix_intervention']
grads = [r for r in extra['checks'] if r['test'] == 'future_gradient']
edges = [r for r in extra['checks'] if r['test'] == 'edge_reference']
for dtype in ('torch.float32', 'torch.bfloat16'):
    error = max(v['rms_relative'] for r in edges if r['dtype'] == dtype for v in r['errors'].values())
    lines.append(f'- 新增极端门值/独立损失的 {dtype} 最坏 RMS 相对误差：{error:.6g}。')
lines += [f'- {len(interventions)} 次未来输入干预：分别修改 q/k/v/g/beta，覆盖块内、块边界与多级 scan；前缀输出最大绝对变化 {max(r["max_abs"] for r in interventions):.6g}。',
          f'- {len(grads)} 次前缀损失梯度检查：未来输入梯度最大绝对值 {max(max(r["max_abs"]) for r in grads):.6g}，门槛 1e-7。',
          f'- {len(edges)} 组边界精度检查：beta=0/1、无衰减、g=-100、输出单独损失及最终状态单独损失，均对齐 FP64。',
          '- FP32/BF16 × C=16/32/64 的 6 组块边界流式调用通过；容差 1%。未承诺任意 token 断点的流式状态等价。', '',
          '## 同卡速度', '',
          'BF16 q/k/v、FP32 门和初始状态；CUDA Graph 排除首次编译，包含输出及最终状态的前反向。每项独立测量三次，表中为中位数，括号为三次范围。训练额外显存是单次前反向分配峰值，不含模型骨干。', '',
          '| B/T/H/Dk/Dv/M | 算子 | forward ms | forward+backward ms（范围） | 额外 GiB |',
          '|---|---|---:|---:|---:|']
medians = {}
for r in bench['rows']:
    assert r['status'] == 'passed'
    cfg = '/'.join(str(r['config'][k]) for k in ('b', 't', 'h', 'dk', 'dv', 'm'))
    samples = r['samples']
    fw = statistics.median(s['fwd_ms'] for s in samples)
    fb = [s['fwd_bwd_ms'] for s in samples]
    mem = statistics.median(s['train_peak_extra_gib'] for s in samples)
    medians.setdefault(cfg, {})[r['operator']] = statistics.median(fb)
    lines.append(f'| {cfg} | {r["operator"]} | {fw:.3f} | {statistics.median(fb):.3f} ({min(fb):.3f}–{max(fb):.3f}) | {mem:.3f} |')
lines += ['', '## 解释', '']
for cfg, values in medians.items():
    lines.append(f'- {cfg}：C=64 的前反向耗时为 GSA 的 {values["block_c64"]/values["gsa"]:.2f} 倍、GDN 的 {values["block_c64"]/values["gdn"]:.2f} 倍。')
    if 'per_token_komp' in values:
        lines.append(f'- 相对同配置逐 token KOMP，C=64 加速 {values["per_token_komp"]/values["block_c64"]:.2f} 倍。')
lines += ['- Block-KOMP 改变了压缩时机，块内最多保留 M+C 个原子；相同跨块状态大小不等于相同临时容量，也不等于旧算法的数值等价加速。',
          '- 当前是每头 scalar log decay；不是之前 GSA-slot KOMP 的 fine-grained decay 配方。',
          '- 本轮验证的是算子正确性与性能，不将旧快照的 Depo/Brevo 训练成绩作为当前快照的训练验收。',
          '- 未发现本轮被测条件下的因果信息泄露；这不代表对所有输入的形式化证明。']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
