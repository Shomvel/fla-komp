"""Render measured BSGD chunk results and verify the frozen implementation."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
E=ROOT/'evidence/bsgd-chunk-20261101-r1'
v=json.loads((E/'work/evidence/bsgd-chunk-validation.json').read_text())
b=json.loads((E/'work/evidence/bsgd-chunk-benchmark.json').read_text())
assert v['status']==b['status']=='passed'
assert all(c['passed'] for c in v['checks'])
assert len(b['rows'])==24 and all(r['status']=='passed' for r in b['rows'])
sha=json.loads((E/'source-sha256.json').read_text())
assert all(hashlib.sha256((E/'work'/p).read_bytes()).hexdigest()==digest for p,digest in sha.items())
assert hashlib.sha256((ROOT/'komp_slot/bsgd_chunk.py').read_bytes()).hexdigest()==sha['komp_slot/bsgd_chunk.py']
assert (E/'exit-code.txt').read_text().strip()=='0'
lines=['# BSGD chunk 实测结果','',f'{len(v["checks"])} 项验收通过，24 组算子测速完成。H100 80GB，Torch 2.10.0a0，Triton 3.7.1。',
       '', '## 数值验收','']
for dtype in ('torch.float32','torch.bfloat16'):
    rows=[x for x in v['checks'] if dtype in x['name'] and 'rms_relative' in x]
    worst=max(rows,key=lambda x:x['rms_relative'])
    lines.append(f'- {dtype}：输出、最终状态与全部输入梯度对齐独立 FP64 参考，最坏 RMS 相对误差 {worst["rms_relative"]:.6g}；门槛 {worst["tolerance"]}。')
for prefix,label in [('causal/','未来输入干预'),('future_grad/','前缀损失对未来输入的梯度')]:
    rows=[x for x in v['checks'] if x['name'].startswith(prefix)]
    lines.append(f'- {label}：{len(rows)} 项，最大绝对值 {max(x["max_abs"] for x in rows):.6g}。')
lines += ['- 另覆盖 C=16/32/64、scalar/slot decay、归零衰减、beta/merge=0/1、尾块、不规则维度、D/V=128、独立输出/状态损失、任意 token 断点续接及重复中心。',
          '- 前向的中心/系数规则与 gated 递归版一致，分块只改变计算顺序；grid-search 不适用。浮点归约与原子加法不保证逐 bit 等价或确定性。',
          '', '## 速度','',
          'BF16 q/k/v、FP32 门和状态；CUDA Graph 中位数，排除编译时间。含输出与最终状态梯度；额外显存为单次训练算子峰值，不是模型显存。', '',
          '| B/T/H/D/V/M | decay | 实现 | forward ms | forward+backward ms | 额外 GiB |',
          '|---|---|---|---:|---:|---:|']
for row in b['rows']:
    config='/'.join(map(str,row['config']))
    lines.append(f'| {config} | {"slot" if row["slot_decay"] else "scalar"} | {row["mode"]} | {row["fwd_ms"]:.3f} | {row["fwd_bwd_ms"]:.3f} | {row["extra_train_gib"]:.3f} |')
lines += ['', '## 同规则加速','']
for cfg in sorted(set(tuple(r['config']) for r in b['rows'])):
    rows=[r for r in b['rows'] if tuple(r['config'])==cfg and not r['slot_decay']]
    serial=next(r for r in rows if r['mode']=='serial')
    block=next(r for r in rows if r['mode']=='block64')
    best=min((r for r in rows if r['mode'].startswith('chunk')),key=lambda r:r['fwd_bwd_ms'])
    lines.append(f'- {cfg}：最快 {best["mode"]}，相对同规则递归版加速 {serial["fwd_bwd_ms"]/best["fwd_bwd_ms"]:.2f}×；耗时为不同算法 block KOMP 的 {best["fwd_bwd_ms"]/block["fwd_bwd_ms"]:.2f}×。')
lines += ['', '中心轨迹按槽并行但槽内仍串行；geometry backward 使用原子累加。Backward 仍分配 O(BTHMV) 局部重算工作区。本轮没有 Depo/Brevo 训练收益结论。',
          '', '源码、环境、验证和运行日志见当前目录。原有 KOMP kernel 未改动；公共 scan/局部 backward 复用冻结依赖。']
(E/'RESULTS.md').write_text('\n'.join(lines)+'\n')
receipt=dict(status='passed',validation_checks=len(v['checks']),benchmark_rows=len(b['rows']),frozen_source_verified=True,
             evidence_sha256={str(p.relative_to(E)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (E/'work/evidence').glob('*.json')})
(E/'verification.json').write_text(json.dumps(receipt,indent=2))
print('\n'.join(lines))
