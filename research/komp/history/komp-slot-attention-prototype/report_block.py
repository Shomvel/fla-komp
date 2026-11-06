"""Render Block-KOMP evidence tables (benchmark, validation) from evidence/accel-20261101/*.json."""
import json
from pathlib import Path

E = Path(__file__).resolve().parent/'evidence/accel-20261101'


def bench_table():
    rows = json.loads((E/'block-benchmark-final.json').read_text())['rows']
    robust = json.loads((E/'block-benchmark-robust.json').read_text())
    names = {'gdn': 'GDN (FLA)', 'gsa': 'GSA (FLA)', 'komp_per_token': 'KOMP per-token (all_triton)',
             'block_c64': 'Block-KOMP C=64, tf32 projection', 'block_c32': 'Block-KOMP C=32, tf32 projection',
             'block_v0_c64': 'Block-KOMP v0 single kernel C=64'}
    out = ['| config | operator | fwd ms | fwd+bwd ms | extra training GiB |', '|---|---|---:|---:|---:|']
    for r in rows:
        cfg = f"B{r['b']} T{r['t']} H{r['h']} Dk{r['dk']} Dv{r['dv']} M{r['m']}"
        out.append(f"| {cfg} | {names[r['operator']]} | {r['fwd_ms']:.3f} | {r['fwd_bwd_ms']:.3f} | {r['train_peak_extra_gib']:.2f} |")
    for r in robust:
        if r['operator'].startswith('block'):
            cfg = 'B1 T8192 H64 Dk128 Dv128 M64' if r.get('config') == '8k' else 'B2 T2048 H4 Dk64 Dv64 M32'
            out.append(f"| {cfg} | **Block-KOMP C={r['chunk']} (default, tf32x3 projection)** | {r['fwd_ms']:.3f} | {r['fwd_bwd_ms']:.3f} | {r['train_peak_extra_gib']:.2f} |")
    return '\n'.join(out)


def validation_table():
    r = json.loads((E/'block-fast-validation.json').read_text())
    worst = {}
    for c in r['checks']:
        if 'rms_relative' not in c:
            continue
        dtype = 'bf16' if 'bfloat16' in c['name'] else 'fp32'
        worst[dtype] = max(worst.get(dtype, 0), c['rms_relative'])
    out = [f"status {r['status']}, {len(r['checks'])} checks", '| inputs | worst RMS relative error vs FP64 token reference | tolerance |', '|---|---:|---:|',
           f"| FP32 | {worst['fp32']:.2e} | 5e-3 |", f"| BF16 | {worst['bf16']:.2e} | 1e-2 |"]
    for c in r['checks']:
        if c['name'].startswith('near'):
            out.append(f"| near-duplicate {c['name'].split('/',1)[1]} (FP32) | {max(c['candidate_rms']):.2e} (max ratio to v0 {c['max_ratio_to_v0']:.2f}) | max(5e-3, 1.25 v0) |")
    return '\n'.join(out)


def bench_v2_table():
    rows = json.loads((E.parent/'accel-20261101b/block-benchmark-v2.json').read_text())['rows']
    names = {'gdn': 'GDN (FLA)', 'gsa': 'GSA (FLA)', 'komp_per_token': 'KOMP per-token (all_triton)',
             'block_c64': 'Block-KOMP C=64 scalar decay', 'block_slot_c64': 'Block-KOMP C=64 per-slot decay',
             'block_c32': 'Block-KOMP C=32 scalar decay'}
    out = ['| config | operator | fwd ms | fwd+bwd ms | extra training GiB |', '|---|---|---:|---:|---:|']
    for r in rows:
        cfg = f"B{r['b']} T{r['t']} H{r['h']} Dk{r['dk']} Dv{r['dv']} M{r['m']}"
        out.append(f"| {cfg} | {names[r['operator']]} | {r['fwd_ms']:.3f} | {r['fwd_bwd_ms']:.3f} | {r['train_peak_extra_gib']:.2f} |")
    return '\n'.join(out)


def validation_v2():
    r = json.loads((E.parent/'accel-20261101b/block-slot-validation.json').read_text())
    worst = {}
    for c in r['checks']:
        if 'rms_relative' in c:
            k = ('per-slot' if '/slot/' in c['name'] else 'scalar', 'BF16' if 'bfloat16' in c['name'] else 'FP32')
            worst[k] = max(worst.get(k, 0), c['rms_relative'])
    out = [f"status {r['status']}, {len(r['checks'])} checks", '| decay | inputs | worst RMS relative error | tolerance |', '|---|---|---:|---:|']
    for (dec, dt), e in sorted(worst.items()):
        out.append(f"| {dec} | {dt} | {e:.2e} | {'1e-2' if dt == 'BF16' else '5e-3'} |")
    return '\n'.join(out)


def bench_v3_table(name='accel-20261101c/block-benchmark-v3.json'):
    rows = json.loads((E.parent/name).read_text())['rows']
    names = {'gdn': 'GDN (FLA)', 'gsa': 'GSA (FLA)', 'komp_per_token': 'KOMP per-token (all_triton)',
             'block_c64': 'Block-KOMP C=64 scalar decay', 'block_slot_c64': 'Block-KOMP C=64 per-slot decay',
             'block_slot_learn_c64': 'Block-KOMP C=64 per-slot decay + learned eviction',
             'block_c32': 'Block-KOMP C=32 scalar decay'}
    out = ['| config | operator | fwd ms | fwd+bwd ms | fwd kernels | fwd+bwd kernels | extra training GiB |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        cfg = f"B{r['b']} T{r['t']} H{r['h']} Dk{r['dk']} Dv{r['dv']} M{r['m']}"
        if 'error' in r:
            out.append(f"| {cfg} | {names[r['operator']]} | error: {r['error'][:60]} | | | | |")
            continue
        out.append(f"| {cfg} | {names[r['operator']]} | {r['fwd_ms']:.3f} | {r['fwd_bwd_ms']:.3f} | {r['fwd_kernels']} | "
                   f"{r['fwd_bwd_kernels']} | {r['train_peak_extra_gib']:.2f} |")
    return '\n'.join(out)


def validation_v3(name='accel-20261101c/block-v3-validation.json'):
    r = json.loads((E.parent/name).read_text())
    worst = {}
    for c in r['checks']:
        if 'rms_relative' not in c or c['name'].startswith('learned_as_fifo'):
            continue
        evict = 'learned' if c['name'].startswith('learned/') else 'fifo'
        k = (evict, 'per-slot' if '/slot/' in c['name'] else 'scalar', 'BF16' if 'bfloat16' in c['name'] else 'FP32')
        worst[k] = max(worst.get(k, 0), c['rms_relative'])
    ref = {'fifo': 'FP64 token reference', 'learned': 'FP64 flex (PyTorch chunk loop)'}
    out = [f"status {r['status']}, {len(r['checks'])} checks",
           '| eviction | decay | inputs | reference | worst RMS relative error | tolerance |', '|---|---|---|---|---:|---:|']
    for (ev, dec, dt), e in sorted(worst.items()):
        out.append(f"| {ev} | {dec} | {dt} | {ref[ev]} | {e:.2e} | {'1e-2' if dt == 'BF16' else '5e-3'} |")
    lf = [c['rms_relative'] for c in r['checks'] if c['name'].startswith('learned_as_fifo')]
    if lf:
        out.append(f"| learned, scores = insertion time | per-slot | FP32 | fifo path | {max(lf):.2e} | 1e-4 |")
    for c in r['checks']:
        if c['name'].startswith('near'):
            out.append(f"| near-duplicate {c['name'].split('/',1)[1]} | scalar | FP32 | FP64 token reference | "
                       f"{max(c['candidate_rms']):.2e} (ratio to v0 {c['max_ratio_to_v0']:.2f}) | max(5e-3, 1.25 v0) |")
    return '\n'.join(out)


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['v5']:
        print(bench_v3_table('accel-20261101e/block-benchmark-r5c.json'))
        print()
        print(validation_v3('accel-20261101e/block-r5c-validation.json'))
        raise SystemExit
    if sys.argv[1:] == ['v4']:
        print(bench_v3_table('accel-20261101d/block-benchmark-r4.json'))
        print()
        print(validation_v3('accel-20261101d/block-r4-validation.json'))
        raise SystemExit
    if sys.argv[1:] == ['v3']:
        print(bench_v3_table())
        print()
        print(validation_v3())
        raise SystemExit
    if sys.argv[1:] == ['v2']:
        print(bench_v2_table())
        print()
        print(validation_v2())
        raise SystemExit
    print(bench_table())
    print()
    print(validation_table())
