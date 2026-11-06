"""Option B (eviction period C=128 vs 64) table from summary.json files: per-arm mean and sample std over
seeds, and the paired (same seed) C=128 - C=64 differences."""
import json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
ARMS = [('per-slot FIFO, C=64', 'flex_slot_c64'), ('per-slot FIFO, C=128', 'flex_slot_c128'),
        ('per-slot learned, C=64', 'flex_slot_learn_c64'), ('per-slot learned, C=128', 'flex_slot_learn_c128')]
METRICS = [('confirm content CE', lambda r: r['confirm']['mean_content_ce']), ('Depo CE', lambda r: r['confirm']['depo']['content_ce']),
           ('Brevo CE', lambda r: r['confirm']['brevo']['content_ce']), ('Depo gen', lambda r: r['generation']['depo']['valid']),
           ('Brevo valid gen', lambda r: r['generation']['brevo']['valid']), ('train+eval s', lambda r: r['elapsed_seconds'])]


def load(kind):
    runs = {}
    for p in sorted(root.glob(f'{kind}-seed*/summary.json')):
        r = json.loads(p.read_text())
        if r.get('status') == 'completed':
            runs[int(p.parent.name.split('seed')[1])] = r
    return runs


fmt = lambda name, v: f'{v:.2%}' if 'gen' in name else (f'{v:.0f}' if name.endswith(' s') else f'{v:.4f}')
lines = ['| arm | seeds | '+' | '.join(m for m, _ in METRICS)+' |', '|---|---:|'+'---:|'*len(METRICS)]
data = {kind: load(kind) for _, kind in ARMS}
for label, kind in ARMS:
    runs = list(data[kind].values())
    if not runs:
        continue
    cells = []
    for name, f in METRICS:
        vals = [f(r) for r in runs]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.
        cells.append(f'{fmt(name, statistics.mean(vals))} ± {fmt(name, sd)}')
    lines.append(f'| {label} | {len(runs)} | '+' | '.join(cells)+' |')
lines += ['', '| paired difference (C=128 - C=64) | seeds | '+' | '.join(m for m, _ in METRICS[:5])+' |', '|---|---:|'+'---:|'*5]
for label, a, b in (('per-slot FIFO', 'flex_slot_c128', 'flex_slot_c64'), ('per-slot learned', 'flex_slot_learn_c128', 'flex_slot_learn_c64')):
    seeds = sorted(set(data[a]) & set(data[b]))
    if not seeds:
        continue
    cells = []
    for name, f in METRICS[:5]:
        d = [f(data[a][s])-f(data[b][s]) for s in seeds]
        sd = statistics.stdev(d) if len(d) > 1 else 0.
        cells.append(f'{fmt(name, statistics.mean(d))} ± {fmt(name, sd)} ({", ".join(fmt(name, x) for x in d)})')
    lines.append(f'| {label} | {len(seeds)} | '+' | '.join(cells)+' |')
print('\n'.join(lines))
