"""Coherence-trial table from summary.json files: per-arm mean ± sample std, paired (same seed) differences
against the no-coherence arm, and the recorded per-layer merge fractions (late in training)."""
import json, statistics, sys
from pathlib import Path

runs_dir = Path(sys.argv[1])
root = runs_dir/'results'
ARMS = [('per-slot FIFO (no coherence)', 'flex_slot_c64'), ('coherence cos > 0.99 merges', 'flex_slot_coh99_c64'),
        ('coherence cos > 0.95 merges', 'flex_slot_coh95_c64'), ('coherence cos > 0.80 merges', 'flex_slot_coh80_c64')]
METRICS = [('confirm content CE', lambda r: r['confirm']['mean_content_ce']), ('Depo CE', lambda r: r['confirm']['depo']['content_ce']),
           ('Brevo CE', lambda r: r['confirm']['brevo']['content_ce']), ('Depo gen', lambda r: r['generation']['depo']['valid']),
           ('Brevo valid gen', lambda r: r['generation']['brevo']['valid']), ('train+eval s', lambda r: r['elapsed_seconds'])]
fmt = lambda name, v: f'{v:.2%}' if 'gen' in name else (f'{v:.0f}' if name.endswith(' s') else f'{v:.4f}')


def load(kind):
    out = {}
    for p in sorted(root.glob(f'{kind}-seed*/summary.json')):
        r = json.loads(p.read_text())
        if r.get('status') == 'completed':
            out[int(p.parent.name.split('seed')[1])] = r
    return out


data = {kind: load(kind) for _, kind in ARMS}
lines = ['| arm | seeds | '+' | '.join(m for m, _ in METRICS)+' |', '|---|---:|'+'---:|'*len(METRICS)]
for label, kind in ARMS:
    runs = list(data[kind].values())
    if runs:
        cells = []
        for name, f in METRICS:
            vals = [f(r) for r in runs]
            cells.append(f'{fmt(name, statistics.mean(vals))} ± {fmt(name, statistics.stdev(vals) if len(vals) > 1 else 0.)}')
        lines.append(f'| {label} | {len(runs)} | '+' | '.join(cells)+' |')
lines += ['', '| paired difference vs no coherence | seeds | '+' | '.join(m for m, _ in METRICS[:5])+' |', '|---|---:|'+'---:|'*5]
base = data['flex_slot_c64']
for label, kind in ARMS[1:]:
    seeds = sorted(set(data[kind]) & set(base))
    if not seeds:
        continue
    cells = []
    for name, f in METRICS[:5]:
        d = [f(data[kind][s])-f(base[s]) for s in seeds]
        cells.append(f'{fmt(name, statistics.mean(d))} ({", ".join(fmt(name, x) for x in d)})')
    lines.append(f'| {label} | {len(seeds)} | '+' | '.join(cells)+' |')
lines += ['', '| arm | seed | merged fraction, last 10% of training steps, layers 0-3 | first 10% |', '|---|---:|---|---|']
for label, kind in ARMS[1:]:
    for p in sorted(runs_dir.glob(f'{kind}-seed*-coherence.json')):
        st = json.loads(p.read_text())['merged']
        seed = p.name.split('seed')[1].split('-')[0]
        lines.append(f'| {label} | {seed} | '+', '.join(f'{st[l]["last10pct"]:.1%}' for l in sorted(st))+' | '
                     + ', '.join(f'{st[l]["first10pct"]:.1%}' for l in sorted(st))+' |')
print('\n'.join(lines))
