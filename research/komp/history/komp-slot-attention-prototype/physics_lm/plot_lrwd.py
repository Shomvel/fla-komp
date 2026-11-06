"""Plot validation-only search and scale-up curves from saved experiment receipts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'comparison_runs'
DEST.mkdir(exist_ok=True)
KINDS = ('komp_slot', 'gsa', 'gdn', 'gsa2')
LABELS = {'komp_slot': 'GSA-slot KOMP', 'gsa': 'GSA', 'gdn': 'GDN', 'gsa2': 'GSA2 (native FLA)'}
COLORS = {'komp_slot': '#d95f02', 'gsa': '#238b45', 'gdn': '#2166ac', 'gsa2': '#8856a7'}
LRS, WDS = (2.5e-4, 5e-4, 1e-3), (0., .01, .1)


def study_root(kind):
    return ROOT / ('gsa2_runs' if kind == 'gsa2' else 'lrwd_runs')


fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
for ax, kind in zip(axes.flat, KINDS):
    values = np.full((3, 3), np.nan)
    for row, wd in enumerate(WDS):
        for col, lr in enumerate(LRS):
            path = study_root(kind) / 'results' / f'{kind}-small-lr{lr:g}-wd{wd:g}-seed731' / 'summary.json'
            if path.exists():
                result = json.loads(path.read_text())
                values[row, col] = result['val']['mean_content_ce']
    img = ax.imshow(values, cmap='viridis_r', vmin=1.6, vmax=3.1)
    for row in range(3):
        for col in range(3):
            value = values[row, col]
            if np.isfinite(value):
                color = 'white' if value > 2.5 else '#111111'
                ax.text(col, row, f'{value:.3f}', ha='center', va='center', color=color, fontsize=12)
    ax.set(title=LABELS[kind], xlabel='Peak learning rate', ylabel='Weight decay',
           xticks=range(3), xticklabels=['2.5e-4', '5e-4', '1e-3'],
           yticks=range(3), yticklabels=['0', '0.01', '0.1'])
fig.colorbar(img, ax=axes, label='Final validation content CE (lower is better)', shrink=.8)
fig.suptitle('10M-token LR/WD search | seed 731 | selection uses validation only')
fig.savefig(DEST / 'lrwd-search.png', dpi=180)
fig.savefig(DEST / 'lrwd-search.svg')
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
for kind in KINDS:
    root = study_root(kind)
    for path in sorted((root / 'results').glob(f'{kind}-large-*/metrics.jsonl')):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records = [r for r in records if r['step'] > 0 and 'val' in r]
        if not records:
            continue
        x = [r['unique_tokens'] / 1e6 for r in records]
        for ax, task in zip(axes, ('depo', 'brevo', 'mean')):
            y = [r['val']['mean_content_ce'] if task == 'mean' else r['val'][task]['content_ce'] for r in records]
            ax.plot(x, y, label=LABELS[kind], color=COLORS[kind], marker='o', markersize=3)
for ax, task in zip(axes, ('Depo', 'Brevo', 'Mean')):
    ax.set(title=task, xlabel='Unique training input tokens (millions)', ylabel='Validation content CE')
    ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle('Scale-up | same seed 731 | native GSA2 has a larger recurrent-state budget')
fig.savefig(DEST / 'scale-validation-curves.png', dpi=180)
fig.savefig(DEST / 'scale-validation-curves.svg')
plt.close(fig)
print(DEST)
