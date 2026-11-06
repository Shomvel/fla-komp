"""Plot measured GPU execution time without mixing eager and CUDA Graph."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parent
report = json.loads((ROOT / 'evidence/fla-comparison.json').read_text())
rows = [r for r in report['rows'] if r['M'] == 32 and r['input_dtype'] == 'torch.bfloat16']
colors = {'gdn': '#2166ac', 'gsa': '#339966', 'komp': '#d95f02'}
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for ax, stage, title in zip(axes, ('forward', 'forward_backward'), ('Forward', 'Forward + backward')):
    for kind in ('gdn', 'gsa', 'komp'):
        ax.plot([r['T'] for r in rows],
                [r['operators'][kind][stage]['graph_ms'] for r in rows],
                color=colors[kind], marker='o', label=kind.upper())
    ax.set(xscale='log', yscale='log', xlabel='Sequence length', ylabel='Milliseconds (log scale)', title=title)
    ax.set_xticks([r['T'] for r in rows], [str(r['T']) for r in rows])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=.2, which='both')
axes[0].legend()
fig.suptitle('H100 / BF16 / CUDA Graph / B=2 H=4 D=64 M=32\nKOMP includes kernel geometry; equal state element budgets', fontsize=11)
fig.savefig(ROOT / 'fla-speed.png', dpi=160)
fig.savefig(ROOT / 'fla-speed.svg')
