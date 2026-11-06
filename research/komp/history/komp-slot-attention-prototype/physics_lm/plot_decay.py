"""Render measured validation curves for the fixed decay ablation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent/'decay_runs'
colors={'sigmoid':'#444444','gdn_scalar':'#2166ac','gsa_scalar':'#339966','gsa_slot':'#d95f02'}
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for path in sorted((ROOT/'results').glob('*/metrics.jsonl')):
    kind=path.parent.name.split('-seed')[0];seed=int(path.parent.name.split('-seed')[1])
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    rows=[r for r in rows if r['step']>0 and 'val' in r]
    for ax,task in zip(axes,('depo','brevo')):
        ax.plot([r['step'] for r in rows],[r['val'][task]['content_ce'] for r in rows],
                color=colors[kind],linestyle='-' if seed==731 else '--',label=f'{kind} {seed}')
        ax.set(title=task.capitalize(),xlabel='Optimizer step',ylabel='Validation content CE')
        ax.grid(alpha=.2)
axes[0].legend(fontsize=7,ncol=2)
fig.savefig(ROOT/'validation-curves.png',dpi=160)
fig.savefig(ROOT/'validation-curves.svg')
