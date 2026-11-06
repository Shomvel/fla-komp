"""Plot recorded validation scores and per-run generation rates."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
colors={'gdn':'#2166ac','gsa':'#339966','komp':'#d95f02'}
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for path in sorted((ROOT/'results').glob('*/metrics.jsonl')):
    kind=path.parent.name.split('-')[0];seed=int(path.parent.name.split('seed')[1])
    records=[json.loads(line) for line in path.open()]
    valid=[x for x in records if x['step']>0 and 'val' in x]
    for ax,task in zip(axes,('depo','brevo')):
        ax.plot([r['step'] for r in valid],[r['val'][task]['content_ce'] for r in valid],
                color=colors[kind],linestyle='-' if seed==731 else '--',marker='o',label=f'{kind} seed{seed}')
        ax.set(xlabel='Optimizer step',ylabel='Validation content CE',title=task.capitalize())
        ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.savefig(ROOT/'validation-curves.png',dpi=160)
fig.savefig(ROOT/'validation-curves.svg')
plt.close(fig)
