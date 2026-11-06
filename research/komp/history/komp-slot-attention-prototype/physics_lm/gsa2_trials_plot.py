"""Export the completed comparison from the script-generated statistics only."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / 'block_runs_v4' / 'gsa2'
ROWS = [
    ('gdn', 'GDN', '#777777'),
    ('gsa', 'GSA', '#777777'),
    ('gsa_slot', 'Token slot-KOMP (gsa_slot)', '#777777'),
    ('block_c64', 'Block scalar FIFO', '#2166ac'),
    ('block_slot_c64', 'Block slot FIFO', '#2166ac'),
    ('block_slot_gate_c64', 'Block slot + gate', '#2166ac'),
    ('block_slot_learn_c64', 'Block slot learned C64', '#2166ac'),
    ('block_slot_learn_c32', 'Block slot learned C32', '#2166ac'),
    ('kapa_scalar_p2_c16', 'KAPA scalar', '#7b3294'),
    ('kapa_slot_p2_c16', 'KAPA slot', '#7b3294'),
    ('gsa2_native_lr5e4', 'GSA2 conv, LR 5e-4', '#c45d15'),
    ('gsa2_native_lr1e3', 'GSA2 conv, LR 1e-3', '#c45d15'),
    ('gsa2_noconv_lr5e4', 'GSA2 no conv, LR 5e-4', '#c45d15'),
]


def main():
    stats = json.loads((ROOT / 'statistics.json').read_text())
    assert stats['complete']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(13, 7.6), sharey=True)
    metrics = [('content CE', 'Content CE (lower is better)'),
               ('Depo gen %', 'Depo exact generation (%)'),
               ('Brevo valid %', 'Brevo valid generation (%)')]
    for ax, (metric, title) in zip(axes, metrics):
        for y, (name, _, color) in enumerate(ROWS):
            result = stats['arms'][name][metric]
            assert result['n'] == 7
            ax.errorbar(result['mean'], y, xerr=result['sample_sd'], fmt='o',
                        color=color, capsize=3, markersize=5, linewidth=1.4)
        for edge in (2.5, 7.5, 9.5):
            ax.axhline(edge, color='#e5e5e5', linewidth=.8)
        ax.set_title(title, loc='left', fontsize=11, pad=12)
        ax.grid(axis='x', color='#eeeeee', linewidth=.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(range(len(ROWS)), [label for _, label, _ in ROWS])
    axes[0].invert_yaxis()
    axes[1].set_xlim(left=0)
    axes[2].set_xlim(left=0)
    fig.suptitle('Block-KOMP / KAPA / native GSA2: fixed-data, 7-seed comparison',
                 x=.03, y=.98, ha='left', fontsize=15, weight='bold')
    fig.text(.03, .025,
             'Points = means; bars = sample SD (not confidence intervals). Seeds 731–737; 10M training tokens/run.\n'
             'GSA2: M64, larger state, native initialization / WD rules; others: M32, no conv. LR=5e-4 unless labelled 1e-3.',
             fontsize=9, color='#444444', va='bottom')
    fig.subplots_adjust(left=.235, right=.98, top=.88, bottom=.14, wspace=.19)
    for suffix in ('png', 'svg'):
        fig.savefig(ROOT / f'COMPARISON.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
