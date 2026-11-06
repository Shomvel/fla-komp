"""Block-KOMP trial table generated from summary.json files, beside the existing matched arms."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
ARMS = [('GDN', ROOT/'results', 'gdn'), ('GSA', ROOT/'results', 'gsa'), ('KOMP per-token (cyclic)', ROOT/'results', 'komp'),
        ('KOMP sigmoid (decay study)', ROOT/'decay_runs/results', 'sigmoid'),
        ('KOMP gsa_slot (decay study)', ROOT/'decay_runs/results', 'gsa_slot'),
        ('Block-KOMP C=16', ROOT/'block_runs/results', 'block_c16'),
        ('Block-KOMP C=32', ROOT/'block_runs/results', 'block_c32'),
        ('Block-KOMP C=64', ROOT/'block_runs/results', 'block_c64'),
        ('Block-KOMP C=64 per-slot decay', ROOT/'block_runs_v2/results', 'block_slot_c64'),
        ('  + projectron writes', ROOT/'block_runs_v2/results', 'block_slot_proj_c64'),
        ('  + evict by residual', ROOT/'block_runs_v2/results', 'block_slot_res_c64'),
        ('  + evict by coef norm', ROOT/'block_runs_v2/results', 'block_slot_coef_c64'),
        ('  + evict by learned score', ROOT/'block_runs_v2/results', 'block_slot_learn_c64'),
        ('  + evict by learned score (Triton, fused BF16)', ROOT/'block_runs_v3/results', 'block_slot_learn_c64'),
        ('Block-KOMP C=64 per-slot decay (fused BF16)', ROOT/'block_runs_v3/results', 'block_slot_c64')]


def rows():
    for label, root, kind in ARMS:
        runs = [json.loads(p.read_text()) for p in sorted(root.glob(f'{kind}-seed*/summary.json'))]
        runs = [r for r in runs if r.get('status') == 'completed']
        if not runs:
            continue
        mean = lambda f: statistics.mean(f(r) for r in runs)
        yield (label, len(runs), mean(lambda r: r['confirm']['mean_content_ce']), mean(lambda r: r['confirm']['depo']['content_ce']),
               mean(lambda r: r['confirm']['brevo']['content_ce']), mean(lambda r: r['generation']['depo']['valid']),
               mean(lambda r: r['generation']['brevo']['valid']), mean(lambda r: r['elapsed_seconds']))


def main():
    lines = ['| arm | seeds | confirm content CE | Depo CE | Brevo CE | Depo gen | Brevo valid gen | train+eval s |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, n, ce, dce, bce, dg, bg, sec in rows():
        lines.append(f'| {label} | {n} | {ce:.4f} | {dce:.4f} | {bce:.4f} | {dg:.2%} | {bg:.2%} | {sec:.0f} |')
    text = '\n'.join(lines)+'\n'
    (ROOT/'block_runs_v2').mkdir(exist_ok=True)
    (ROOT/'block_runs_v2/TABLE.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
