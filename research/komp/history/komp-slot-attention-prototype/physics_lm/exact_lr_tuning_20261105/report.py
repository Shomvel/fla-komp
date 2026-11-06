"""Combine two new learning rates with one authenticated reused result per task."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
plan = json.loads((ROOT/'lr-screen.json').read_text())
result = {'tasks': {}, 'nominal_new_training_positions': 40_000_000, 'new_completed': 0}
lines = ['# Exact-KOMP learning-rate screen', '',
         'Alpha0=.995, beta0=.2, temperature=.5, ridge=1e-4; FP64 projection, FP32 state, BF16 model. Independent10M schedules, seed731. One central LR per task is reused without additional training.', '',
         '| Task | LR | Source | Status | Final validation CE |', '|---|---:|---|---|---:|']
for task in ('depo2', 'brevo2'):
    receipt = json.loads((ROOT/'evidence'/f'reuse-{task}.json').read_text())
    rows = [{'lr': receipt['lr'], 'source': 'reused original alpha0-.995', 'status': 'completed', 'ce': receipt['val']['ce']}]
    for entry in plan['new_runs'][task]:
        path = ROOT/'results'/task/entry['name']/'summary.json'
        row = {'lr': entry['lr'], 'source': 'new10M run', 'status': 'pending', 'ce': None}
        if path.exists():
            summary = json.loads(path.read_text())
            assert summary['status'] == 'completed' and summary['config']['lr'] == entry['lr']
            assert summary['config']['candidate'] == plan['fixed_candidate']
            assert summary['config']['data_order_sha256'] == receipt['original_config']['data_order_sha256']
            assert summary['tokens'] == receipt['original_config']['train_positions']
            row.update(status='completed', ce=summary['val']['ce'])
            result['new_completed'] += 1
        elif (path.parent/'metrics.jsonl').exists():
            metrics = [json.loads(x) for x in (path.parent/'metrics.jsonl').read_text().splitlines()]
            row.update(status='running', latest=metrics[-1] if metrics else None)
        rows.append(row)
    rows.sort(key=lambda x: x['lr'])
    result['tasks'][task] = {'rows': rows, 'winner_lr': min(rows, key=lambda x:x['ce'])['lr'] if all(x['status']=='completed' for x in rows) else None}
    for row in rows:
        value = f"{row['ce']:.9f}" if row['ce'] is not None else '—'
        lines.append(f"| {task} | {row['lr']:g} | {row['source']} | {row['status']} | {value} |")
result['status'] = 'completed' if result['new_completed'] == 4 else 'running'
(ROOT/'results.json').write_text(json.dumps(result, indent=2)+'\n')
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'status': result['status'], 'new_completed': result['new_completed']}))
