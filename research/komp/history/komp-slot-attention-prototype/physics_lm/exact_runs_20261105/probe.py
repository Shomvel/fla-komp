import json
from pathlib import Path

root = Path('/gpfs/users/linlizhi/komp-exact-physics-20261105')
progress_path = root/'progress.jsonl'
progress = [json.loads(line) for line in progress_path.read_text().splitlines()] if progress_path.exists() else []
starts = [line.removeprefix('START ') for line in (root/'runner.log').read_text().splitlines() if line.startswith('START ')]
active = starts[-1] if starts else None
result = {'completed': sum(row['name'].startswith('exact_') and row['exit_code'] == 0 for row in progress),
          'total': 28, 'active': active, 'last_finished': progress[-1] if progress else None,
          'done': (root/'completion.json').exists()}
if active and active.startswith('exact_'):
    path = root/'results'/active/'metrics.jsonl'
    if path.exists():
        lines = path.read_text().splitlines()
        if lines:
            row = json.loads(lines[-1])
            result['latest'] = {key: row[key] for key in ('step', 'loss', 'grad_norm', 'seconds') if key in row}
            if 'val' in row:
                result['latest']['validation_ce'] = row['val']['mean_content_ce']
control = root/'control/gdn-seed731/summary.json'
if control.exists():
    row = json.loads(control.read_text())
    result['gdn_control'] = {'ce': row['confirm']['mean_content_ce'], 'generation': row['generation']}
print(json.dumps(result))
