"""Read-only remote progress snapshot. Run over SSH stdin."""
import json
from pathlib import Path

root = Path('/gpfs/users/linlizhi/physics100m-exact-20261105')
out = {'root': str(root), 'preflight': {}, 'active': [], 'finished_train': [], 'finished_eval': []}
for p in sorted((root/'runs/preflight').glob('*.json')):
    row = json.loads(p.read_text())
    out['preflight'][p.stem] = {'status': row['status'], 'model': row['model'], 'losses': [row['losses'][0], row['losses'][-1]],
                                'real_batch_shape': row.get('real_batch_shape')}
for claim in sorted((root/'queue').glob('*.running')):
    name = claim.name.rsplit('.', 2)[0]
    if (root/'queue'/f'{name}.done').exists():
        continue
    spec = json.loads(claim.read_text())
    row = {'claim': claim.name, 'spec': spec}
    log = root/'logs'/f'{name}.log'
    if log.exists():
        row['log_tail'] = log.read_text().splitlines()[-3:]
    if spec.get('run'):
        metrics = root/'runs/results'/spec['run']/'metrics.jsonl'
        if metrics.exists() and metrics.stat().st_size:
            row['latest'] = json.loads(metrics.read_text().splitlines()[-1])
    out['active'].append(row)
for p in sorted((root/'runs/results').glob('*/summary.json')):
    row = json.loads(p.read_text())
    out['finished_train'].append({'name': p.parent.name, 'status': row['status'], 'val': row['val'], 'tokens': row['tokens']})
for p in sorted((root/'runs/results').glob('*/heldout.json')):
    row = json.loads(p.read_text())
    out['finished_eval'].append({'name': p.parent.name, 'status': row['status'], 'test': row['test'], 'generation': row['generation_accuracy']})
failure = root/'queue/FAILED.json'
out['failure'] = json.loads(failure.read_text()) if failure.exists() else None
out['pending'] = len([p for p in (root/'queue').glob('*.json') if p.name != 'FAILED.json'])
out['complete'] = (root/'evidence/ALL_COMPLETE.json').exists()
print(json.dumps(out, ensure_ascii=False))
