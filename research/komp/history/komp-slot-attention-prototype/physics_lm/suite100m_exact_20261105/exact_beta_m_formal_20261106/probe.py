"""Read-only remote progress; runnable via ssh python3 stdin."""
import json
from pathlib import Path

ROOT = Path('/gpfs/users/linlizhi/physics100m-exact-20261105/exact_beta_m_formal_20261106')
rows = []
for variant in sorted((ROOT / 'variants').glob('*')):
    row = dict(variant=variant.name, numerical='pending', ddp_preflight='pending', training='pending', heldout='pending')
    numerical = variant / 'evidence/numerical.json'
    ticket = variant / 'runs-ddp/ddp-preflight/exact.json'
    if numerical.exists():
        row['numerical'] = json.loads(numerical.read_text())['status']
    if ticket.exists():
        row['ddp_preflight'] = json.loads(ticket.read_text())['status']
    for summary in variant.glob('runs-ddp/results/*/summary.json'):
        data = json.loads(summary.read_text())
        row.update(training=data['status'], validation_ce=data['val']['ce'], positions=data['tokens'])
    if row['training'] == 'pending':
        for metrics in variant.glob('runs-ddp/results/*/metrics.jsonl'):
            lines = metrics.read_text().splitlines()
            if lines:
                row.update(training='running', latest=json.loads(lines[-1]))
    for progress in variant.glob('runs-ddp/results/*/eval-progress.json'):
        data = json.loads(progress.read_text())
        row.update(heldout='running', generation_probes=len(data['records']), test_ce=data['test']['ce'])
    for heldout in variant.glob('runs-ddp/results/*/heldout.json'):
        data = json.loads(heldout.read_text())
        row.update(heldout='completed', test_ce=data['test']['ce'], generation_accuracy=data['generation_accuracy'])
    rows.append(row)
failures = [dict(path=str(p.relative_to(ROOT)), data=json.loads(p.read_text()))
            for p in [ROOT / 'FAILED.json', *ROOT.glob('variants/*/evidence/*failure.json')]
            if p.exists()]
print(json.dumps(dict(experiment='Exact beta/M formal', candidates=rows, failures=failures)))
