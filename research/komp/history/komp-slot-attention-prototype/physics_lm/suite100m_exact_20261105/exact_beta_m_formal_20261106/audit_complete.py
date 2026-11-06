"""Final acceptance of all six beta/M runs, heldout identities and cleanup."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


subprocess.run([sys.executable, str(ROOT / 'report.py')], check=True)
report = read(ROOT / 'results.json')
plan = read(ROOT / 'plan.json')
assert report['new_training_complete'] == report['new_heldout_complete'] == 6
assert read(ROOT / 'evidence/jobs-final.json') == []
checkpoints = read(ROOT / 'evidence/checkpoints.json')
assert len(checkpoints) == 6
manifest = read(ROOT / 'source-manifest.json')
evaluated = {}
for row in report['runs']:
    if row['reused']:
        continue
    task, beta, slots = row['task'], row['beta0'], row['slots']
    variant = ROOT / 'variants' / f'{task}-beta{beta:g}-m{slots}'
    name = f'{task}-exact-beta{beta:g}-m{slots}-final-lr{plan["lr"][task]:g}-wd0.1-seed731'
    directory = variant / 'runs-ddp/results' / name
    summary = read(directory / 'summary.json')
    numerical = read(variant / 'evidence/numerical.json')
    ddp = read(variant / 'runs-ddp/ddp-preflight/exact.json')
    assert numerical['status'] == ddp['status'] == 'passed'
    assert ddp['sources'] == summary['config']['sources']
    assert ddp['model']['beta0'] == numerical['beta0'] == beta
    assert ddp['model']['block_slots'] == numerical['slots'] == slots
    progress = read(directory / 'eval-progress.json')
    records = read(directory / 'generations.json')
    heldout = read(directory / 'heldout.json')
    identity = progress['identity']
    assert progress['records'] == records and len(records) == 256
    assert progress['test'] == heldout['test']
    assert identity['run'] == name
    assert identity['summary_sha256'] == sha(directory / 'summary.json')
    assert identity['evaluator_sha256'] == hashlib.sha256((ROOT / 'evaluate.py').read_bytes() + (ROOT / 'snapshot/eval_progress.py').read_bytes()).hexdigest()
    assert identity['probes_sha256'] == hashlib.sha256(json.dumps([r['probe'] for r in records], sort_keys=True).encode()).hexdigest()
    checkpoint = checkpoints[name]
    assert checkpoint['identity']['summary_sha256'] == identity['summary_sha256']
    assert checkpoint['identity']['size'] == identity['checkpoint_size']
    assert checkpoint['identity']['mtime_ns'] == identity['checkpoint_mtime_ns']
    evaluated[name] = dict(checkpoint_sha256=checkpoint['checkpoint_sha256'],
                           summary_sha256=identity['summary_sha256'], heldout_sha256=sha(directory / 'heldout.json'),
                           generations_sha256=sha(directory / 'generations.json'),
                           numerical_sha256=sha(variant / 'evidence/numerical.json'),
                           ddp_sha256=sha(variant / 'runs-ddp/ddp-preflight/exact.json'))
positions = sum(row['positions'] for row in report['runs'] if not row['reused'])
assert positions == plan['additional_actual_positions'] == 600004761
result = dict(status='completed_and_audited', new_training_complete=6, new_heldout_complete=6,
              reused_baselines=2, additional_positions=positions, generation_probes_independently_regraded=1536,
              source_manifest_sha256=sha(ROOT / 'source-manifest.json'), results_sha256=sha(ROOT / 'results.json'),
              checkpoint_manifest_sha256=sha(ROOT / 'evidence/checkpoints.json'),
              artifacts=evaluated, validation_only_winners=report['validation_only_winners'],
              allocations='Training and heldout jobs terminal, deleted by exact alias, and absent in final prefix query.',
              limitations=[plan['precision_limit'], report['scope']])
(ROOT / 'evidence/complete.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}))
