"""Finalize the formal LR sweep from verified reports and evaluation identities."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BEST = ROOT.parent / 'best_lr_formal_20261105'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


for package in (ROOT, BEST):
    result = subprocess.run([sys.executable, str(package / 'report.py')], check=True,
                            capture_output=True, text=True)
    (package / 'evidence/report-latest.json').write_text(result.stdout)

report = read(ROOT / 'results.json')
assert report['new_training_complete'] == report['new_heldout_complete'] == 4
assert report['new_completed_positions'] == 400003174
assert read(ROOT / 'evidence/jobs-final.json') == []
evaluator = hashlib.sha256((ROOT / 'evaluate.py').read_bytes() +
                           (ROOT / 'snapshot/eval_progress.py').read_bytes()).hexdigest()
identities = {}
for row in report['runs']:
    if row['reused']:
        continue
    name = f"{row['task']}-exact-final-lr{row['lr']:g}-wd0.1-seed731"
    directory = ROOT / 'runs-ddp/results' / name
    progress = read(directory / 'eval-progress.json')
    records = read(directory / 'generations.json')
    heldout = read(directory / 'heldout.json')
    identity = progress['identity']
    assert records == progress['records'] and len(records) == 256
    assert progress['test'] == heldout['test']
    assert identity['run'] == name
    assert identity['summary_sha256'] == sha(directory / 'summary.json')
    assert identity['evaluator_sha256'] == evaluator
    probes_digest = hashlib.sha256(json.dumps([r['probe'] for r in records],
                                              sort_keys=True).encode()).hexdigest()
    assert identity['probes_sha256'] == probes_digest
    identities[name] = dict(identity=identity,
                            heldout_sha256=sha(directory / 'heldout.json'),
                            generations_sha256=sha(directory / 'generations.json'))

manifest = read(BEST / 'manifest.json')
best = read(BEST / 'results.json')
assert best['status'] == 'complete' and best['new_training_positions'] == 0
for task, entry in manifest['tasks'].items():
    assert entry['lr'] == report['validation_only_winners'][task]
    identity = identities[Path(entry['selected_directory']).name]['identity']
    assert identity['checkpoint_size'] == entry['checkpoint_stat']['size']
    assert identity['checkpoint_mtime_ns'] == entry['checkpoint_stat']['mtime_ns']

receipt = dict(status='completed_and_audited', training_complete=4,
               heldout_complete=4, reused_centers=2, new_positions=400003174,
               source_manifest_sha256=sha(ROOT / 'source-manifest.json'),
               results_sha256=sha(ROOT / 'results.json'),
               generation_audit='All 1024 new probes independently regraded by report.py; both reused centers also regraded.',
               evaluation_identities=identities,
               validation_only_winners=report['validation_only_winners'],
               allocations='Both 8-GPU training jobs and all four 1-GPU heldout jobs deleted and verified absent.',
               jobs_final_sha256=sha(ROOT / 'evidence/jobs-final.json'))
save(ROOT / 'evidence/complete.json', receipt)
save(BEST / 'evidence/complete.json', dict(
    status='completed_and_audited', new_training_positions=0,
    selected_positions=sum(x['positions'] for x in manifest['tasks'].values()),
    manifest_sha256=sha(BEST / 'manifest.json'), results_sha256=sha(BEST / 'results.json'),
    lr_completion_sha256=sha(ROOT / 'evidence/complete.json'), rows=best['rows']))
print(json.dumps({'status': receipt['status'], 'winners': receipt['validation_only_winners'],
                  'selected_results': best['rows']}))
