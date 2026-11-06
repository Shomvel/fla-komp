"""Bind the two existing best-LR checkpoints without training or copying weights."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
old = ROOT.parent / 'best_lr_formal_20261105/manifest.json'
manifest = json.loads(old.read_text())
entries = {}
for task, row in manifest['tasks'].items():
    directory = Path(row['selected_directory'])
    summary = json.loads((directory / 'summary.json').read_text())
    stat = (directory / 'checkpoint.pt').stat()
    assert summary['status'] == 'completed'
    assert {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns} == row['checkpoint_stat']
    assert hashlib.sha256((directory / 'summary.json').read_bytes()).hexdigest() == row['selected_summary_sha256']
    entries[task] = dict(beta0=.2, slots=32, lr=row['lr'], alpha0=.995,
                        checkpoint_sha256=row['checkpoint_sha256'], checkpoint_stat=row['checkpoint_stat'],
                        directory=str(directory), sources=row['sources'],
                        artifacts={name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                   for name in ('summary.json', 'heldout.json', 'generations.json', 'metrics.jsonl')})
result = dict(status='bound_completed_baselines', new_training_positions=0,
              best_lr_manifest_sha256=hashlib.sha256(old.read_bytes()).hexdigest(), tasks=entries)
(ROOT / 'evidence/baseline-reuse.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
