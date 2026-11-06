"""Incrementally hash completed candidate checkpoints on GPFS."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
target = ROOT / 'evidence/checkpoints.json'
prior = json.loads(target.read_text()) if target.exists() else {}
current = {}
for summary_path in sorted(ROOT.glob('variants/*/runs-ddp/results/*/summary.json')):
    summary = json.loads(summary_path.read_text())
    assert summary['status'] == 'completed'
    checkpoint = summary_path.with_name('checkpoint.pt')
    stat = checkpoint.stat()
    identity = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest())
    name = summary_path.parent.name
    old = prior.get(name, {})
    if old.get('identity') == identity:
        current[name] = old
        continue
    digest = hashlib.sha256()
    with checkpoint.open('rb') as file:
        for chunk in iter(lambda: file.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    current[name] = dict(path=str(checkpoint), identity=identity, checkpoint_sha256=digest.hexdigest(),
                         config=summary['config'], positions=summary['tokens'])
target.write_text(json.dumps(current, indent=2) + '\n')
print(json.dumps(dict(completed_checkpoint_count=len(current))))
