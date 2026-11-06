"""Final source identity and completion receipt, executed through SSH stdin."""
from pathlib import Path
import hashlib,json
root=Path('/gpfs/users/linlizhi/komp-gsa2-trials-20261101');snap=root/'snapshot'
manifest=json.loads((root/'snapshot-sha256.json').read_text())
for name,digest in manifest.items():assert hashlib.sha256((snap/name).read_bytes()).hexdigest()==digest,name
progress=[json.loads(s) for s in (root/'results/progress.jsonl').read_text().splitlines()]
assert len(progress)==22 and all(r['exit_code']==0 for r in progress)
assert json.loads((root/'completion.json').read_text())['runs']==21
assert not list((root/'results').rglob('*.pt'))
receipt=dict(status='passed',source_files=len(manifest),fla_python_files=sum(n.startswith('fla-source/fla/') and n.endswith('.py') for n in manifest),
             completed_runs=21,completed_preflights=1,no_checkpoints=True)
(root/'final-source-audit.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
