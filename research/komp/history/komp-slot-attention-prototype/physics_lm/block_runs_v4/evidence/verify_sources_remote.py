"""Final source identity and completion receipt, executed through SSH stdin."""
from pathlib import Path
import hashlib,json
root=Path('/gpfs/users/linlizhi/komp-block-trials-20261101');snap=root/'snapshot'
manifest=json.loads((root/'snapshot-sha256.json').read_text())
for name,digest in manifest.items():assert hashlib.sha256((snap/name).read_bytes()).hexdigest()==digest,name
fla=Path('/gpfs/users/linlizhi/komp-gsa-gdn-20261031-r1/fla-source')
old=json.loads((root/'fla-provenance.json').read_text())['source_sha256']
now={str(p.relative_to(fla)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (fla/'fla').rglob('*.py')}
assert old==now,'FLA source changed during trials'
progress=[json.loads(s) for s in (root/'results/progress.jsonl').read_text().splitlines()]
assert len(progress)==73 and all(r['exit_code']==0 for r in progress)
assert json.loads((root/'completion.json').read_text())['runs']==70
assert not list((root/'results').rglob('*.pt'))
receipt=dict(status='passed',source_files=len(manifest),fla_python_files=len(now),
             completed_runs=70,completed_preflights=3,no_checkpoints=True)
(root/'final-source-audit.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
