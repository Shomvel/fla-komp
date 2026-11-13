"""Preserve the abandoned attempt and select its checkpoint-backed metric prefix."""
import json,hashlib
from pathlib import Path
r=Path('/gpfs/users/linlizhi/komp-kda-options-ff2272-340m-20c-20261106');ev=r/'evidence/block-resume-20261106-r1';run=r/'runs/block'
assert json.loads((run/'latest.json').read_text())['step']==1536
for name in ('metrics.jsonl','validation.jsonl'):
 p=run/name;archive=ev/name
 assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(archive.read_bytes()).digest(),name
 rows=[s for s in p.read_text().splitlines() if json.loads(s)['step']<=1536]
 tmp=p.with_suffix('.resume.tmp');tmp.write_text('\n'.join(rows)+'\n');tmp.replace(p)
(ev/'metrics-normalized.json').write_text(json.dumps(dict(status='passed',retained_through_step=1536,old_attempt_preserved=True)))
