"""Read-only remote progress probe; send over SSH stdin, never python -c."""
from pathlib import Path
import json
root=Path('/gpfs/users/linlizhi/komp-block-trials-20261101')
progress=root/'results/progress.jsonl'
rows=[json.loads(s) for s in progress.read_text().splitlines()] if progress.exists() else []
trains=[r for r in rows if not r['name'].startswith('preflight')]
logs=list((root/'results/logs').glob('*.log'))
latest=max(logs,key=lambda p:p.stat().st_mtime) if logs else None
last=[]
if latest:
 for line in latest.read_text().splitlines()[-3:]:
  if 'COMPLETE ' in line:line=line[:120]
  elif '{' in line:
   try:
    d=json.loads(line[line.index('{'):]);line=json.dumps({k:d[k] for k in ('step','loss','grad_norm','unique_tokens','seconds') if k in d})
   except ValueError:line=line[:300]
  last.append(line)
print(json.dumps(dict(completed_runs=len(trains),completed_preflights=len(rows)-len(trains),
 last_completed=trains[-1]['name'] if trains else None,
 last_duration=trains[-1]['elapsed_seconds'] if trains else None,
 active_log=latest.name if latest else None,progress=last,
 exit_code=(root/'exit-code.txt').read_text().strip() if (root/'exit-code.txt').exists() else None)))
