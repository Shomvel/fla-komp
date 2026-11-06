from pathlib import Path
import json
r=Path('/gpfs/users/linlizhi/komp-proj-swa-20261101')
for name in ('cpu-acceptance.json','preflight-ready.json','failure.json'):
 p=r/name
 if p.exists():print(name,p.read_text())
p=r/'workers/r1/preflight-all10.log'
if p.exists():print(p.read_text()[-10000:])
