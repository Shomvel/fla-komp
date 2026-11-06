from pathlib import Path
import json
r=Path('/gpfs/users/linlizhi/komp-coh-trials-20261101')
f=r/'workers/r1/preflight-all12.log'
print(json.dumps({'ready':(r/'preflight-ready.json').exists(),'failure':json.loads((r/'failure.json').read_text()) if (r/'failure.json').exists() else None,'passes':[s for s in f.read_text().splitlines() if s.startswith('PASS')] if f.exists() else [],'tail':f.read_text().splitlines()[-5:] if f.exists() else []}))
