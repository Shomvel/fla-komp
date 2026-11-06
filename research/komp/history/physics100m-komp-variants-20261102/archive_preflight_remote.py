import json,shutil,time
from pathlib import Path
r=Path('/gpfs/users/linlizhi/physics100m-komp-variants-20261102');a=r/'evidence/preflight-state-order-fix';a.mkdir(exist_ok=False)
assert not list((r/'runs/results').glob('*/summary.json'))
assert not list((r/'queue').glob('*.running'))
for p in (r/'runs/preflight').glob('*.json'):p.rename(a/p.name)
for p in (r/'logs').glob('preflight-*.log'):p.rename(a/p.name)
shutil.copyfile(r/'suite100m/checks.py',a/'checks-before.py')
print(json.dumps(dict(at=time.time(),reason='Acceptance harness FIFO returned-state order mismatch: fast chronological vs flex physical. Rotate reference keys and coefficients in autograd; kernels unchanged.',archive=str(a))))
