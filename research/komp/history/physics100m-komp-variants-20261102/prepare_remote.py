"""Create an isolated experiment; share only frozen sources and read-only data."""
import json
from pathlib import Path
r=Path('/gpfs/users/linlizhi/physics100m-komp-variants-20261102')
assert not r.exists()
r.mkdir()
for name in ['suite100m','evidence','logs','queue','runs','runs/preflight','runs/selection','runs/results']:
 (r/name).mkdir(parents=True,exist_ok=True)
old=Path('/gpfs/users/linlizhi/physics100m-20261101')
assert (old/'snapshot/komp_slot/block_fast.py').exists()
(r/'data').symlink_to(old/'data',target_is_directory=True)
print(json.dumps({'root':str(r),'data':str((r/'data').resolve())}))
