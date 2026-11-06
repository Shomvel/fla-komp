from pathlib import Path
import shutil,json
r=Path('/gpfs/users/linlizhi/komp-coh-trials-20261101')
s=r/'snapshot-fine';shutil.copytree(r/'snapshot',s,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(r/'results-fine').mkdir();(r/'fine-workers').mkdir()
print(json.dumps(dict(status='fine_snapshot_copied')))
