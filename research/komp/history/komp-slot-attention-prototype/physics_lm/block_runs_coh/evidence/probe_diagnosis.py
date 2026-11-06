from pathlib import Path
import json
p=Path('/gpfs/users/linlizhi/komp-coh-trials-20261101/diagnosis')
print(json.dumps({'exit':(p/'exit-code.txt').read_text() if (p/'exit-code.txt').exists() else None}))
f=p/'run.log'
if f.exists():print('\n'.join(f.read_text().splitlines()[-8:]))
