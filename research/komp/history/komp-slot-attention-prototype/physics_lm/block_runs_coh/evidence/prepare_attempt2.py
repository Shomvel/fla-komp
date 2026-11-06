from pathlib import Path
import json
root=Path('/gpfs/users/linlizhi/komp-coh-trials-20261101')
assert not list((root/'results').rglob('summary.json'))
assert all((root/'workers'/f'r{i}'/'exit-code.txt').exists() for i in range(1,17))
archive=root/'attempt1';archive.mkdir(exist_ok=False)
for name in ('workers','results','failure.json','snapshot-sha256.json','cpu-acceptance.json'):
 p=root/name
 if p.exists():p.rename(archive/name)
(root/'workers').mkdir();(root/'results').mkdir()
for i in range(1,17):(root/'workers'/f'r{i}').mkdir()
print(json.dumps(dict(status='archived_failed_attempt',formal_runs=0)))
