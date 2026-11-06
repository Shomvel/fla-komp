from pathlib import Path
root=Path('/gpfs/users/linlizhi/komp-coh-trials-20261101')
for name in ('cpu-acceptance.json','workers/r1/runner.log','workers/r1/preflight-all12.log'):
 p=root/name
 print(name)
 print('\n'.join(p.read_text().splitlines()[-65:]) if p.exists() else 'missing')
