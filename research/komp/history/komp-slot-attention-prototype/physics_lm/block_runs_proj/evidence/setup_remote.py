from pathlib import Path
import hashlib,json,shutil
R=Path('/gpfs/users/linlizhi/komp-proj-swa-20261101')
R.mkdir(exist_ok=False)
for name in ('snapshot','results','workers'):(R/name).mkdir()
for i in range(1,17):(R/'workers'/f'r{i}').mkdir()
fla=R/'snapshot/fla-source/fla'
shutil.copytree('/gpfs/users/linlizhi/komp-gsa-gdn-20261031-r1/fla-source/fla',fla,ignore=shutil.ignore_patterns('__pycache__'))
manifest={str(p.relative_to(R/'snapshot')):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(fla.rglob('*.py'))}
(R/'fla-provenance.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(root=str(R),fla_files=len(manifest))))
