from pathlib import Path
import hashlib,json,shutil
root=Path('/gpfs/users/linlizhi/komp-coh-trials-20261101');root.mkdir(exist_ok=False)
fla=Path('/gpfs/users/linlizhi/komp-gsa-gdn-20261031-r1/fla-source')
assert (fla/'fla/ops/gsa/__init__.py').exists()
(root/'snapshot/fla-source').mkdir(parents=True)
shutil.copytree(fla/'fla',root/'snapshot/fla-source/fla',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(root/'results').mkdir()
for i in range(1,17):(root/'workers'/f'r{i}').mkdir(parents=True)
print(json.dumps(dict(root=str(root),fla_files=len(list((root/'snapshot/fla-source/fla').rglob('*.py'))))))
