import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
files=list(root.glob('*.py'))+list((root/'snapshot').rglob('*.py'))+list((root/'configs').glob('*.json'))+list(root.glob('job-*.yaml'))+list(root.glob('*.sh'))
manifest={'files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}
p=root/'source-manifest.json'
if p.exists():assert json.loads(p.read_text())==manifest,'Source changed after freeze'
else:p.write_text(json.dumps(manifest,indent=2)+'\n')
print(hashlib.sha256(p.read_bytes()).hexdigest())
