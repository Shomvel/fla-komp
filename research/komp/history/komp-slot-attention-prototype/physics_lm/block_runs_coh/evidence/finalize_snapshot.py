from pathlib import Path
import json,hashlib
root=Path('/gpfs/users/linlizhi/komp-coh-trials-20261101');snap=root/'snapshot'
files={str(p.relative_to(snap)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(snap.rglob('*')) if p.is_file() and '__pycache__' not in str(p) and p.suffix!='.pyc'}
assert files['komp_slot/block_flex.py']=='04e049363c3adcc4e675f2076b9977e585c3ce655e8566ffaad14519964c3144'
assert files['komp_slot/block_fast.py']=='a8e4c72527473bb1928841df26c5e4625e9441f5b7f2b309f420cabf8d0c74ce'
(root/'snapshot-sha256.json').write_text(json.dumps(files,indent=2));print(json.dumps(dict(status='frozen',files=len(files))))
