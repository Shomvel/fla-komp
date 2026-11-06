"""Publish only after GPU acceptance; refuse to overwrite concurrent kernel edits."""
import difflib,hashlib,json,shutil
from pathlib import Path
r=Path(__file__).parent
project=r.parent/'komp-slot-attention-prototype'
for name in ['c128-validation.json','legacy-validation.json','integration.json']:
    data=json.loads((r/'evidence'/name).read_text());assert data['status']=='passed',(name,data.get('error'))
source=r/'work/komp_slot/block_fast.py';target=project/'komp_slot/block_fast.py';base=r/'evidence/block_fast.before.py'
assert target.read_bytes()==base.read_bytes(),'Concurrent main-tree edits; manually merge the minimal patch first'
validated=json.loads((r/'evidence/c128-validation.json').read_text())['sources']['komp_slot/block_fast.py']
assert hashlib.sha256(source.read_bytes()).hexdigest()==validated
freeze=r/'frozen';freeze.mkdir(exist_ok=False)
for sub in ['komp_slot','komp_bench']:
    shutil.copytree(r/'work'/sub,freeze/sub,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copy2(r/'work/benchmark_fla.py',freeze/'benchmark_fla.py')
files=sorted(p for p in freeze.rglob('*') if p.is_file())
(freeze/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(freeze)}\n' for p in files))
# Recheck immediately before replacing this single file.
assert target.read_bytes()==base.read_bytes()
shutil.copy2(source,target)
(r/'evidence/publication.json').write_text(json.dumps(dict(target=str(target),sha256=validated,frozen=str(freeze),prior_sha256=hashlib.sha256(base.read_bytes()).hexdigest()),indent=2))
shutil.copy2(r/'C128_TRITON.md',project/'BLOCK_KOMP_C128.md')
print('Published',target,validated)
