"""Collect only JSON, logs and evaluator source; never copy checkpoints."""
import hashlib,json,subprocess,tarfile,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
script=r'''
import hashlib,io,json,sys,tarfile
from pathlib import Path
r=Path('/gpfs/users/linlizhi/physics100m-komp-variants-20261102')
paths=set()
for folder in ['runs/results','runs/selection','runs/preflight','logs','queue','evidence','recovery-20261102T0257Z','recovery-20261102T0719Z']:
 for p in (r/folder).rglob('*'):
  if p.is_file() and p.suffix in ('.json','.jsonl','.log','.running','.done'):paths.add(p)
for name in ['evaluate.py','eval_progress.py','runner.py','controller.py']:
 paths.add(r/name)
manifest=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for p in sorted(paths):
  rel=p.relative_to(r);assert '.pt' not in p.suffixes
  data=p.read_bytes();info=tarfile.TarInfo(str(rel));info.size=len(data);tar.addfile(info,io.BytesIO(data))
  manifest.append(dict(path=str(rel),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
 data=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('ARCHIVE_MANIFEST.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
'''
archive=ROOT/'evidence/final-json-logs.tar.gz'
with archive.open('wb') as f:
 result=subprocess.run(['ssh','linlizhi-bp-dev','python3','-'],input=script.encode(),stdout=f,stderr=subprocess.PIPE,timeout=240)
assert result.returncode==0,result.stderr.decode()
out=ROOT/'final_archive';out.mkdir(exist_ok=True)
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
  assert '.pt' not in Path(member.name).suffixes
  target=out/member.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tar.extractfile(member).read())
manifest=json.loads((out/'ARCHIVE_MANIFEST.json').read_text())
for row in manifest:assert hashlib.sha256((out/row['path']).read_bytes()).hexdigest()==row['sha256']
receipt=dict(files=len(manifest),summaries=len(list((out/'runs/results').glob('*/summary.json'))),heldout=len(list((out/'runs/results').glob('*/heldout.json'))),generations=len(list((out/'runs/results').glob('*/generations.json'))),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),no_checkpoints=True)
assert receipt['summaries']==165 and receipt['heldout']==receipt['generations']==30
(ROOT/'evidence/final-archive-verification.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
