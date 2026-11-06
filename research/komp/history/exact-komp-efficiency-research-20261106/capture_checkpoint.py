"""Standard-library-only immutable copy, safe on the CPU jump host."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
assert (a.checkpoint/'complete.json').exists();a.out.mkdir(parents=True,exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
files=['model.pt',*[f'rank{i}.pt' for i in range(8)],'complete.json'];receipt={}
for name in files:
 src=a.checkpoint/name;before=sha(src);shutil.copy2(src,a.out/name);assert before==sha(a.out/name)==sha(src),name
 receipt[name]=before
(a.out/'capture.json').write_text(json.dumps(dict(source=str(a.checkpoint),files=receipt),indent=2));print(json.dumps(dict(source=str(a.checkpoint),files=len(files),out=str(a.out))))
