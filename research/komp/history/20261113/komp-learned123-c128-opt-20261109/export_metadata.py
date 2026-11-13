from pathlib import Path
import json
root=Path('/tmp/komp-learned123-c128-opt-20261109-triton')
names=('ut_bwd','out_bwd','compose','proj_bwd_a','keygrad')
rows=[]
for p in root.rglob('*.json'):
    try:x=json.loads(p.read_text())
    except Exception:continue
    if not isinstance(x,dict):continue
    name=str(x.get('name',p.stem))
    if not any(n in name or n in p.name for n in names):continue
    rows.append(dict(path=str(p),name=name,available_fields=sorted(x),metadata={k:v for k,v in x.items() if isinstance(v,(str,int,float,bool,type(None)))}))
out=Path('evidence/triton-metadata.json');out.write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(dict(metadata_entries=len(rows),output=str(out))))
