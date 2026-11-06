import hashlib,json,time
from pathlib import Path
r=Path('/gpfs/users/linlizhi/physics100m-komp-variants-20261102');old=Path('/gpfs/users/linlizhi/physics100m-20261101/snapshot')
result={'new_suite':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((r/'suite100m').glob('*.py'))},'kernel':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((old/'komp_slot').glob('*.py'))},'base_models':hashlib.sha256((old/'suite100m/models.py').read_bytes()).hexdigest()}
now=time.time();recent=[]
for p in (r/'triton-cache').glob('*/*'):
 try:mtime=p.stat().st_mtime
 except FileNotFoundError:continue
 if now-mtime<120:recent.append((mtime,p.name))
result['recent_cache_files']=len(recent);result['latest_cache']=[dict(name=n,age_seconds=now-t) for t,n in sorted(recent,reverse=True)[:6]]
print(json.dumps(result))
