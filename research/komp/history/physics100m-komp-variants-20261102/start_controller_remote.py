import json,subprocess,time
from pathlib import Path
r=Path('/gpfs/users/linlizhi/physics100m-komp-variants-20261102');receipt=r/'evidence/controller-start.json'
assert not receipt.exists()
with (r/'logs/controller.log').open('a') as log:
 p=subprocess.Popen(['python3',str(r/'controller.py')],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(pid=p.pid,command=['python3',str(r/'controller.py')],at=time.time());receipt.write_text(json.dumps(result));print(json.dumps(result))
