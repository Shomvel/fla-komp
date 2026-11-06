from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess,sys
E=Path(__file__).resolve().parent
OUT=E/sys.argv[1];OUT.mkdir(exist_ok=True)
HELPER='/Users/shom/.codex/skills/northjob-gpu-test-bp-dev/scripts/northjob.py'
def call(args,path):
 p=subprocess.run([sys.executable,HELPER,*args],capture_output=True,text=True)
 assert p.returncode==0,p.stderr
 data=json.loads(p.stdout);path.write_text(json.dumps(data,indent=2));return data
def clean(i):
 alias=f'komp-coh-trials-20261101-r{i}'
 jobs=call(['jobs','--alias',alias],OUT/f'r{i}-terminal.json')
 assert len(jobs)==1 and jobs[0]['status'] not in ('Running','Pending'),jobs
 call(['delete','--alias',alias],OUT/f'r{i}-delete.json')
 remaining=call(['jobs','--alias',alias],OUT/f'r{i}-cleanup.json');assert remaining==[]
 print(json.dumps(dict(worker=i,status=jobs[0]['status'],deleted=True)),flush=True)
start=int(sys.argv[2]) if len(sys.argv)>2 else 1
stop=int(sys.argv[3]) if len(sys.argv)>3 else 17
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(clean,range(start,stop)))
