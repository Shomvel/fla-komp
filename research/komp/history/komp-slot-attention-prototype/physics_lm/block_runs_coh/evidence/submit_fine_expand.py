from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,subprocess,sys
p=Path(__file__).resolve().parent;helper='/Users/shom/.codex/skills/northjob-gpu-test-bp-dev/scripts/northjob.py'
screen=json.loads((p/'fine-screening.json').read_text());assert screen['expand_tags']
def submit(i):
 alias=f'komp-coh-trials-20261101-r{i+19}'
 q=subprocess.run([sys.executable,helper,'jobs','--alias',alias],capture_output=True,text=True)
 assert q.returncode==0 and json.loads(q.stdout)==[]
 out=subprocess.run([sys.executable,helper,'submit','--yaml',str(p/f'job-expand-{i}.yaml')],capture_output=True,text=True)
 assert out.returncode==0,out.stderr
 (p/f'expand-{i}-submit.json').write_text(out.stdout);print(json.dumps(dict(worker=i,result=json.loads(out.stdout))),flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(submit,range(12)))
