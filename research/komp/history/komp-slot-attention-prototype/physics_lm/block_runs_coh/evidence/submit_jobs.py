"""Submit only the exact reviewed 16 jobs; API credentials remain inside the helper on SSH host."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
E=Path(__file__).resolve().parent
HELPER='/Users/shom/.codex/skills/northjob-gpu-test-bp-dev/scripts/northjob.py'
def submit(i):
    alias=f'komp-coh-trials-20261101-r{i}'
    check=subprocess.run([sys.executable,HELPER,'jobs','--alias',alias],capture_output=True,text=True)
    assert check.returncode==0,check.stderr
    assert json.loads(check.stdout)==[],f'Alias already exists: {alias}'
    process=subprocess.run([sys.executable,HELPER,'submit','--yaml',str(E/f'job-r{i}.yaml')],capture_output=True,text=True)
    (E/f'job-r{i}-submit.json').write_text(process.stdout)
    if process.returncode: raise RuntimeError(f'{alias}: submission failed: {process.stderr}')
    result=json.loads(process.stdout)
    print(json.dumps(dict(worker=i,result=result)),flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(submit,range(int(sys.argv[1]) if len(sys.argv)>1 else 1,17)))
