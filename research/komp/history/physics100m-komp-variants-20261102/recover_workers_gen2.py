"""Replace terminal initial variant workers only, preserving exact identity receipts."""
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
HELPER=Path('/Users/shom/.codex/skills/northjob-gpu-test-bp-dev/scripts/northjob.py')
GEN='recovery-20261102T0719Z'; DEST=ROOT/'evidence'/GEN; DEST.mkdir(exist_ok=True)
expected={j['user_def_name']:j['name'] for j in json.loads((ROOT/'evidence/jobs-20261102T0719Z.json').read_text())}
def call(label,*args):
 path=DEST/(label+'.json'); assert not path.exists(), 'receipt exists: '+label
 p=subprocess.run([sys.executable,str(HELPER),*args],capture_output=True,text=True,timeout=150)
 path.write_text(p.stdout)
 assert p.returncode==0,'northjob failed: '+label
 return json.loads(p.stdout)
for worker in map(int,sys.argv[1:]):
 assert worker in range(1,13)
 alias=f'physics100m-komp-variants-20261102-r{worker}'
 rows=call(f'r{worker}-terminal','jobs','--alias',alias)
 assert len(rows)==1 and rows[0]['name']==expected[alias]
 assert rows[0]['status'] in ('Completed','Aborted') and rows[0]['ready_status']=='0/1', rows
 call(f'r{worker}-delete','delete','--alias',alias)
 assert call(f'r{worker}-absent','jobs','--alias',alias)==[]
 script=(ROOT/'requeue_worker_gen2_remote.py').read_text()
 p=subprocess.run(['ssh','linlizhi-bp-dev','python3','-',str(worker),GEN],input=script,capture_output=True,text=True,timeout=150)
 (DEST/f'r{worker}-requeue.json').write_text(p.stdout)
 assert p.returncode==0, 'reconciliation failed for '+alias+': '+p.stderr
 receipt=json.loads(p.stdout);assert receipt['worker']==worker
 assert call(f'r{worker}-before-submit','jobs','--alias',alias)==[]
 result=call(f'r{worker}-submit','submit','--yaml',str(DEST/f'job-r{worker}.yaml'))
 assert result.get('success') is not False
 print(json.dumps(dict(alias=alias,requeued=receipt['requeued'],reconciled=receipt['reconciled'],submitted=True)),flush=True)
