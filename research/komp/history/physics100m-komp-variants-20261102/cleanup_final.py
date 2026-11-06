"""Remove only the 12 completed, identity-pinned variant workers."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;DEST=ROOT/'evidence/cleanup-final-20261102T1049Z';DEST.mkdir(exist_ok=True)
helper='/Users/shom/.codex/skills/northjob-gpu-test-bp-dev/scripts/northjob.py'
expected={j['user_def_name']:j['name'] for j in json.loads((ROOT/'evidence/jobs-20261102T1049Z.json').read_text())}
def call(label,*args):
 p=DEST/(label+'.json');assert not p.exists()
 result=subprocess.run([sys.executable,helper,*args],capture_output=True,text=True,timeout=120)
 p.write_text(result.stdout);assert result.returncode==0,result.stderr
 return json.loads(result.stdout)
for i in range(1,13):
 alias=f'physics100m-komp-variants-20261102-r{i}'
 rows=call(f'r{i}-terminal','jobs','--alias',alias)
 assert len(rows)==1 and rows[0]['name']==expected[alias] and rows[0]['status']=='Completed' and rows[0]['ready_status']=='0/1'
 call(f'r{i}-delete','delete','--alias',alias)
 assert call(f'r{i}-absent','jobs','--alias',alias)==[]
 print(alias,'deleted and confirmed absent',flush=True)
assert call('all-absent','jobs','--prefix','physics100m-komp-variants-20261102-r')==[]
