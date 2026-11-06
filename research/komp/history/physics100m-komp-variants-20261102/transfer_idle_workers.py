"""Transfer verified idle original aliases to isolated variants, never exceeding16jobs."""
import json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent;e=r/'evidence';helper='/Users/shom/.codex/skills/northjob-gpu-test-bp-dev/scripts/northjob.py'

def call(label,args):
 path=e/(label+'.json');assert not path.exists(),'receipt already exists: '+label
 p=subprocess.run([sys.executable,helper,*args],capture_output=True,text=True,timeout=150);path.write_text(p.stdout)
 assert p.returncode==0,label
 return json.loads(p.stdout)

probe=subprocess.run(['ssh','linlizhi-bp-dev','python3','-'],input=(r/'probe_remote.py').read_text(),capture_output=True,text=True,timeout=150)
assert probe.returncode==0
state=json.loads(probe.stdout);assert state['failure'] is None
assert set(state['tickets'])=={'komp_learned','komp_gdn_slot','komp_learned_gdn_slot'}
assert all(t['status']=='passed' for t in state['tickets'].values())
(e/'preflight-passed.json').write_text(probe.stdout)
for new,old in enumerate([1,3,4,6,7,8,9,10,12,14,15],2):
 old_alias=f'physics100m-20261101-r{old}';new_alias=f'physics100m-komp-variants-20261102-r{new}'
 assert call(f'new-r{new}-before-submit',['jobs','--alias',new_alias])==[]
 rows=call(f'old-r{old}-before-delete',['jobs','--alias',old_alias]);assert len(rows)==1 and rows[0]['status']=='Running' and rows[0]['user_def_name']==old_alias
 p=subprocess.run(['ssh','linlizhi-bp-dev','python3','-',str(old)],input=(r/'inspect_idle_remote.py').read_text(),capture_output=True,text=True,timeout=150)
 (e/f'old-r{old}-idle.json').write_text(p.stdout)
 assert p.returncode==0,'original worker not idle; do not retry blindly: '+old_alias
 call(f'old-r{old}-delete',['delete','--alias',old_alias])
 assert call(f'old-r{old}-absent',['jobs','--alias',old_alias])==[]
 result=call(f'new-r{new}-submit',['submit','--yaml',str(e/f'job-r{new}.yaml')]);assert result.get('success') is not False
 print(json.dumps(dict(retired=old_alias,submitted=new_alias)),flush=True)
