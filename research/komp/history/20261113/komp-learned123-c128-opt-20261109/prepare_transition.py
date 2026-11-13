"""Run on BP Dev after the parent saves a complete pause checkpoint and all nodes exit 0."""
import hashlib,json
from pathlib import Path
parent=Path('/gpfs/users/linlizhi/komp-learned123-1b-50b-20261109-mb16-direct')
root=parent.with_name('komp-learned123-1b-50b-20261109-opt1')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while chunk:=f.read(8*1024*1024):h.update(chunk)
 return h.hexdigest()
paused=json.loads((parent/'runs/block/paused.json').read_text())
assert paused['status']=='paused'
assert all((parent/f'nodes/{i}/exit.txt').read_text().strip()=='0' for i in range(4))
latest=json.loads((parent/'runs/block/latest.json').read_text());checkpoint=Path(latest['path'])
complete=json.loads((checkpoint/'complete.json').read_text())
assert complete['step']==paused['step']==latest['step']
assert complete['tokens']==paused['tokens']==latest['tokens']==complete['step']*1048576
assert complete['world_size']==32
assert complete['tokens']<50000000000,'Parent already reached target; do not start another run'
manifest=json.loads((parent/'source-manifest.json').read_text())
allowed={'snapshot/fla/ops/komp/block_fast.py','configs/resume-parent.json','configs/optimization.json',
 'job-train.yaml','run_train.sh','models.py','snapshot/fla/modules/rounded_swiglu.py',
 'snapshot/fla/models/komp/modeling_komp.py'}
policy=json.loads((root/'configs/optimization.json').read_text())
assert policy['verified'] is True and policy['loss_tolerance']==.001
for name,expected in policy['verification_evidence_sha256'].items():assert sha(root/'evidence'/name)==expected,(name,'verification evidence drift')
assert sha(root/'snapshot/fla/ops/komp/block_fast.py')==policy['operator_sha256']
changed={n for n,h in manifest['files'].items() if not (root/n).exists() or sha(root/n)!=h}
# This transition descriptor is materialized below after the parent checkpoint is fixed.
changed.add('configs/resume-parent.json')
assert changed<=allowed,changed-allowed
assert {'snapshot/fla/ops/komp/block_fast.py','configs/resume-parent.json','job-train.yaml','run_train.sh'}<=changed
new_files={str(p.relative_to(root)) for p in list(root.glob('*.py'))+list((root/'snapshot').rglob('*.py'))+list((root/'configs').glob('*.json'))+list(root.glob('job-*.yaml'))+list(root.glob('*.sh'))+list((root/'tests').rglob('*.py'))}-set(manifest['files'])
assert new_files<=allowed,new_files-allowed
for name,expected in policy['candidate_files_sha256'].items():assert sha(root/name)==expected,(name,'verified candidate changed')
unchanged=[]
for name,expected in manifest['files'].items():
 assert sha(parent/name)==expected,(name,'parent drift')
 if name not in changed:
  assert sha(root/name)==expected,(name,'unexpected candidate change')
  unchanged.append(name)
assert all((root/n).exists() for n in changed)
transition=dict(parent_root=str(parent),checkpoint_path=str(checkpoint),checkpoint_sha256=sha(checkpoint/'model.pt'),
 rank_sha256={str(i):sha(checkpoint/f'rank{i}.pt') for i in range(32)},source_sha=sha(parent/'source-manifest.json'),
 data_sha=complete['data_sha'],step=complete['step'],tokens=complete['tokens'],world_size=32,
 unchanged_files=unchanged,change=policy['description'],
 changed_files=sorted(changed),added_files=sorted(new_files),optimization_policy_sha=sha(root/'configs/optimization.json'),operator_sha=sha(root/'snapshot/fla/ops/komp/block_fast.py'),
 microbatch=16,accumulation=1,global_sequences=512,target_supervised_tokens=50000000000)
assert complete['source_sha']==transition['source_sha']
(root/'configs/resume-parent.json').write_text(json.dumps(transition,indent=2)+'\n')
(root/'runs/block/latest.json').write_text(json.dumps(dict(path=str(checkpoint),step=complete['step'],tokens=complete['tokens']),indent=2)+'\n')
(root/'evidence/parent-pause.json').write_text(json.dumps(dict(paused=paused,checkpoint_complete=complete,node_exits=[0]*4),indent=2)+'\n')
print(json.dumps({k:v for k,v in transition.items() if k not in ('rank_sha256','unchanged_files')},indent=2))
