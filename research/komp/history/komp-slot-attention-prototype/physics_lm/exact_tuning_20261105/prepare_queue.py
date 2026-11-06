"""Build reviewable priority manifests; does not submit jobs or publish readiness."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SUITE=HERE.parent/'suite100m_exact_20261105'
REMOTE=Path('/gpfs/users/linlizhi/physics100m-exact-20261105')
TUNING=REMOTE/HERE.name
plan=json.loads((HERE/'initial-screen.json').read_text())
assert len(plan['screen'])==9
assert len({c['name'] for c in plan['screen']})==9
pins={str(TUNING/p.name):hashlib.sha256(p.read_bytes()).hexdigest()
      for p in [*sorted(HERE.glob('*.py')),HERE/'initial-screen.json']}
for name in ('ddp_train.py','evidence/snapshot-sha256.json'):
    pins[str(REMOTE/name)]=hashlib.sha256((SUITE/name).read_bytes()).hexdigest()
(HERE/'queue').mkdir(exist_ok=True)
for task in ('depo2','brevo2'):
    old_path=HERE/'queue'/f'priority-{task}.json'
    old=json.loads(old_path.read_text())['commands'] if old_path.exists() else []
    before=[e for e in old if e['name'].endswith(('-native-layout','-native-bf16-grad'))]
    after=[e for e in old if e['name'].endswith('-formal-gate')]
    commands=[{'name':f'tune-{task}-numerical','argv':['/opt/venv/bin/python',str(TUNING/'numerical.py'),'--task',task],
               'done_path':str(TUNING/'evidence'/f'numerical-{task}.json'),'source_sha256':pins}]
    commands.append({'name':f'tune-{task}-precision-ddp',
        'argv':['/opt/venv/bin/python','-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                str(TUNING/'tuned_preflight.py'),'--task',task],
        'done_path':str(TUNING/'preflight'/task/'runs-ddp/ddp-preflight/exact.json'),'source_sha256':pins})
    for candidate in plan['screen']:
        name=candidate['name']
        commands.append({'name':f'tune-{task}-{name}',
            'argv':['/opt/venv/bin/python','-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                    str(TUNING/'tune.py'),'--task',task,'--candidate',name],
            'done_path':str(TUNING/'results'/task/name/'summary.json'),'source_sha256':pins})
    manifest={'task':task,'world_size':8,'ready':True,'commands':before+commands+after,
              'scope':'9 Exact candidates x 10M train positions; final validation CE selection'}
    (HERE/'queue'/f'priority-{task}.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Prepared two manifests; publish atomically only after uploads and DDP source freeze.')
