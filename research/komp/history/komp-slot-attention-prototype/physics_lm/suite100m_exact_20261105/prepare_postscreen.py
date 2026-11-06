"""Prepare, but do not activate, a no-training post-screen queue on GPFS."""
import hashlib,json
from pathlib import Path
R=Path('/gpfs/users/linlizhi/physics100m-exact-20261105');T=R/'exact_tuning_20261105'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while part:=f.read(8*1024*1024):h.update(part)
 return h.hexdigest()
plan=json.loads((T/'initial-screen.json').read_text())
for task in ['depo2','brevo2']:
 all_results=[]
 for candidate in plan['screen']:
  path=T/'results'/task/candidate['name']/'summary.json';item=json.loads(path.read_text());assert item['status']=='completed'
  all_results.append((item['val']['ce'],candidate['name']))
 winner=min(all_results)[1];checkpoint=T/'results'/task/winner/'checkpoint.pt';checkpoint_sha=sha(checkpoint)
 p=R/'evidence'/f'priority-{task}.json';manifest=json.loads(p.read_text());commands=manifest['commands'];index=next(i for i,c in enumerate(commands) if c['name']==f'tune-{task}-formal-gate')
 script=R/'check_geometry_transfer.py'
 commands.insert(index,{'name':f'tune-{task}-geometry-transfer','argv':['/opt/venv/bin/python','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(script),'--task',task,'--candidate',winner,'--checkpoint-sha256',checkpoint_sha],'done_path':str(R/'evidence'/f'geometry-transfer-{task}.json'),'source_sha256':{str(script):sha(script)}})
 if task=='brevo2':
  index=next(i for i,c in enumerate(commands) if c['name']==f'tune-{task}-formal-gate');script=R/'diagnose_gsa2_adam_witness.py'
  commands.insert(index,{'name':'tune-brevo2-native-adam-witness','argv':['/opt/venv/bin/python',str(script)],'done_path':str(R/'evidence/gsa2-bf16-adam-witness.json'),'source_sha256':{str(script):sha(script)}})
 (R/'evidence'/f'priority-{task}-postscreen-draft.json').write_text(json.dumps(manifest,indent=2)+'\n')
 (R/'evidence'/f'postscreen-selection-{task}.json').write_text(json.dumps({'status':'prepared','task':task,'winner':winner,'ranking':all_results,'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_sha,'new_training_steps':0},indent=2)+'\n')
 print(task,winner,checkpoint_sha,flush=True)
