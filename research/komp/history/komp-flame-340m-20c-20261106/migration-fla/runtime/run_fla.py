import json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent;kind=sys.argv[1];assert kind in ('exact','block')
folder=json.loads((root.parents[0]/f'stop-{kind}.json').read_text())['latest']['path']
subprocess.run([sys.executable,str(root/'old_replay.py'),kind,folder],check=True)
subprocess.run([sys.executable,str(root/'profile_memory.py'),kind,folder],check=True)
memory=json.loads((root/f'evidence/memory-{kind}.json').read_text())
selected=memory['activation_memory'][0 if memory['selected_checkpoint_policy']=='mixer_only' else 1]
assert selected['conservative_full_stack_gib']<70, 'Measured saved-storage estimate exceeds safe full-stack capacity'
subprocess.run(['torchrun','--standalone','--nproc_per_node=8',str(root/'real_compat.py'),kind,folder],check=True)
subprocess.run(['torchrun','--standalone','--nproc_per_node=8',str(root/'migrate.py'),'--kind',kind],check=True)
subprocess.run(['torchrun','--standalone','--nproc_per_node=8',str(root/'train.py'),'--kind',kind],check=True)
