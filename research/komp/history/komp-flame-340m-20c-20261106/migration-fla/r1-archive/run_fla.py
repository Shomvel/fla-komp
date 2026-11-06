import json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent;kind=sys.argv[1];assert kind in ('exact','block')
folder=json.loads((root.parents[0]/f'stop-{kind}.json').read_text())['latest']['path']
subprocess.run([sys.executable,str(root/'validate_compat.py'),kind,folder],check=True)
subprocess.run(['torchrun','--standalone','--nproc_per_node=8',str(root/'migrate.py'),'--kind',kind],check=True)
subprocess.run(['torchrun','--standalone','--nproc_per_node=8',str(root/'train.py'),'--kind',kind],check=True)
