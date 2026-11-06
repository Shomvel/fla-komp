"""Freeze executable sources after implementation, before release validation."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
names=['run_fla.py','cpu_contract.py','legacy_models.py','validate_compat.py','migrate.py','batch_benchmark.py','ddp_scaling.py','models.py','data.py','train.py','data_check.py','numerical.py','ddp_preflight.py','environment.py','reference.py']
files=[root/n for n in names]+list((root/'snapshot').rglob('*.py'))
manifest={'files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}
p=root/'source-manifest.json'
if p.exists():assert json.loads(p.read_text())==manifest,'source changed after freeze; use new release'
else:p.write_text(json.dumps(manifest,indent=2)+'\n')
print(hashlib.sha256(p.read_bytes()).hexdigest())
