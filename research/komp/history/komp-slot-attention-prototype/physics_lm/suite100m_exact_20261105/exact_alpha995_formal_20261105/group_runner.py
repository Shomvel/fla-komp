"""One original-budget alpha995 run; new-source preflight precedes training."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TASK=os.environ['GROUP_TASK']
manifest=json.loads((ROOT/'source-manifest.json').read_text())
for name,wanted in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==wanted,name
for name,wanted in json.loads((ROOT/'evidence/snapshot-sha256.json').read_text()).items():assert hashlib.sha256((ROOT/'snapshot'/name).read_bytes()).hexdigest()==wanted,name
import torch,triton,google.protobuf
from google.protobuf.internal.api_implementation import Type
assert torch.cuda.device_count()==8
(ROOT/'evidence'/f'environment-{TASK}.json').write_text(json.dumps({'gpu':[torch.cuda.get_device_name(i) for i in range(8)],'torch':str(torch.__version__),'triton':triton.__version__,'cuda':torch.version.cuda,'protobuf':google.protobuf.__version__,'protobuf_backend':Type()},indent=2)+'\n')
# Task-specific roots keep DDP preflight tickets/checkpoints separate.
preflight_root=ROOT/f'preflight-{TASK}'
preflight_root.mkdir(exist_ok=True)
for name in (*manifest['files'],'source-manifest.json'):
    path=preflight_root/name;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():continue
    if name=='ddp_train.py':path.write_bytes((ROOT/name).read_bytes())
    else:path.symlink_to(ROOT/name)
if not (preflight_root/'snapshot').exists():(preflight_root/'snapshot').symlink_to(ROOT/'snapshot')
command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8']
result=subprocess.run(command+[str(preflight_root/'ddp_train.py'),'--preflight','--kind','exact'],cwd=ROOT)
if result.returncode:raise SystemExit(result.returncode)
ticket=preflight_root/'runs-ddp/ddp-preflight/exact.json'
assert json.loads(ticket.read_text())['status']=='passed'
target=ROOT/'runs-ddp/ddp-preflight';target.mkdir(parents=True,exist_ok=True)
# Both records have identical source digests. Write one atomic ticket only.
p=target/f'exact-{TASK}.tmp';p.write_bytes(ticket.read_bytes());p.replace(target/'exact.json')
plan=json.loads((ROOT/'plan.json').read_text());lr=plan['lr'][TASK]
args=[str(ROOT/'ddp_train.py'),'--kind','exact','--task',TASK,'--seed','731','--lr',str(lr),'--wd','.1']
result=subprocess.run(command+args,cwd=ROOT)
if result.returncode:raise SystemExit(result.returncode)
name=f'{TASK}-exact-final-lr{lr:g}-wd0.1-seed731'
summary=ROOT/'runs-ddp/results'/name/'summary.json'
assert json.loads(summary.read_text())['status']=='completed'
(ROOT/'evidence'/f'training-{TASK}-complete.json').write_text(json.dumps({'status':'completed','run':name,'summary_sha256':hashlib.sha256(summary.read_bytes()).hexdigest()})+'\n')
