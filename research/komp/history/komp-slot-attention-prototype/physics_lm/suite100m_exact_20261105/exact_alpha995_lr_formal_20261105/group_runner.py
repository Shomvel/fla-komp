"""Two additional100M LR runs per task, reusing identical-source preflight."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TASK=os.environ['GROUP_TASK']
manifest=json.loads((ROOT/'source-manifest.json').read_text())
for name,wanted in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==wanted,name
for name,wanted in json.loads((ROOT/'evidence/snapshot-sha256.json').read_text()).items():assert hashlib.sha256((ROOT/'snapshot'/name).read_bytes()).hexdigest()==wanted,name
binding=json.loads((ROOT/'evidence/preflight-reuse.json').read_text())
ticket=ROOT/'runs-ddp/ddp-preflight/exact.json'
assert hashlib.sha256(ticket.read_bytes()).hexdigest()==binding['ticket_sha256']
assert json.loads(ticket.read_text())['status']=='passed'
import torch,triton,google.protobuf
from google.protobuf.internal.api_implementation import Type
assert torch.cuda.device_count()==8
(ROOT/'evidence'/f'environment-{TASK}.json').write_text(json.dumps({'gpus':[torch.cuda.get_device_name(i) for i in range(8)],'torch':str(torch.__version__),'triton':triton.__version__,'cuda':torch.version.cuda,'protobuf':google.protobuf.__version__,'protobuf_backend':Type()},indent=2)+'\n')
plan=json.loads((ROOT/'plan.json').read_text())
for lr in plan['new_lrs'][TASK]:
    assert not (ROOT/'FAILED.json').exists(),'Peer failure blocks new runs'
    name=f'{TASK}-exact-final-lr{lr:g}-wd0.1-seed731'
    summary=ROOT/'runs-ddp/results'/name/'summary.json'
    if summary.exists():
        assert json.loads(summary.read_text())['status']=='completed';continue
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(ROOT/'ddp_train.py'),'--kind','exact','--task',TASK,'--seed','731','--lr',str(lr),'--wd','.1']
    print('START',name,flush=True)
    result=subprocess.run(command,cwd=ROOT)
    if result.returncode:
        (ROOT/'FAILED.json').write_text(json.dumps({'task':TASK,'run':name,'exit_code':result.returncode})+'\n')
        raise SystemExit(result.returncode)
    assert summary.exists() and json.loads(summary.read_text())['status']=='completed'
(ROOT/'evidence'/f'training-{TASK}-complete.json').write_text(json.dumps({'status':'completed','new_lrs':plan['new_lrs'][TASK]})+'\n')
