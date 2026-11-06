"""Read-only validation of the exact mature evaluator's resume identity."""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'snapshot'))
sys.path.insert(0,str(ROOT/'snapshot/suite100m'))
from tasks import Factory
from eval_progress import load_progress
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--gpu',action='store_true');p.add_argument('--hash-checkpoint',action='store_true');a=p.parse_args()
for name,wanted in json.loads((ROOT/'evidence/snapshot-sha256.json').read_text()).items():
    assert hashlib.sha256((ROOT/'snapshot'/name).read_bytes()).hexdigest()==wanted,name
dest=ROOT/'runs-ddp/results'/a.run
summary=json.loads((dest/'summary.json').read_text());cfg=summary['config']
assert summary['status']=='completed' and cfg['phase']=='final'
assert cfg['model']['kind'] in ('exact','komp')
assert not (dest/'heldout.json').exists(),'Already complete; no evaluator needed'
probes=json.loads((Path(os.environ['SUITE_DATA'])/cfg['task']/'test.probes.json').read_text())[:256]
stat=(dest/'checkpoint.pt').stat()
identity=dict(run=a.run,summary_sha256=hashlib.sha256((dest/'summary.json').read_bytes()).hexdigest(),
    probes_sha256=hashlib.sha256(json.dumps(probes,sort_keys=True).encode()).hexdigest(),
    evaluator_sha256=hashlib.sha256((ROOT/'snapshot/evaluate.py').read_bytes()+(ROOT/'snapshot/eval_progress.py').read_bytes()).hexdigest(),
    checkpoint_size=stat.st_size,checkpoint_mtime_ns=stat.st_mtime_ns)
progress=dest/'eval-progress.json'
for attempt in range(3):
    original=progress.read_bytes()
    resume=load_progress(progress,identity,probes,Factory(cfg['task']).grade)
    if original==progress.read_bytes():break
else:raise RuntimeError('Progress changing during audit; no migration action')
assert resume is not None,'No persisted progress to migrate'
report={'status':'passed','run':a.run,'identity':identity,'saved_probes':len(resume['records']),
        'test':resume['test'],'seconds':resume['seconds'],'snapshot_manifest_sha256':hashlib.sha256((ROOT/'evidence/snapshot-sha256.json').read_bytes()).hexdigest()}
report['progress_sha256']=hashlib.sha256(original).hexdigest()
if a.hash_checkpoint:
    h=hashlib.sha256()
    with (dest/'checkpoint.pt').open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    report['checkpoint_sha256']=h.hexdigest()
if a.gpu:
    import platform,torch,triton,google.protobuf
    from google.protobuf.internal.api_implementation import Type
    prior=json.loads((ROOT/'evidence'/f"group-{cfg['task']}-environment.json").read_text())
    assert torch.cuda.device_count()==1
    assert torch.cuda.get_device_name(0)==prior['gpus'][0]
    assert str(torch.__version__)==prior['torch'] and triton.__version__==prior['triton'] and torch.version.cuda==prior['cuda']
    report['environment']={'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),'triton':triton.__version__,'cuda':torch.version.cuda,'protobuf':google.protobuf.__version__,'protobuf_backend':Type()}
print(json.dumps(report))
