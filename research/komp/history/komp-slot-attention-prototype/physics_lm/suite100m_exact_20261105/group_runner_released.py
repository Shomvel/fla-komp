"""One8GPU group per task, serial models; all7 DDP checks precede training."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
TASK=os.environ['GROUP_TASK']
FORMAL_KINDS=['exact','komp','gdn','full','swa','gsa']
ASSIGNED={'depo2':['gdn','exact','full','komp'],'brevo2':['gsa2','gsa','swa']}[TASK]
KINDS=['exact','komp','gdn','full','swa','gsa','gsa2']
EXPECTED=json.loads((ROOT/'evidence/snapshot-sha256.json').read_text())


def verify():
    own=json.loads((ROOT/'evidence/released-runner-sha256.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==own[Path(__file__).name]
    for name,wanted in EXPECTED.items():
        assert hashlib.sha256((ROOT/'snapshot'/name).read_bytes()).hexdigest()==wanted,name
    wanted=json.loads((ROOT/'evidence/ddp-source-sha256.json').read_text())
    for name,digest in wanted.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name


def failed():
    return (ROOT/'ddp-FAILED.json').exists()


def run(command,name):
    if (ROOT/'ddp-STOP').exists():raise SystemExit(0)
    assert not failed(),'Peer failure blocks new commands'
    verify()
    started=time.time()
    print('START',name,flush=True)
    with (ROOT/'logs'/f'{name}.log').open('a') as out:
        status=subprocess.call(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
    record={'name':name,'task':TASK,'exit_code':status,'seconds':time.time()-started,'command':command}
    with (ROOT/'evidence'/f'group-{TASK}-progress.jsonl').open('a') as out:
        out.write(json.dumps(record)+'\n')
    print('END',json.dumps(record),flush=True)
    if status:
        try:
            with (ROOT/'ddp-FAILED.json').open('x') as out:json.dump(record,out)
        except FileExistsError:pass
        raise RuntimeError(name)


def torchrun(args):
    return [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(ROOT/'ddp_train.py'),*args]



def run_priority():
    manifest=ROOT/'evidence'/f'priority-released-{TASK}.json'
    print('WAIT_PRIORITY',str(manifest),flush=True)
    while True:
        if (ROOT/'ddp-STOP').exists():raise SystemExit(0)
        assert not failed(),'Peer failure blocks priority work'
        if manifest.exists():
            plan=json.loads(manifest.read_text())
            if plan.get('ready') is True:break
        time.sleep(5)
    assert plan['task']==TASK and plan['world_size']==8
    for entry in plan['commands']:
        name=entry['name']
        assert name.startswith(f'tune-{TASK}-') and '/' not in name
        for filename,wanted in entry['source_sha256'].items():
            assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==wanted,filename
        done=Path(entry['done_path'])
        if done.exists():
            result=json.loads(done.read_text())
            assert result.get('status') in ('passed','completed'),str(done)
            continue
        args=entry['argv']
        assert isinstance(args,list) and all(isinstance(x,str) for x in args)
        run(args,name)
        assert done.exists(),f'Priority command missing success artifact: {done}'
        assert json.loads(done.read_text()).get('status') in ('passed','completed'),str(done)
    (ROOT/'evidence'/f'priority-{TASK}-complete.json').write_text(json.dumps({'status':'completed','commands':len(plan['commands']),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}))


def main():
    import torch,triton
    import google.protobuf
    from google.protobuf.internal.api_implementation import Type
    assert torch.cuda.device_count()==8
    env={'task':TASK,'gpus':[torch.cuda.get_device_name(i) for i in range(8)],'torch':str(torch.__version__),
         'triton':triton.__version__,'cuda':torch.version.cuda,'protobuf':google.protobuf.__version__,'protobuf_backend':Type()}
    (ROOT/'evidence'/f'group-{TASK}-environment.json').write_text(json.dumps(env,indent=2))
    for kind in ASSIGNED:
        ticket=ROOT/'runs-ddp/ddp-preflight'/f'{kind}.json'
        current={'snapshot':hashlib.sha256((ROOT/'evidence/snapshot-sha256.json').read_bytes()).hexdigest(),
                 'ddp_train':hashlib.sha256((ROOT/'ddp_train.py').read_bytes()).hexdigest()}
        if ticket.exists():
            old=json.loads(ticket.read_text())
            if old.get('status')=='passed' and old.get('world_size')==8 and old.get('sources')==current:
                continue
            ticket.rename(ticket.with_suffix(f'.superseded-{time.time_ns()}.json'))
        run(torchrun(['--preflight','--kind',kind]),f'ddp-preflight-{kind}')
    def valid_ticket(kind):
        path=ROOT/'runs-ddp/ddp-preflight'/f'{kind}.json'
        if not path.exists():return False
        item=json.loads(path.read_text())
        return item.get('status')=='passed' and item.get('world_size')==8 and item.get('sources')==current
    while not all(valid_ticket(kind) for kind in KINDS):
        assert not failed(),'Peer preflight failure'
        if (ROOT/'ddp-STOP').exists():return
        time.sleep(5)
    run_priority()
    names=[]
    for kind in FORMAL_KINDS:
        if (ROOT/'ddp-STOP').exists():return
        selected=json.loads((ROOT/'runs-ddp/selection'/f'{TASK}-{kind}.json').read_text())
        lr,wd=selected['lr'],selected['wd']
        name=f'{TASK}-{kind}-final-lr{lr:g}-wd{wd:g}-seed731';names.append(name)
        summary=ROOT/'runs-ddp/results'/name/'summary.json'
        if not summary.exists():
            run(torchrun(['--kind',kind,'--task',TASK,'--seed','731','--lr',str(lr),'--wd',str(wd)]),name)
    # Training uses all8 GPUs for one model. Once training is finished, independent
    # heldout evaluators use one GPU each, preserving the mature resumable scorer.
    processes=[]
    for device,name in enumerate(names):
        if (ROOT/'runs-ddp/results'/name/'heldout.json').exists():continue
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(device)
        log=(ROOT/'logs'/f'eval-{name}.log').open('a')
        cmd=[sys.executable,str(ROOT/'snapshot/evaluate.py'),'--run',name]
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        processes.append((process,name,log))
    for process,name,log in processes:
        status=process.wait();log.close()
        if status:
            (ROOT/'ddp-FAILED.json').write_text(json.dumps({'stage':'evaluate','run':name,'exit_code':status}))
            raise RuntimeError(name)
    report={'task':TASK,'status':'completed','runs':len(FORMAL_KINDS),'evaluations':len(FORMAL_KINDS),'held_models':['gsa2'],'scope':'Six models completed; original GSA2 request remains pending numerical diagnosis'}
    (ROOT/'evidence'/f'group-{TASK}-six-complete.json').write_text(json.dumps(report,indent=2))
    print('GROUP_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
