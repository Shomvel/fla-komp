"""Durable one-shot handoff: completed 50B training -> unique evaluation job."""
import contextlib, datetime, io, json, os, socket, subprocess, time
from pathlib import Path
from northjob_remote import main as northjob
from sharding import write_atomic
ROOT=Path(__file__).resolve().parent
WATCH=ROOT/'watcher'
ALIAS='komp1b-downstream-eval-20261110'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def api(action,**extra):
    opts=dict(action=action,namespace='qiji',creator='linlizhi',**extra)
    output=io.StringIO()
    with contextlib.redirect_stdout(output):northjob(opts)
    return json.loads(output.getvalue())
def training_ready(job,runroot,identity):
    assert job['name']==identity,'Training identity changed'
    if job['status'] in ['Aborted','Failed','Terminated']:
        raise RuntimeError('Training terminated without completion: '+job['status'])
    if job['status']!='Completed':return False
    final=json.loads((runroot/'runs/block/complete.json').read_text())
    assert final['status']=='completed' and final['supervised_tokens']==50000000000 and final['step']==47684
    for i in range(4):assert (runroot/f'nodes/{i}/exit.txt').read_text().strip()=='0'
    return True
def submit_once():
    existing=api('jobs',alias=ALIAS)
    if existing:
        assert len(existing)==1
        receipt=ROOT/'sharded/job.json'
        assert receipt.exists() and json.loads(receipt.read_text())['identity']==existing[0]['name'], 'Unowned alias exists'
        return existing[0]
    attempt=WATCH/'submission-attempt.json'
    # Exclusive durable intent before POST: a crash/timeout never causes a repeat POST.
    with attempt.open('x') as f:
        json.dump(dict(time=now(),alias=ALIAS),f);f.flush();os.fsync(f.fileno())
    try:
        response=api('submit',yaml=(ROOT/'job.yaml').read_text())
        write_atomic(WATCH/'submission-response.json',response)
    except Exception as e:
        write_atomic(WATCH/'submission-uncertain.json',dict(time=now(),exception_type=type(e).__name__,action='reconcile exact alias; never repeat POST'))
    for _ in range(30):
        try:jobs=api('jobs',alias=ALIAS)
        except Exception:
            time.sleep(10);continue
        if jobs:
            assert len(jobs)==1
            job=jobs[0]
            write_atomic(ROOT/'sharded/job.json',dict(alias=ALIAS,identity=job['name'],created_utc=now()))
            return job
        time.sleep(10)
    raise RuntimeError('Submission not reconciled; intent retained and POST will not repeat')
def execute():
    config=json.loads((WATCH/'config.json').read_text());runroot=Path(config['runroot'])
    last_error=None
    while True:
        try:jobs=api('jobs',alias=config['training_alias'])
        except Exception as e:
            last_error=type(e).__name__
            write_atomic(WATCH/'status.json',dict(status='waiting_for_platform',time=now(),exception_type=last_error))
            time.sleep(60);continue
        if len(jobs)!=1:raise RuntimeError('Training exact alias missing or ambiguous')
        job=jobs[0]
        write_atomic(WATCH/'status.json',dict(status='waiting_for_50b_training',time=now(),job=job,runroot=str(runroot)))
        if training_ready(job,runroot,config['training_identity']):break
        time.sleep(60)
    write_atomic(WATCH/'training-platform-completed.json',dict(time=now(),job=job))
    from prepare_final import prepare
    pinned=prepare(runroot)
    assert pinned['source_sha']==config['source_sha'] and pinned['data_sha']==config['data_sha'], 'Training source/data identity changed'
    write_atomic(WATCH/'final-checkpoint-verified.json',dict(time=now(),checkpoint=pinned['checkpoint'],checkpoint_sha=pinned['checkpoint_sha']))
    job=submit_once()
    write_atomic(WATCH/'status.json',dict(status='eval_submitted',time=now(),job=job))
    # Wait for all 8 real GPU workers' atomic probes, then release their load gate.
    running_since=None
    while not (ROOT/'sharded/cross-node-verified.json').exists():
        jobs=api('jobs',alias=ALIAS)
        assert len(jobs)==1 and jobs[0]['name']==job['name']
        if jobs[0]['status'] in ['Aborted','Failed']:raise RuntimeError('Evaluation failed before worker probes')
        if jobs[0]['status']=='Running' and running_since is None:running_since=time.time()
        write_atomic(WATCH/'status.json',dict(status='waiting_for_eval_workers',time=now(),job=jobs[0]))
        probes=list((ROOT/'sharded/claims-v2').glob('*/publish-probe.json'))
        if len(probes)==8:
            subprocess.run(['python3',str(ROOT/'verify_storage.py')],check=True);break
        if running_since is not None and time.time()-running_since>1800:raise RuntimeError('Evaluation worker probe timeout; no automatic resubmit')
        time.sleep(10)
    # Keep collecting until success/failure so the persistent handoff has evidence.
    from collect_sharded_remote import main as collect
    while True:
        with contextlib.redirect_stdout(io.StringIO()):snapshot=collect()
        states=snapshot['states'];failures={k:v for k,v in states.items() if k.endswith('-failure.json')}
        if failures:
            write_atomic(WATCH/'worker-failures.json',failures)
            raise RuntimeError('Evaluation worker failure; preserve evidence and stop automatic control')
        jobs=api('jobs',alias=ALIAS)
        assert len(jobs)==1 and jobs[0]['name']==job['name']
        done=sum(p['count'] for p in snapshot['progress'])
        accepts=sum(k.endswith('-acceptance.json') for k in states)
        write_atomic(WATCH/'status.json',dict(status='eval_running',time=now(),job=jobs[0],done=done,total=20457,acceptances=accepts,complete_cases=len(snapshot['summaries'])))
        if jobs[0]['status'] in ['Aborted','Failed']:raise RuntimeError('Evaluation platform terminated; no resubmit')
        if jobs[0]['status']=='Completed':
            assert done==20457 and len(snapshot['summaries'])==23
            assert sum(k.endswith('-complete.json') for k in states)==8
            write_atomic(WATCH/'complete.json',dict(status='completed',time=now(),job=jobs[0],done=done,cases=23,plan_sha=snapshot['plan_sha']))
            return
        time.sleep(60)
def main():
    WATCH.mkdir(exist_ok=True)
    claim=WATCH/'claim';claim.mkdir(exist_ok=False)
    write_atomic(claim/'owner.json',dict(host=socket.gethostname(),pid=os.getpid(),time=now()))
    try:execute()
    except Exception as e:
        # Do not serialize API exception bodies, credentials, or auth headers.
        write_atomic(WATCH/'failure.json',dict(time=now(),exception_type=type(e).__name__,error=str(e) if isinstance(e,(AssertionError,RuntimeError,FileNotFoundError,FileExistsError)) else 'Operation failed; inspect local evidence'))
        raise SystemExit(1)
if __name__=='__main__':main()
