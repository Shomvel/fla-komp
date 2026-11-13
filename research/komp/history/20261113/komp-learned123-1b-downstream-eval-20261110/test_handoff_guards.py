"""CPU-only acceptance for final-budget gating and one-shot submission."""
import json,tempfile
from pathlib import Path
import watch_after_training as w
import prepare_final as p
def rejects(fn,kind):
    try:fn()
    except kind:return
    raise AssertionError('Expected rejection')
with tempfile.TemporaryDirectory() as tmp:
    r=Path(tmp);run=r/'runs/block';run.mkdir(parents=True)
    final=run/'complete.json';final.write_text(json.dumps(dict(status='completed',step=31744,supervised_tokens=33285996544)))
    (run/'latest.json').write_text(json.dumps(dict(step=31744,tokens=33285996544)))
    job=dict(name='training-id',status='Running')
    assert not w.training_ready(job,r,'training-id')
    rejects(lambda:w.training_ready(dict(job,status='Completed'),r,'training-id'),AssertionError)
    rejects(lambda:w.training_ready(dict(job,status='Aborted'),r,'training-id'),RuntimeError)
    rejects(lambda:w.training_ready(job,r,'wrong-id'),AssertionError)
    rejects(lambda:p.prepare(r),AssertionError)
    final.write_text(json.dumps(dict(status='completed',step=47684,supervised_tokens=50000000000)))
    for i in range(4):
        q=r/f'nodes/{i}';q.mkdir(parents=True);(q/'exit.txt').write_text('0')
    assert w.training_ready(dict(job,status='Completed'),r,'training-id')
    (r/'nodes/3/exit.txt').write_text('1')
    rejects(lambda:w.training_ready(dict(job,status='Completed'),r,'training-id'),AssertionError)
    w.ROOT=r;w.WATCH=r/'watcher';w.WATCH.mkdir();(r/'sharded').mkdir();(r/'job.yaml').write_text('placeholder')
    calls=[];existing=[]
    def fake(action,**kwargs):
        calls.append(action)
        if action=='submit':existing.append(dict(name='eval-id',status='Pending'));return {'code':200}
        return list(existing)
    w.api=fake
    assert w.submit_once()['name']=='eval-id'
    assert w.submit_once()['name']=='eval-id'
    assert calls.count('submit')==1
    # A prior intent with no visible job must never be POSTed again.
    existing.clear()
    rejects(w.submit_once,FileExistsError)
    assert calls.count('submit')==1
print(json.dumps({'status':'passed','checks':['running_waits','partial_completed_rejected','aborted_rejected','identity_mismatch_rejected','partial_pin_rejected','all_four_exits_required','completed50b_accepted','repeat_submit_uses_receipt','ambiguous_prior_intent_never_reposts']}))
