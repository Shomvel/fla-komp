"""Resume only terminal owned8GPU groups; no automatic numerical-error retries."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
HELPER=Path('/Users/shom/.codex/skills/northjob-gpu-test-bp-dev/scripts/northjob.py')
REMOTE='/gpfs/users/linlizhi/physics100m-exact-20261105'
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
RECEIPTS=ROOT/'evidence'/f'ddp-recovery-{STAMP}'


def operation(args,receipt):
    result=subprocess.run([sys.executable,str(HELPER),*args],capture_output=True,text=True,check=True)
    obj=json.loads(result.stdout);(RECEIPTS/receipt).write_text(json.dumps(obj,indent=2));return obj


def main():
    RECEIPTS.mkdir(parents=True,exist_ok=False)
    check=subprocess.run(['ssh','-o','BatchMode=yes','linlizhi-bp-dev','test','!','-f',REMOTE+'/ddp-FAILED.json'])
    assert check.returncode==0,'Investigate recorded command/numerical failure before retrying'
    check=subprocess.run(['ssh','-o','BatchMode=yes','linlizhi-bp-dev','test','!','-f',REMOTE+'/ddp-STOP'])
    assert check.returncode==0,'An explicit stop is active; remove only when user has resumed training'
    for task in ('depo2','brevo2'):
        if (ROOT/'evidence'/f'group-{task}-eval-offload.json').exists():
            print(task,'evaluation managed by independent 1GPU jobs; do not restart original 8GPU group');continue
        alias=f'physics100m-exact-ddp-{task}-20261105'
        rows=operation(['jobs','--alias',alias],f'{task}-terminal.json')
        if len(rows)!=1 or rows[0]['status'] not in ('Completed','Aborted','Failed'):
            print(alias,'is not a unique terminal group; no action');continue
        operation(['delete','--alias',alias],f'{task}-delete.json')
        assert operation(['jobs','--alias',alias],f'{task}-absent.json')==[]
        finished=subprocess.run(['ssh','-o','BatchMode=yes','linlizhi-bp-dev','test','-f',REMOTE+f'/evidence/group-{task}-complete.json'])
        six_finished=subprocess.run(['ssh','-o','BatchMode=yes','linlizhi-bp-dev','test','-f',REMOTE+f'/evidence/group-{task}-six-complete.json'])
        if finished.returncode==0:
            print(alias,'complete and cleaned');continue
        if six_finished.returncode==0:
            print(alias,'six-model training and evaluation phase complete and cleaned; GSA2 remains held');continue
        # group_runner skips completed summaries, resumes incomplete model/AdamW/
        # all-rank RNG checkpoints, and continues resumable generation outputs.
        operation(['submit','--yaml',str(ROOT/f'job-ddp-{task}.yaml')],f'{task}-submit.json')
        print(alias,'resubmitted unchanged8GPU group,24hour deadline')


if __name__=='__main__':main()
