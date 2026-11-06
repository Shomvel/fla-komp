"""Keep formal comparisons behind independent native GSA2 layout acceptance."""
import argparse,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--task',required=True);a=p.parse_args()
marker=ROOT/'evidence/gsa2-layout-acceptance.json'
while not marker.exists():
    if (ROOT/'ddp-STOP').exists() or (ROOT/'ddp-FAILED.json').exists():raise SystemExit('Experiment stopped while waiting for native GSA2 acceptance')
    print('WAIT_NATIVE_GSA2_ACCEPTANCE',str(marker),flush=True)
    time.sleep(30)
report=json.loads(marker.read_text());assert report['status']=='passed'
done=ROOT/'evidence'/f'formal-gate-{a.task}.json'
done.write_text(json.dumps({'status':'passed','task':a.task,'acceptance':report},indent=2))
