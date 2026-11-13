"""Read atomic batch outputs and publish only complete exact-once cases."""
import json,datetime,os,socket,shutil
from pathlib import Path
from sharding import digest,collect_model,write_atomic,lock_worker
ROOT=Path(__file__).resolve().parent
def main():
 # Atomic namespace claim; a crashed collector claim requires explicit owner verification.
 claim=ROOT/'sharded/collector-claim'
 claim.mkdir(exist_ok=False)
 write_atomic(claim/'owner.json',dict(host=socket.gethostname(),pid=os.getpid()))
 try:return collect()
 finally:shutil.rmtree(claim)
def collect():
 plan=json.loads((ROOT/'sharded/plan.json').read_text());plan_sha=digest(ROOT/'sharded/plan.json')
 all_summaries=[];all_progress=[]
 for name in plan['models']:
  summaries,progress=collect_model(name,plan,plan_sha)
  all_summaries.extend(summaries);all_progress.extend(progress)
  for rec in summaries:write_atomic(ROOT/'sharded/results'/name/f"{rec['case']}.summary.json",rec)
 states={p.name:json.loads(p.read_text()) for p in sorted((ROOT/'sharded/evidence-v2').glob('*.json'))}
 out=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha=plan_sha,states=states,summaries=all_summaries,progress=all_progress)
 write_atomic(ROOT/'sharded/latest.json',out)
 print(json.dumps(out))
 return out
if __name__=='__main__':main()
