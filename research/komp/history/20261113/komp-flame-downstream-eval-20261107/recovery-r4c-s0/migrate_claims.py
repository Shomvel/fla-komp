"""Single authorized migration from verified terminal r4b-s0 to r4c-s0."""
from pathlib import Path
import os,json,hashlib,shutil
ROOT=Path(__file__).resolve().parent.parent
REC=ROOT/'recovery-r4c-s0'
OLD='job-bdaafde6397e-20261107143915'
NAMES=['block123','exact123','learned','learned123','block','value','exact_partial']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 new=json.loads(Path(os.environ['EVAL_JOB_RECEIPT']).read_text())
 assert new['alias']==os.environ['EVAL_JOB_ALIAS']=='komp340m-downstream-eval-20261107-r4c-s0'
 assert new['identity']!=OLD
 check=json.loads((REC/'precheck.json').read_text())
 data=check['jobs'][OLD]['overview']['data']
 assert data['phase']=='Aborted' and data['ready_status']=='0/1'
 assert data['instance_info']['instances']==[]
 before=json.loads((REC/'before.json').read_text())
 for rel,h in before['files'].items():assert sha(ROOT/rel)==h,rel
 assert json.loads((ROOT/'sharded/evidence-v2/kda-s0-complete.json').read_text())['status']=='completed'
 archive=REC/'claim-evidence-archive';archive.mkdir(exist_ok=False)
 # Validate the complete scope before moving any claim. Never touch KDA/s1/s2.
 for name in NAMES:
  p=ROOT/'sharded/claims-v2'/f'{name}-s0'
  owner=json.loads((p/'owner.json').read_text())
  assert owner['job_identity']==OLD and owner['model']==name and owner['shard']==0
 for name in NAMES:
  p=ROOT/'sharded/claims-v2'/f'{name}-s0';shutil.move(p,archive/p.name)
  for p in (ROOT/'sharded/evidence-v2').glob(f'{name}-s0-*.json'):
   shutil.copy2(p,archive/p.name)
 # Lock migration evidence precedes any worker launch. Batch files remain untouched.
 (REC/'migration.json').write_text(json.dumps(dict(old_identity=OLD,new_identity=new['identity'],models=NAMES,shard=0,kda_skipped=True,verified_before_files=len(before['files']),method='explicit owner migration after platform Aborted and zero instances; no TTL takeover'),indent=2)+'\n')
if __name__=='__main__':main()
