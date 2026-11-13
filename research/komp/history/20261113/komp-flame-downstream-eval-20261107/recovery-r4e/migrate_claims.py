"""Authorized per-shard restart after verified terminal old owners; frozen batches preserved."""
from pathlib import Path
import argparse,os,json,hashlib,shutil,datetime
ROOT=Path(__file__).resolve().parent.parent
REC=ROOT/'recovery-r4e'
OLD={0:'job-de184fdde2fc-20261107192117',1:'job-b975c4a6b8da-20261108013315',2:'job-ab970a5f32c2-20261108013328'}
NAMES=['block123','learned','learned123','block']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,choices=[0,1,2],required=True);s=ap.parse_args().shard
 new=json.loads(Path(os.environ['EVAL_JOB_RECEIPT']).read_text())
 assert new['alias']==os.environ['EVAL_JOB_ALIAS']==f'komp340m-downstream-eval-20261108-r4e-s{s}'
 assert new['identity']!=OLD[s]
 check=json.loads((REC/'precheck.json').read_text());data=check['jobs'][OLD[s]]['overview']['data']
 assert data['phase']=='Aborted' and data['ready_status']=='0/1'
 assert all(x['status'] in ['Failed','Succeeded'] for x in data['instance_info']['instances'])
 before=json.loads((REC/'before.json').read_text())['shards'][str(s)]
 for rel,h in before['files'].items():assert sha(ROOT/rel)==h,rel
 for rel,h in before['batch_hashes'].items():assert sha(ROOT/rel)==h,rel
 for name in ['kda','value','exact123','exact_partial']:
  assert json.loads((ROOT/f'sharded/evidence-v2/{name}-s{s}-complete.json').read_text())['status']=='completed'
 archive=REC/f'claim-evidence-archive-s{s}'
 assert not archive.exists()
 for name in NAMES:
  p=ROOT/'sharded/claims-v2'/f'{name}-s{s}'
  owner=json.loads((p/'owner.json').read_text())
  assert owner['job_identity']==OLD[s] and owner['model']==name and owner['shard']==s
  assert not (ROOT/f'sharded/evidence-v2/{name}-s{s}-complete.json').exists()
 archive.mkdir()
 for name in NAMES:
  p=ROOT/'sharded/claims-v2'/f'{name}-s{s}';shutil.move(p,archive/p.name)
  for p in (ROOT/'sharded/evidence-v2').glob(f'{name}-s{s}-*.json'):shutil.copy2(p,archive/p.name)
 (REC/f'migration-s{s}.json').write_text(json.dumps(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),old_identity=OLD[s],new_identity=new['identity'],models=NAMES,shard=s,skipped=['kda','value','exact123','exact_partial'],verified_old_batches=len(before['batch_hashes']),method='explicit migration after old job Aborted and all instances terminal; old batches unchanged'),indent=2)+'\n')
if __name__=='__main__':main()
