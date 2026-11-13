"""Pin new-model provenance while retaining all historical prompts/batch shapes."""
import json, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REFERENCE = Path('/gpfs/users/linlizhi/komp-flame-downstream-eval-20261107')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def main():
    target=ROOT/'sharded/plan.json'
    assert not target.exists(), 'Refuse to overwrite frozen plan'
    pinned=json.loads((ROOT/'evidence/checkpoint.json').read_text())
    assert not pinned['partial'] and pinned['tokens']==50000000000
    old=json.loads((REFERENCE/'sharded/plan.json').read_text())
    assert sha(ROOT/'evidence/requests-manifest.json')==old['request_manifest_sha']
    assert sha(ROOT/'evidence/generation-contract-audit.json')==old['protocol_audit_sha']
    plan=dict(version=1,shards=8,reference_plan_sha=sha(REFERENCE/'sharded/plan.json'),
              request_manifest_sha=old['request_manifest_sha'],protocol_audit_sha=old['protocol_audit_sha'],
              models={'learned123_1b':dict(pinned,request_manifest_sha=old['request_manifest_sha'],
                         protocol_audit_sha=old['protocol_audit_sha'])},cases=old['cases'])
    total=0
    for case in plan['cases']:
        p=ROOT/'requests'/f"{case['case']}.jsonl"
        assert sha(p)==case['sha256'],str(p)
        ids=[]
        for i,batch in enumerate(case['batches']):
            batch['shard']=i%8
            ids.extend(batch['ids'])
        assert len(ids)==len(set(ids))==case['count']
        total+=len(ids)
    assert total==20457 and len(plan['cases'])==23
    target.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan_sha=sha(target),samples=total,cases=len(plan['cases']),shards=8)))
if __name__=='__main__': main()
