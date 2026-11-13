"""Freeze only the user's completed, exact-50B training checkpoint."""
import argparse, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def prepare(runroot):
    run=runroot/'runs/block'
    final=json.loads((run/'complete.json').read_text())
    latest=json.loads((run/'latest.json').read_text())
    assert final['status']=='completed'
    assert final['supervised_tokens']==latest['tokens']==50000000000
    assert final['step']==latest['step']==47684
    ck=Path(latest['path']);complete=json.loads((ck/'complete.json').read_text())
    assert ck.parent==run and complete['step']==47684 and complete['tokens']==50000000000
    assert complete['world_size']==32
    source_sha=sha(runroot/'source-manifest.json');data_sha=sha(runroot/'configs/data-manifest.json')
    assert final['source_sha']==complete['source_sha']==source_sha
    assert final['data_sha']==complete['data_sha']==data_sha
    for name,expected in json.loads((runroot/'source-manifest.json').read_text())['files'].items():
        assert sha(runroot/name)==expected,name
    ranks={}
    for i in range(32):
        p=ck/f'rank{i}.pt';assert p.stat().st_size>0
        ranks[p.name]=dict(bytes=p.stat().st_size,sha256=sha(p))
    assert (ck/'model.pt').stat().st_size>0
    for i in range(4):assert (runroot/f'nodes/{i}/exit.txt').read_text().strip()=='0'
    value=dict(runroot=str(runroot),checkpoint=str(ck/'model.pt'),checkpoint_sha=sha(ck/'model.pt'),
               checkpoint_bytes=(ck/'model.pt').stat().st_size,step=47684,tokens=50000000000,
               partial=False,source_sha=source_sha,data_sha=data_sha,rank_files=ranks,training_final=final)
    p=ROOT/'evidence/checkpoint.json'
    if p.exists():assert json.loads(p.read_text())==value,'Pinned checkpoint changed'
    else:p.write_text(json.dumps(value,indent=2)+'\n')
    requests=ROOT/'requests'
    if not requests.exists():requests.symlink_to('/gpfs/users/linlizhi/komp-flame-downstream-eval-20261107/requests',target_is_directory=True)
    from prepare_plan import main
    if not (ROOT/'sharded/plan.json').exists():main()
    files=list(ROOT.glob('*.py'))+[ROOT/'job.yaml',ROOT/'sharded/plan.json']
    files += [ROOT/'evidence'/f for f in ['checkpoint.json','requests-manifest.json','generation-contract-audit.json','inference-parity-static.json']]
    (ROOT/'source-manifest.json').write_text(json.dumps({'files':{str(p.relative_to(ROOT)):sha(p) for p in files}},indent=2)+'\n')
    return value
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runroot',type=Path);a=p.parse_args()
    v=prepare(a.runroot);print(json.dumps({k:v[k] for k in ['checkpoint','checkpoint_sha','step','tokens','partial']}))
