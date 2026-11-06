from pathlib import Path
import hashlib,json,math
root=Path('/gpfs/users/linlizhi/komp-proj-swa-20261101');snap=root/'snapshot';results=root/'results'
manifest=json.loads((root/'snapshot-sha256.json').read_text())
manifest.update(json.loads((root/'fla-provenance.json').read_text()))
for name,digest in manifest.items():assert hashlib.sha256((snap/name).read_bytes()).hexdigest()==digest,name
plan=json.loads((root/'plan.json').read_text());arms=plan['main']+plan['optional']
expected={(a,s) for a in arms for s in range(731,738)};actual=set()
for path in (results/'results').glob('*/summary.json'):
 r=json.loads(path.read_text());cfg=r['config'];pair=(cfg['architecture'],cfg['seed'])
 assert pair in expected and pair not in actual;actual.add(pair)
 assert r['status']=='completed' and r['unique_tokens']==10001108 and r['reload_validation_ce_error']<1e-4
 assert cfg==json.loads((path.parent/'manifest.json').read_text())
 assert cfg['deterministic'] is True
 logs=[json.loads(s) for s in (path.parent/'metrics.jsonl').read_text().splitlines()]
 assert logs[-1]['step']==2231 and logs[-1]['unique_tokens']==10001108
 assert all(math.isfinite(x['loss']) and math.isfinite(x['grad_norm']) for x in logs if x['step']>0)
 assert all(v>=0 for k,v in r['timing'].items() if k.endswith('_seconds'))
assert actual==expected,expected-actual
assert not list(results.rglob('*.pt'))
for i in range(1,17):
 w=root/'workers'/f'r{i}'
 assert json.loads((w/'completion.json').read_text())['status']=='completed'
 assert (w/'exit-code.txt').read_text().strip()=='0'
 assert all(json.loads(line)['exit_code']==0 for line in (w/'progress.jsonl').read_text().splitlines())
assert not (root/'failure.json').exists()
receipt=dict(status='passed',runs=len(expected),source_files=len(manifest),all_workers_completed=True,no_checkpoints=True,
             frozen_source_unchanged=True,deterministic_algorithms=True,tokens_per_run=10001108,steps_per_run=2231)
(root/'final-source-audit.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
