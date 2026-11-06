from pathlib import Path
import hashlib,json,math
root=Path('/gpfs/users/linlizhi/komp-coh-trials-20261101');snap=root/'snapshot';results=root/'results'
manifest=json.loads((root/'snapshot-sha256.json').read_text())
for name,digest in manifest.items():assert hashlib.sha256((snap/name).read_bytes()).hexdigest()==digest,name
selection=json.loads((root/'selection.json').read_text());tag=selection['selected_tag']
arms=['coh_fifo_off','coh_fifo_99','coh_fifo_95','coh_fifo_80','coh_gate_off',f'coh_gate_{tag}','coh_learned_off',f'coh_learned_{tag}']
expected={(a,s) for a in arms for s in range(731,738)};actual=set()
for path in (results/'results').glob('*/summary.json'):
 r=json.loads(path.read_text());cfg=r['config'];pair=(cfg['architecture'],cfg['seed'])
 assert pair in expected and pair not in actual;actual.add(pair)
 assert r['status']=='completed' and r['unique_tokens']==10001108 and r['reload_validation_ce_error']<1e-4
 assert cfg==json.loads((path.parent/'manifest.json').read_text())
 assert cfg['deterministic_algorithms'] is True
 logs=[json.loads(s) for s in (path.parent/'metrics.jsonl').read_text().splitlines()]
 assert logs[-1]['step']==2231 and logs[-1]['unique_tokens']==10001108
 assert all(math.isfinite(x['loss']) and math.isfinite(x['grad_norm']) for x in logs if x['step']>0)
 assert (results/f'{pair[0]}-seed{pair[1]}-coherence.json').exists()
assert actual==expected,expected-actual
assert not list(results.rglob('*.pt'))
assert len(list(results.glob('*-diagnostic.json')))==8
for i in range(1,17):
 w=root/'workers'/f'r{i}'
 assert json.loads((w/'completion.json').read_text())['status']=='completed'
 assert (w/'exit-code.txt').read_text().strip()=='0'
 assert all(json.loads(line)['exit_code']==0 for line in (w/'progress.jsonl').read_text().splitlines())
assert not (root/'failure.json').exists()
receipt=dict(status='passed',runs=56,diagnostics=8,source_files=len(manifest),all_workers_completed=True,no_checkpoints=True,
             frozen_source_unchanged=True,deterministic_algorithms=True,tokens_per_run=10001108,steps_per_run=2231)
(root/'final-source-audit.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
