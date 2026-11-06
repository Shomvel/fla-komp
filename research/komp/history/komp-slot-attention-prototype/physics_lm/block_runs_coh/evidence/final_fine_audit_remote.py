from pathlib import Path
import hashlib,json,math
r=Path('/gpfs/users/linlizhi/komp-coh-trials-20261101');snap=r/'snapshot-fine';result=r/'results-fine'
manifest=json.loads((r/'fine-snapshot-sha256.json').read_text());original=json.loads((r/'snapshot-sha256.json').read_text())
for n,h in manifest.items():assert hashlib.sha256((snap/n).read_bytes()).hexdigest()==h,n
assert [n for n,h in manifest.items() if original.get(n)!=h]==['physics_lm/block_coherence.py']
screen=json.loads((r/'fine-screening.json').read_text());selection=json.loads((r/'fine-selection.json').read_text());winner=selection['selected_tag']
expected={(f'coh_fifo_{tag}',731) for tag in ('995','999')}
expected|={(f'coh_fifo_{tag}',s) for tag in screen['expand_tags'] for s in range(732,738)}
if winner!='99':expected|={(f'coh_{f}_{winner}',s) for f in ('gate','learned') for s in range(731,738)}
actual=set();ticket=json.loads((result/'preflight.json').read_text());assert len(ticket['additional_checks'])==6
for p in (result/'results').glob('*/summary.json'):
 x=json.loads(p.read_text());c=x['config'];pair=(c['architecture'],c['seed']);assert pair in expected and pair not in actual;actual.add(pair)
 assert c['source_sha256']==ticket['source_sha256'] and c['deterministic_algorithms']
 assert x['status']=='completed' and x['unique_tokens']==10001108 and x['reload_validation_ce_error']<1e-4
 assert c==json.loads((p.parent/'manifest.json').read_text())
 logs=[json.loads(s) for s in (p.parent/'metrics.jsonl').read_text().splitlines()]
 assert logs[-1]['step']==2231 and logs[-1]['unique_tokens']==10001108
 assert all(math.isfinite(t['loss']) and math.isfinite(t['grad_norm']) for t in logs if t['step']>0)
assert expected==actual
assert not list(result.rglob('*.pt'))
for w in (r/'fine-workers').iterdir():
 assert json.loads((w/'completion.json').read_text())['status']=='completed'
 assert (w/'exit-code.txt').read_text().strip()=='0'
 assert all(json.loads(s)['exit_code']==0 for s in (w/'progress.jsonl').read_text().splitlines())
assert not (r/'fine-failure.json').exists()
receipt=dict(status='passed',runs=len(actual),diagnostics=len(list(result.glob('*-diagnostic.json'))),frozen_kernel_unchanged=True,
 source_files=len(manifest),preflight_variants=6,no_checkpoints=True,deterministic_algorithms=True)
(r/'final-fine-audit.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
