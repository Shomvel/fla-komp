"""Audit archived result contracts and frozen source without running training."""
from pathlib import Path
import hashlib,json,math
R=Path(__file__).resolve().parent.parent
S=R/'evidence/snapshot'
frozen=json.loads((R/'evidence/frozen-verification.json').read_text())
for n,d in frozen.items():assert hashlib.sha256((S/n).read_bytes()).hexdigest()==d,n
manifest=json.loads((R/'evidence/snapshot-sha256.json').read_text())
for n,d in manifest.items():assert hashlib.sha256((S/n).read_bytes()).hexdigest()==d,n
plan=json.loads((R/'evidence/plan.json').read_text())
expected={(a,s) for a in plan['main']+plan['optional'] for s in plan['seeds']}
actual=set();common={};datasets=set()
for path in sorted((R/'results').glob('*/summary.json')):
 s=json.loads(path.read_text());c=s['config'];key=(c['architecture'],c['seed'])
 assert key in expected and key not in actual;actual.add(key)
 assert s['status']=='completed' and s['unique_tokens']==10001108 and s['reload_validation_ce_error']<1e-4
 assert c==json.loads((path.parent/'manifest.json').read_text())
 assert c['steps']==2231 and c['batch']==64 and c['deterministic'] and not c['tf32']
 assert all(c['ablation_source_sha256'][n]==d for n,d in manifest.items() if n in c['ablation_source_sha256'])
 common.setdefault(c['seed'],[]).append(c['shared_init_sha256']);datasets.add(json.dumps(c['data'],sort_keys=True))
 metrics=[json.loads(line) for line in (path.parent/'metrics.jsonl').read_text().splitlines()]
 assert metrics[-1]['step']==2231 and metrics[-1]['unique_tokens']==10001108
 assert all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in metrics if row['step'])
 assert all(s['generation'][t]['n']==128 for t in ('depo','brevo'))
 timing=s['timing'];parts=sum(timing[k] for k in ('train_loop_and_bookkeeping_seconds','evaluation_seconds','generation_seconds'))
 assert abs(parts-s['elapsed_seconds'])<1e-6 and timing['train_loop_and_bookkeeping_seconds']>0
assert actual==expected,sorted(expected-actual)
assert len(datasets)==1
for seed,rows in common.items():assert all(r==rows[0] for r in rows)
assert not list(R.rglob('*.pt'))
receipt=dict(status='passed',runs=len(actual),frozen_files=len(frozen),same_all_shared_initialization=True,same_data=True,
 no_checkpoints=True,timing_adds_up=True,tokens_per_run=10001108,steps_per_run=2231)
(R/'evidence/local-result-audit.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
