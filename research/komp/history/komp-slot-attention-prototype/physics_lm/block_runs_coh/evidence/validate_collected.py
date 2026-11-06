from pathlib import Path
import json,hashlib,math
root=Path(__file__).resolve().parent.parent;e=root/'evidence'
ticket=json.loads((root/'preflight.json').read_text())
assert ticket['status']=='passed' and len(ticket['variants'])==len(ticket['additional_checks'])==12
assert all(x['future_gradient']==0 and x['train_only_masked_statistics'] and x['diagnostic_trace_bitwise_neutral'] for x in ticket['additional_checks'])
for path,digest in json.loads((e/'prior-artifact-sha256.json').read_text()).items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
frozen=json.loads((e/'frozen-verification.json').read_text())
for name,digest in frozen.items():assert hashlib.sha256((e/'snapshot'/name).read_bytes()).hexdigest()==digest,name
sources=json.loads((e/'snapshot-sha256.json').read_text())
for name,digest in ticket['source_sha256'].items():assert sources[name]==digest,name
stats=json.loads((root/'original-statistics.json').read_text());assert stats['complete'] and len(stats['arms'])==8
assert all(x['n']==7 for arm in stats['arms'].values() for x in arm.values())
assert len(list((root/'results').glob('*/summary.json')))==56
assert len(list(root.glob('*-coherence.json')))==56 and len(list(root.glob('*-diagnostic.json')))==8
assert not list(root.rglob('*.pt'))
gen=json.loads((e/'independent-generation-audit.json').read_text())
assert gen['status']=='passed' and gen['completed_runs']==56 and gen['checked_generations']==14336
receipt=dict(status='passed',formal_runs=56,arms=8,seeds_per_arm=7,diagnostics=8,generations=14336,
 preflight_variants=12,frozen_files_verified=len(frozen),prior_artifacts_unchanged=True,no_checkpoints=True)
(e/'result-audit.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
