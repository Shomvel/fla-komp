"""Stage the exact validated candidate without touching the running parent."""
import hashlib,json,shutil,statistics,argparse
from pathlib import Path
r=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--snapshot',default='snapshot-final-v2');ap.add_argument('--prefix',default='final-v2');args=ap.parse_args()
out=r.with_name('komp-learned123-1b-50b-20261109-opt1')
assert not (out/'source-manifest.json').exists(),'Never mutate frozen training source'
required=[f'{args.prefix}-{name}.json' for name in ['gdn-differential','model-adam','reference-acceptance','model-abba']]
assert json.loads((r/'evidence/cache-loss-lifetime.json').read_text())['passed']
evidence={n:json.loads((r/'evidence'/n).read_text()) for n in required}
for n in required[:3]:assert evidence[n]['passed'],n
measurement=evidence[required[3]]
operator=r/args.snapshot/'fla/ops/komp/block_fast.py'
operator_sha=hashlib.sha256(operator.read_bytes()).hexdigest()
assert measurement['operator_sha']==operator_sha
assert evidence[required[0]]['candidate_sha']==operator_sha
assert evidence[required[1]]['optimizer_step']['passed']
assert evidence[required[1]]['candidate_source_files']==measurement['source_files']
assert evidence[required[1]]['model_adapter_sha256']==hashlib.sha256((r/'models.py').read_bytes()).hexdigest()
assert evidence[required[2]]['candidate_sha256']==operator_sha
ref=statistics.mean(x['median_seconds'] for x in measurement['abba'] if not x['candidate'])
new=statistics.mean(x['median_seconds'] for x in measurement['abba'] if x['candidate'])
assert new<ref*.9,(ref,new,'No material step benefit')
assert measurement['peak_bytes']<76_000_000_000,'Leave formal DDP memory headroom'
# Re-hash every timed snapshot file; numerical validation used the same candidate.
for n,h in measurement['source_files'].items():assert hashlib.sha256((r/args.snapshot/n).read_bytes()).hexdigest()==h,n
shutil.copytree(r/args.snapshot,out/'snapshot',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(r/'models.py',out/'models.py')
cache=measurement['cache_ut_layers']
script=out/'run_train.sh';s=script.read_text();assert 'KOMP_UT_CACHE_LAYERS' not in s
s=s.replace('export CUBLAS_WORKSPACE_CONFIG=:4096:8',f'export CUBLAS_WORKSPACE_CONFIG=:4096:8\nexport KOMP_UT_CACHE_LAYERS={cache}\nexport KOMP_SNAPSHOT="$TASK_ROOT/snapshot"')
script.write_text(s)
parent=r.with_name('komp-learned123-1b-50b-20261109-mb16-direct')
manifest=json.loads((parent/'source-manifest.json').read_text())['files']
files=list((out/'snapshot').rglob('*.py'))+[out/'models.py']
changed={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if manifest.get(str(p.relative_to(out)))!=hashlib.sha256(p.read_bytes()).hexdigest()}
policy=dict(verified=True,verification_scope='single_H100_fixed_trained_checkpoint_inputs; formal_32GPU_acceptance_recorded_separately',description='C128 learned forward/backward optimization: direct slot reduction, measured warps, rounded SwiGLU and exact graph-local recomputation cache; retain IEEE UT and original PyTorch RMSNorm; checkpoint cache storage releases automatically during backward',
 operator_sha256=operator_sha,candidate_files_sha256=changed,loss_tolerance=.001,numerical_policy='upstream_gdn_bf16_test_chunk',
 verification_evidence_sha256={n:hashlib.sha256((r/'evidence'/n).read_bytes()).hexdigest() for n in required},
 cache_ut_layers=cache,single_h100_reference_seconds=ref,single_h100_candidate_seconds=new,
 single_h100_speedup=ref/new,peak_bytes=measurement['peak_bytes'],formal_32gpu_measurement='pending',
 world_size=32,microbatch=16,accumulation=1,global_sequences=512,target_supervised_tokens=50000000000)
(out/'configs/optimization.json').write_text(json.dumps(policy,indent=2)+'\n')
for n in required+[f'{args.prefix}-reference.json','cache-loss-lifetime.json']:shutil.copy2(r/'evidence'/n,out/'evidence'/n)
print(json.dumps(policy,indent=2))
