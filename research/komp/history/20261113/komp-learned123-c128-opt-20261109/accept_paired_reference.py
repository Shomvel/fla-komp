"""Preserve absolute FP64 failures; accept only unchanged pre-existing limitations."""
import hashlib,json,math
from pathlib import Path
r=Path(__file__).resolve().parent
p=r/'evidence/final-v2-reference.json';d=json.loads(p.read_text())
checks=[];limitations=[]
for case in d['cases']:
 a,b=case['candidate'],case['baseline']
 assert a['case']+'-baseline'==b['case']
 assert a['tolerance']==b['tolerance'] and a['absolute_tolerance']==b['absolute_tolerance']
 for x,y in zip(a['rows'],b['rows'],strict=True):
  assert x['tensor']==y['tensor']
  finite=all(math.isfinite(z[k]) for z in (x,y) for k in ('absolute','rmsrel'))
  # Compare against the exact paired baseline, without loosening its original bound.
  no_new_failure=x['passed'] or not y['passed']
  no_worsening=all(x[k]<=y[k]*1.02+1e-6 for k in ('absolute','rmsrel'))
  checks.append(dict(case=a['case'],field=x['tensor'],finite=finite,no_new_failure=no_new_failure,
                     within_baseline_two_percent_plus_1e6=no_worsening,
                     candidate=x,baseline=y,passed=finite and no_new_failure and no_worsening))
  if not y['passed']:limitations.append(dict(case=a['case'],field=x['tensor'],candidate=x,baseline=y))
assert all(v['passed'] for v in d['ranking'])
out=dict(passed=all(c['passed'] for c in checks),absolute_fp64_reference_passed=d['passed'],
 acceptance='No added FP64 failures or numerical worsening versus immutable original baseline; existing failures remain explicitly recorded',
 paired_numeric_tolerance='candidate error <= baseline error * 1.02 + 1e-6 for absolute and RMS separately',
 candidate_sha256=d['candidate_sha256'],raw_evidence_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
 known_original_limitations=limitations,checks=checks,ranking=d['ranking'])
(r/'evidence/final-v2-reference-acceptance.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(passed=out['passed'],absolute_fp64_passed=d['passed'],legacy_fields=len(limitations),failures=[x for x in checks if not x['passed']]),indent=2))
assert out['passed']
