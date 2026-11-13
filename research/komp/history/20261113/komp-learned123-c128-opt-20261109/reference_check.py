"""Final C128 candidate against the existing independent FP64 algebra and ranking checks."""
import argparse,json,os,importlib.util,sys,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--snapshot',required=True);ap.add_argument('--tag',required=True);args=ap.parse_args()
os.environ['KOMP_SNAPSHOT']=str(ROOT/args.snapshot)
import operator_checks as checks
compare,ranking_checks=checks.compare,checks.ranking_checks
actual=checks.block_komp_fast
spec=importlib.util.spec_from_file_location('fla.ops.komp.block_fast_paired_reference',ROOT/'snapshot-baseline/fla/ops/komp/block_fast.py')
baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
torch.set_num_threads(2);torch.manual_seed(1234)
cases=[]
for dtype in [torch.float32,torch.bfloat16]:
 for near in [False,True]:
  t=257
  xs=[torch.randn(1,t,2,64,device='cuda',dtype=dtype) for _ in range(3)]
  xs+=[-torch.rand(1,t,2,32,device='cuda')*.02,torch.rand(1,t,2,device='cuda')*.3,
       torch.randn(1,2,32,64,device='cuda'),torch.randn(1,2,32,64,device='cuda')*.1]
  if near:
   # Keeps the original projection precision test active; no rank-1 inverse approximation.
   center=torch.randn(1,1,2,64,device='cuda')
   xs[1]=(center+.02*xs[1].float()).to(dtype);xs[5]=center.transpose(1,2)+.02*xs[5]
  scores=torch.randn(1,t,2,device='cuda');initial=torch.randn(1,2,32,device='cuda')
  rng=torch.cuda.get_rng_state();row=compare(xs,scores,initial,f'{dtype}-near{near}')
  torch.cuda.set_rng_state(rng);checks.block_komp_fast=baseline.block_komp_fast
  ref=compare(xs,scores,initial,f'{dtype}-near{near}-baseline');checks.block_komp_fast=actual
  record=dict(candidate=row,baseline=ref)
  print(json.dumps(record),flush=True);cases.append(record)
routing=ranking_checks()
result=dict(passed=all(c['candidate']['passed'] for c in cases) and all(v['passed'] for v in routing),cases=cases,ranking=routing,reference='existing independent FP64 block_flex algebra',
            tolerance='unchanged original .003 FP32/.025 BF16 or maxabs <=3e-4')
result['candidate_sha256']=hashlib.sha256((ROOT/args.snapshot/'fla/ops/komp/block_fast.py').read_bytes()).hexdigest()
(ROOT/'evidence'/f'{args.tag}.json').write_text(json.dumps(result,indent=2)+'\n')

assert result['passed'], [(c['candidate']['case'],c['baseline']['passed']) for c in cases if not c['candidate']['passed']]
