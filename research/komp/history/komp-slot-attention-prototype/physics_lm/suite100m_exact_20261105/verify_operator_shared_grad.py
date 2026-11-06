"""Correct cross-mode operator gradients with one shared cotangent per shape."""
import hashlib,json,sys
from pathlib import Path
import torch
R=Path(__file__).resolve().parent;P=R/'precision_bench_20261105';sys.path.insert(0,str(P))
import bench_projection as b

torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
report={'status':'running','source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),P/'bench_projection.py',P/'variants.json']},'scope':'Replaces ONLY operator-results.json 8k_or_1k_difference_vs_fp64 fields: old loop sampled a new upstream weight per mode. Stress/model results and timings remain valid. All KOMP modes here share identical inputs and upstream weights within each shape.','shapes':[]}
for length,heads in [(8192,2),(1024,8)]:
 xs=b.inputs(length,heads)
 with torch.no_grad():out=b.run('fp64',xs)
 torch.manual_seed(1981+length)
 weights=[torch.randn_like(x)*.01 for x in out]
 weight_sha=[hashlib.sha256(w.detach().float().cpu().numpy().tobytes()).hexdigest() for w in weights]
 ref=None
 for mode in b.MODES:
  outputs=b.run(mode,xs);grads=torch.autograd.grad(outputs,xs[:7],weights)
  values=tuple(outputs)+tuple(grads)
  if mode=='fp64':ref=[x.detach().clone() for x in values]
  checks=[b.metric(name,a,z,True) for name,a,z in zip(['output','keys','state','dq','dk','dv','dg','dbeta','dkeys0','dstate0'],values,ref)]
  row={'T':length,'H':heads,'mode':mode,'weights_sha256':weight_sha,'finite':all(c['finite'] for c in checks),'checks':checks,'verdict':'PASSED' if all(c['passed'] for c in checks) else 'FAILED'}
  report['shapes'].append(row);print('SHARED_COTANGENT',json.dumps(row),flush=True)
  del outputs,grads,values
 del xs,weights,out,ref
 torch.cuda.empty_cache()
report['status']='completed'
(R/'evidence/operator-shared-grad-correction.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
