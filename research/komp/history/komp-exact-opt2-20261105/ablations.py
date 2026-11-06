"""Materialize independent kernel variants and record matched correctness/timings."""
import importlib
import json
from pathlib import Path
import shutil
import torch
from compare import inputs,measure
from baseline_komp import exact_komp as baseline

root=Path(__file__).resolve().parent
variants=['solve_only','current','no_versions','no_hierarchy','dense_local','reuse_z','mixed_state']
functions={}
for name in variants:
    package='ablation_'+name;target=root/package
    shutil.copytree(root/'ablation_base_komp',target,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    p=target/'operator.py';s=p.read_text()
    if name in ('no_versions','solve_only'):s=s.replace('reuse=learned and not fp64 and m<=32','reuse=False')
    if name in ('no_hierarchy','reuse_z','solve_only'):s=s.replace('if nc>=32:','if False:')
    if name in ('dense_local','solve_only'):s=s.replace('cb.local_gradients_scan[(nc,bh,m*nv)](g,','cb.local_gradients[(nc,bh,m*nv)](gc,')
    if name=='reuse_z':s=s.replace('rec.state_scan[(bh,nv)](a,bmap,f,fv,','rec.state_scan_reuse[(bh,nv)](a,bmap,u,gc,')
    p.write_text(s)
    if name=='mixed_state':
        # Only state matmul operands; FP32 accumulators, geometry and local scan unchanged.
        for file in ('hierarchical.py','chunk_backward.py','recurrence.py'):
            p=target/file;s=p.read_text()
            import re
            s=re.sub(r"tl.dot\((tl.trans\([a-z]+\)|[a-z]+),(tl.trans\([a-z]+\)|[a-z]+),input_precision='tf32x3'\)",
                     lambda m: 'tl.dot('+m[1]+'.to(tl.bfloat16),'+m[2]+'.to(tl.bfloat16))' if file!='recurrence.py' else m[0],s)
            p.write_text(s)
    functions[name]=importlib.import_module(package).exact_komp

functions['hierarchy_only']=importlib.import_module('hierarchy_only_komp').exact_komp
rows=[]
for learned in (False,True):
    xs=inputs(8192,torch.bfloat16)
    q,k,v,g,beta,k0,s0,unused,scores=xs
    leaves=[xs[i] for i in (0,1,2,3,4,5,6,8)]
    def run(fn):
        o,(kf,sf),_=fn(q,k,v,g,beta,(k0,s0),eviction='learned' if learned else 'fifo',scores=scores)
        return o,kf,sf
    ref=run(baseline);weights=[torch.randn_like(x)*.01 for x in ref]
    rg=torch.autograd.grad(ref,leaves,weights,allow_unused=True)
    ref=tuple(x.detach() for x in ref)
    for name,fn in [('baseline',baseline),*functions.items()]:
        out=run(fn);grad=torch.autograd.grad(out,leaves,weights,allow_unused=True)
        out=tuple(x.detach() for x in out)
        errors=[]
        for i,(a,b) in enumerate(zip((*out,*grad),(*ref,*rg))):
            if a is None or b is None:continue
            rms=float((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-6))
            nmax=float((a.float()-b.float()).abs().max()/b.float().abs().max().clamp_min(1))
            errors.append({'index':i,'rms_relative':rms,'normalized_max':nmax})
        finite=all(torch.isfinite(a).all() for a in (*out,*grad) if a is not None)
        accepted=bool(finite) and all(x['rms_relative']<.01 and x['normalized_max']<.01 for x in errors)
        row={'variant':name,'learned':learned,'errors':errors,'accepted':accepted}
        if name!='mixed_state':assert accepted,row
        # Mixed precision is an explicit experiment; a failed candidate is not benchmarked or adopted.
        if accepted:
            del out,grad
            with torch.no_grad():row['forward']=measure(lambda:run(fn))
            row['forward_backward']=measure(lambda:torch.autograd.grad(run(fn),leaves,weights,allow_unused=True))
        rows.append(row)
        Path('evidence/ablations.json').write_text(json.dumps({'rows':rows},indent=2))
        print(json.dumps(row),flush=True)
