"""Version reuse must preserve updates and gradients across reject/replace runs."""
import json
from pathlib import Path
import torch
from exact_komp import exact_komp
from baseline_komp import exact_komp as baseline
from validate_gpu import make,check,checks


for pattern in ('reject','replace','ties','sparse','boundary'):
    data=list(make(731,1,513,2,8,16,17,torch.float32,pattern if pattern in ('reject','replace','ties') else 'random',scalar=True))
    if pattern in ('sparse','boundary'):
        data[7].fill_(-1);data[8].zero_()
        for t in ([0,31,32,63,96,128,255,256,511,512] if pattern=='boundary' else [5,17,30,61,92,190]):
            data[7][:,t]=t+1
    xs=tuple(x.detach().requires_grad_(i<8) for i,x in enumerate(data))
    q,k,v,g,beta,k0,s0,score,init=xs
    def run(fn):
        o,(kf,sf),meta=fn(q,k,v,g,beta,(k0,s0),eviction='learned',scores=score,initial_scores=init)
        return (o,kf,sf),meta
    a,am=run(exact_komp);b,bm=run(baseline)
    assert all(torch.equal(am[key],bm[key]) for key in am)
    weights=[torch.randn_like(x)*.01 for x in a]
    ga=torch.autograd.grad(a,xs[:8],weights,allow_unused=True)
    gb=torch.autograd.grad(b,xs[:8],weights,allow_unused=True)
    for i,(x,y) in enumerate(zip((*a,*ga),(*b,*gb))):
        if x is not None and y is not None:check(f'version/{pattern}/{i}',x,y,.005,.005)
    print('PASS version',pattern,flush=True)
Path('evidence/versions.json').write_text(json.dumps({'status':'passed','check_count':len(checks),'checks':checks},indent=2))
