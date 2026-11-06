"""CPU differential acceptance against the frozen KOMP C=1 reference."""
import importlib.util
import json
from pathlib import Path
import time
import torch
from c1_komp import parallel, sequential

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/Users/shom/Downloads/komp-accel-scratch/frozen-20261101-r3/komp_slot/block_reference.py')
spec = importlib.util.spec_from_file_location('frozen_reference', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(1)
checks = []


def compare(name, actual, expected, tolerance):
    difference = (actual-expected).detach()
    normalized = difference.abs().max().item() / max(1., expected.detach().abs().max().item())
    row = dict(name=name, max_abs=difference.abs().max().item(), normalized_max=normalized, tolerance=tolerance)
    checks.append(row)
    if not torch.isfinite(actual).all() or normalized > tolerance:
        raise AssertionError(row)


def inputs(seed, length, m, d, dv, dtype, near_duplicate=False, strong_decay=False, scalar=False):
    rng = torch.Generator().manual_seed(seed)
    rand = lambda *shape: torch.randn(*shape, generator=rng, dtype=dtype)
    q, k, v = rand(1,length,2,d), rand(1,length,2,d), rand(1,length,2,dv)
    keys, state = rand(1,2,m,d), .2*rand(1,2,m,dv)
    if near_duplicate:
        center = rand(1,1,2,d)
        k = center + .002*k
        keys = center.transpose(1,2) + .002*keys
    g = -.01-.2*torch.rand(1,length,2,m, generator=rng, dtype=dtype)
    if scalar:
        g = g[..., 0]
    if strong_decay:
        g = 35*g
    beta = torch.sigmoid(rand(1,length,2))
    return q,k,v,g,beta,keys,state


def call(fn, x, **kw):
    return fn(*x[:5], (x[5],x[6]), **kw)


def frozen(x):
    return module.block_komp_reference(*x[:5], (x[5],x[6]), chunk_size=1)


def chronological(state, length):
    m = state.shape[-2]
    order = (torch.arange(m, device=state.device)+length) % m
    return state[..., order, :]


def forward_cases():
    cases = [(1,1,4,6,3,False,False,False), (2,7,4,6,3,False,False,False),
             (3,19,8,12,5,False,False,False), (4,33,8,12,5,True,False,False),
             (5,65,32,64,64,False,False,False), (6,33,8,12,5,False,True,False),
             (7,19,8,12,5,False,False,True)]
    for dtype in [torch.float64, torch.float32]:
        for seed,length,m,d,dv,near,strong,scalar in cases:
            x = inputs(seed,length,m,d,dv,dtype,near,strong,scalar)
            y0,(k0,s0) = frozen(x)
            tolerance = 2e-9 if dtype==torch.float64 else 3e-3
            label = f'{dtype}/seed{seed}/T{length}/M{m}'
            yr,(kr,sr) = call(sequential,x)
            compare(label+'/literal/output',yr,y0,tolerance)
            compare(label+'/literal/state',chronological(sr,length),s0,tolerance)
            for c in [1,4,16,32,128]:
                y,(keys,state) = call(parallel,x,compute_chunk=c)
                compare(label+f'/C{c}/output',y,y0,tolerance)
                compare(label+f'/C{c}/state',chronological(state,length),s0,tolerance)
                compare(label+f'/C{c}/keys',chronological(keys,length),k0,tolerance)


def gradient_cases():
    for seed,near in [(8,False),(9,True)]:
        source = inputs(seed,11,4,6,3,torch.float64,near)
        rng = torch.Generator().manual_seed(20261105)
        wo = torch.randn(1,11,2,3,generator=rng,dtype=torch.float64)
        ws = torch.randn(1,2,4,3,generator=rng,dtype=torch.float64)
        for c in [1,4,16]:
            gradients=[]
            for mode in ['frozen','parallel']:
                x = tuple(a.clone().requires_grad_() for a in source)
                y,(_,state) = frozen(x) if mode=='frozen' else call(parallel,x,compute_chunk=c)
                if mode=='parallel':state=chronological(state,11)
                loss=(y*wo).sum()+(state*ws).sum()
                gradients.append(torch.autograd.grad(loss,x))
            for name,a,b in zip(['q','k','v','log_decay','beta','initial_keys','initial_coefficients'],gradients[1],gradients[0]):
                compare(f'gradient/seed{seed}/C{c}/{name}',a,b,2e-8)


def causal_and_streaming():
    x = inputs(10,29,8,12,5,torch.float64)
    full,(keys,state) = call(parallel,x,compute_chunk=16)
    cut=13
    altered=[]
    for a in x[:5]:
        b=a.clone(); b[:,cut:]=b[:,cut:]+.07
        altered.append(b)
    changed,_=call(parallel,tuple(altered)+x[5:],compute_chunk=16)
    compare('causality/prefix',changed[:,:cut],full[:,:cut],0.)
    first=tuple(a[:,:cut] for a in x[:5])+x[5:]
    y1,(k1,s1)=call(parallel,first,compute_chunk=4)
    rest=tuple(a[:,cut:] for a in x[:5])+(k1,s1)
    y2,(k2,s2)=call(parallel,rest,compute_chunk=7,start_position=cut)
    compare('streaming/output',torch.cat((y1,y2),dim=1),full,2e-9)
    compare('streaming/keys',k2,keys,2e-9)
    compare('streaming/state',s2,state,2e-9)


if __name__ == '__main__':
    started=time.perf_counter()
    try:
        forward_cases()
        gradient_cases()
        causal_and_streaming()
    except Exception as exc:
        (ROOT/'validation-failure.json').write_text(json.dumps({'error':str(exc),'checks':checks},indent=2))
        raise
    result={'status':'passed','device':'CPU','torch':torch.__version__,
            'reference':str(SOURCE),'checks':checks,'check_count':len(checks),
            'elapsed_seconds':time.perf_counter()-started,
            'max_normalized_error':max(x['normalized_max'] for x in checks)}
    (ROOT/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
