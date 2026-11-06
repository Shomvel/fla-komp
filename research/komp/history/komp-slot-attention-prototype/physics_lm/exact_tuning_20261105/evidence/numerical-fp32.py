"""FP64 output/state/all-input-gradient gate for the actual tuning range."""
import argparse
import json
import math
import time
import traceback
import tune
import torch
from komp_slot.exact_komp import exact_komp
from komp_slot.block_reference import block_komp_token_reference


def check(name, actual, expected, low, checks):
    a,b = actual.detach().double(),expected.detach().double()
    delta = a-b
    rms = float(delta.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-6))
    maximum = float(delta.abs().max())/max(1.,float(b.abs().max()))
    rt,mt = (.025,.03) if low else (.005,.005)
    row = dict(name=name,rms_relative=rms,normalized_max=maximum,rms_tolerance=rt,max_tolerance=mt)
    checks.append(row)
    assert torch.isfinite(a).all() and rms <= rt and maximum <= mt, row


def run_case(candidate, pattern, dtype, checks):
    torch.manual_seed(917)
    b,t,h,m,k,v = 1,65,1,32,64,64
    rand = lambda *shape: torch.randn(*shape,device='cuda',dtype=dtype)
    q,keys,values = rand(b,t,h,k),rand(b,t,h,k),rand(b,t,h,v)
    initial = rand(b,h,m,k)
    if pattern in ('near','duplicate'):
        center = rand(b,1,h,k)
        noise = .002 if pattern == 'near' else 0.
        keys = (center+noise*keys).to(dtype)
        initial = (center.transpose(1,2)+noise*initial).to(dtype)
    g = torch.full((b,t,h,m), math.log(candidate['alpha0']),device='cuda')
    if pattern == 'strong':
        g = -torch.rand_like(g)*3.-.01
    beta = torch.full((b,t,h),candidate['beta0'],device='cuda')
    state = rand(b,h,m,v).float()*.1
    data = [x.detach().requires_grad_() for x in (q,keys,values,g,beta,initial,state)]
    gold = [x.detach().double().requires_grad_() for x in data]
    kwargs = dict(temperature=candidate['temperature'],ridge=candidate['ridge'])
    name = f"{candidate['name']}/{pattern}/{dtype}"
    print('START',name,flush=True)
    out,(fk,fs),_ = exact_komp(*data[:5],tuple(data[5:]),eviction='fifo',compute_chunk=16,
                              projection_dtype='float32',**kwargs)
    ref,(rk,rs) = block_komp_token_reference(*gold[:5],tuple(gold[5:]),chunk_size=1,evict='fifo',**kwargs)
    actual,expected = (out,fk,fs),(ref,rk,rs)
    weights = [torch.randn_like(x.float()) for x in actual]
    grad = torch.autograd.grad(actual,data,weights)
    rgrad = torch.autograd.grad(expected,gold,[w.double() for w in weights])
    for label,a,e in zip(('output','keys','state','dq','dk','dv','dg','dbeta','dkeys0','dstate0'),
                         (*actual,*grad),(*expected,*rgrad)):
        check(name+'/'+label,a,e,dtype==torch.bfloat16,checks)
    print('PASS',name,flush=True)


def main(task):
    dest = tune.HERE/'evidence'
    dest.mkdir(exist_ok=True)
    checks = []
    start = time.monotonic()
    report = dict(sources=tune.sources(),checks=checks,gpu=torch.cuda.get_device_name(),
                  torch=str(torch.__version__),scope='M32 K64 T65, forward/final state/all seven input gradients')
    try:
        plan = json.loads((tune.HERE/'initial-screen.json').read_text())
        for candidate in plan['screen']:
            for dtype in (torch.float32,torch.bfloat16):
                for pattern in ('random','near','duplicate','strong'):
                    run_case(candidate,pattern,dtype,checks)
        report.update(status='passed',check_count=len(checks),seconds=time.monotonic()-start)
        tune.d.base.atomic_json(dest/f'numerical-{task}.json',report)
    except BaseException:
        report.update(status='failed',error=traceback.format_exc(),seconds=time.monotonic()-start)
        tune.d.base.atomic_json(dest/f'numerical-{task}-failure.json',report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',required=True,choices=['depo2','brevo2'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    main(args.task)
