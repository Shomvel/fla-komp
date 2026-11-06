"""GPU acceptance: independent references, gradients, streaming, training, and timing."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from komp_slot import projected_slot_attention
from komp_slot.reference import prepare_geometry, projection_oracle, recurrent_reference

ROOT = Path(__file__).parent
REPORT = {'checks': []}


def check(name, got, expected, atol=3e-5, rtol=3e-4):
    torch.testing.assert_close(got, expected.to(got), atol=atol, rtol=rtol)
    if not bool(torch.isfinite(got).all()):
        raise AssertionError(name + ' is nonfinite')
    REPORT['checks'].append({'name': name, 'max_abs_error': float((got-expected.to(got)).abs().max()),
                             'atol': atol, 'rtol': rtol})


def clone_grad(xs):
    return [x.detach().clone().requires_grad_() for x in xs]


def core_inputs(shape, device, seed=0):
    b, t, h, m, v = shape
    torch.manual_seed(seed)
    rand = lambda *s: torch.randn(*s, device=device)
    hist = torch.softmax(rand(b, t, h, m), -1)
    proj = rand(b, t, h, m)*0.15
    read = torch.softmax(rand(b, t, h, m), -1)
    values = rand(b, t, h, v)*0.3
    alpha = torch.rand(b, t, h, device=device)*0.15+0.8
    beta = torch.rand(b, t, h, device=device)*0.6
    state = rand(b, h, m, v)*0.1
    slots = torch.randint(m, (b, t, h), device=device)
    return [hist, proj, read, values, alpha, beta, state], slots


def call_core(fn, xs, slots):
    return fn(*xs[:6], slots, xs[6])


def validate_core():
    from komp_slot.kernels import fused_recurrent_slot
    cases = [(1, 1, 1, 1, 1), (2, 7, 3, 5, 37), (1, 33, 2, 16, 64),
             (2, 65, 2, 32, 17), (1, 129, 1, 64, 65)]
    for n, shape in enumerate(cases):
        xs, slots = core_inputs(shape, 'cuda', n)
        if n == 0:
            xs[4].zero_()
            xs[5].fill_(1)
        if n == 1:
            xs[4][:, 0].zero_()
            xs[4][:, -1].fill_(1)
            xs[5][:, 0].zero_()
            slots.fill_(shape[3]-1)
        candidate = clone_grad(xs)
        reference = [x.double().requires_grad_() for x in xs]
        out, final = call_core(fused_recurrent_slot, candidate, slots)
        ro, rf = call_core(recurrent_reference, reference, slots)
        check(f'core{n}/output', out, ro)
        check(f'core{n}/state', final, rf)
        go, gf = torch.randn_like(out), torch.randn_like(final)
        cg = torch.autograd.grad((out, final), candidate, (go, gf))
        rg = torch.autograd.grad((ro, rf), reference, (go.double(), gf.double()))
        for name, g, expected in zip(('h', 'p', 'read', 'v', 'alpha', 'beta', 'initial'), cg, rg):
            check(f'core{n}/grad_{name}', g, expected, atol=6e-5, rtol=6e-4)
        print('PASS core', shape, flush=True)

    # final-state-only and output-only losses exercise absent cotangents
    xs, slots = core_inputs((1, 5, 2, 8, 33), 'cuda', 9)
    for which in (0, 1):
        a, b = clone_grad(xs), clone_grad(xs)
        co = call_core(fused_recurrent_slot, a, slots)[which]
        ro = call_core(recurrent_reference, b, slots)[which]
        ga = torch.autograd.grad(co.square().sum(), a)
        gb = torch.autograd.grad(ro.square().sum(), b, allow_unused=True)
        for i, (got, ref) in enumerate(zip(ga, gb)):
            check(f'cotangent{which}/grad{i}', got, torch.zeros_like(got) if ref is None else ref)

    # noncontiguous public core inputs
    a = [torch.stack((x, x), -1)[..., 0] for x in xs]
    out, final = call_core(fused_recurrent_slot, a, slots)
    ro, rf = call_core(recurrent_reference, xs, slots)
    check('noncontiguous/output', out, ro)
    check('noncontiguous/state', final, rf)


def model_inputs(device, dtype=torch.float32, t=9, m=5):
    torch.manual_seed(20261031)
    b, h, dk, dv = 2, 2, 7, 11
    xs = [torch.randn(b, t, h, dk, device=device, dtype=dtype) for _ in range(2)]
    xs += [torch.randn(b, t, h, dv, device=device, dtype=dtype)*0.2,
           torch.full((b,t,h), .96, device=device, dtype=dtype),
           torch.full((b,t,h), .2, device=device, dtype=dtype),
           torch.randn(b,h,m,dk, device=device, dtype=dtype),
           torch.randn(b,h,m,dv, device=device, dtype=dtype)*0.05]
    slots = torch.arange(t, device=device)[None,:,None].expand(b,t,h).remainder(m).contiguous()
    return xs, slots


def call_model(xs, slots, backend, ridge=1e-4):
    return projected_slot_attention(*xs[:5], slots, (xs[5], xs[6]), ridge=ridge, backend=backend)


def validate_model(device):
    xs, slots = model_inputs(device, torch.float64, t=6)
    out, (keys, vals) = call_model(xs, slots, 'torch', ridge=0)
    ro, rk, rv = projection_oracle(*xs[:5], slots, xs[5], xs[6])
    check('full_projection/output', out, ro, 1e-10, 1e-9)
    check('full_projection/keys', keys, rk, 1e-12, 1e-12)
    check('full_projection/values', vals, rv, 1e-10, 1e-9)

    # finite-difference check verifies the differentiable geometry independently
    small = [x[:1,:2,:1].clone() for x in xs[:5]] + [x[:1,:1,:3].clone() for x in xs[5:]]
    small = clone_grad(small)
    small_slots = slots[:1,:2,:1].clone()
    def gc(*args):
        o, (kk, vv) = call_model(args, small_slots, 'torch', ridge=1e-3)
        return o, kk, vv
    assert torch.autograd.gradcheck(gc, tuple(small), eps=1e-6, atol=2e-5, rtol=2e-4, fast_mode=True)
    REPORT['checks'].append({'name': 'geometry_gradcheck', 'passed': True})

    if device == 'cpu':
        return
    for dtype in (torch.float32, torch.bfloat16):
        xs, slots = model_inputs(device, dtype)
        a, b = clone_grad(xs), clone_grad(xs)
        out, state = call_model(a, slots, 'triton')
        ro, rs = call_model(b, slots, 'torch')
        tag = str(dtype).split('.')[-1]
        for name, got, ref in zip(('output','keys','values'), (out,*state), (ro,*rs)):
            check(f'end_to_end_{tag}/{name}', got, ref)
        weights = [torch.randn_like(x) for x in (out,*state)]
        ga = torch.autograd.grad((out,*state), a, weights)
        gb = torch.autograd.grad((ro,*rs), b, weights)
        for name, got, ref in zip(('q','k','v','alpha','beta','keys0','values0'),ga,gb):
            check(f'end_to_end_{tag}/grad_{name}', got.float(), ref.float(),
                  atol=3e-3 if dtype == torch.bfloat16 else 2e-4,
                  rtol=1e-2 if dtype == torch.bfloat16 else 1e-3)

    # duplicates, zero features, and zero/one gates remain finite with explicit ridge
    xs, slots = model_inputs(device, t=5)
    xs[0].zero_()
    xs[1].zero_()
    xs[5].zero_()
    xs[4].zero_()
    xs[3][:,0].zero_()
    slots.zero_()
    a, b = clone_grad(xs), clone_grad(xs)
    out, state = call_model(a, slots, 'triton', ridge=1e-3)
    ro, rs = call_model(b, slots, 'torch', ridge=1e-3)
    ga = torch.autograd.grad(out.sum()+state[1].sum(), a, allow_unused=True)
    gb = torch.autograd.grad(ro.sum()+rs[1].sum(), b, allow_unused=True)
    check('duplicate_zero/output', out, ro)
    for i,(got,ref) in enumerate(zip(ga,gb)):
        check(f'duplicate_zero/grad{i}',got,ref,atol=3e-4,rtol=1e-3)

    xs, slots = model_inputs(device, t=13)
    a, b = clone_grad(xs), clone_grad(xs)
    full, sf = call_model(a, slots, 'triton')
    first, s1 = call_model([x[:,:5] for x in b[:5]]+b[5:], slots[:,:5], 'triton')
    second, s2 = call_model([x[:,5:] for x in b[:5]]+list(s1), slots[:,5:], 'triton')
    joined = torch.cat((first,second),1)
    check('streaming/output', joined, full)
    check('streaming/keys',s2[0],sf[0])
    check('streaming/values',s2[1],sf[1])
    ga = torch.autograd.grad(full.square().sum()+sf[1].square().sum(),a)
    gb = torch.autograd.grad(joined.square().sum()+s2[1].square().sum(),b)
    for i,(got,ref) in enumerate(zip(ga,gb)):
        check(f'streaming/grad{i}',got,ref,atol=2e-4,rtol=1e-3)

    changed = [x.clone() for x in xs]
    changed[0][:,7:] += 1.5
    changed[1][:,7:] -= 2
    changed[2][:,7:] += 4
    other = call_model(changed, slots, 'triton')[0]
    check('causality/prefix', other[:,:7], full.detach()[:,:7], 1e-6, 1e-6)

    bad = slots.clone()
    bad[0,0,0] = xs[5].shape[-2]
    try:
        call_model(xs,bad,'triton')
    except ValueError as exc:
        assert 'range' in str(exc)
    else:
        raise AssertionError('Out-of-range slot was accepted')
    REPORT['checks'].append({'name':'invalid_slot_rejected','passed':True})
    print('PASS end-to-end, streaming, causality, geometry gradients', flush=True)


def train_smoke():
    torch.manual_seed(73)
    b,t,h,d,m = 2,12,2,8,4
    x = torch.randn(b,t,h,d,device='cuda')
    target = .2*torch.tanh(x)
    maps = torch.nn.ModuleList([torch.nn.Linear(d,d,bias=False) for _ in range(3)]).cuda()
    router = torch.nn.Linear(d,m).cuda()
    initial_keys = torch.nn.Parameter(torch.randn(b,h,m,d,device='cuda'))
    params = list(maps.parameters())+list(router.parameters())+[initial_keys]
    opt = torch.optim.Adam(params,lr=.02)
    values0 = torch.zeros(b,h,m,d,device='cuda')
    losses = []
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        q,k,v = [layer(x) for layer in maps]
        probs = router(x).softmax(-1)
        slots = probs.argmax(-1)
        beta = .5*probs.gather(-1,slots[...,None]).squeeze(-1)
        alpha = torch.full_like(beta,.98)
        output,_ = projected_slot_attention(q,k,v,alpha,beta,slots,(initial_keys,values0))
        loss = F.mse_loss(output,target)
        loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0], losses
    REPORT['training_smoke'] = {'steps':30,'losses':losses,
                                 'scope':'Synthetic fitting only, not LM/capacity evidence'}
    print('PASS training smoke',losses[0],losses[-1],flush=True)


def benchmark():
    import triton
    from komp_slot.kernels import fused_recurrent_slot
    rows = []
    for shape in [(2,128,4,8,32),(2,256,4,16,64)]:
        xs, slots = core_inputs(shape,'cuda')
        def measure(fn):
            return triton.testing.do_bench(fn,warmup=30,rep=100)
        with torch.no_grad():
            naive = measure(lambda:call_core(recurrent_reference,xs,slots))
            fused = measure(lambda:call_core(fused_recurrent_slot,xs,slots))
        gradxs = clone_grad(xs)
        def train(fn):
            o,s = call_core(fn,gradxs,slots)
            torch.autograd.grad(o.sum()+s.sum(),gradxs)
        naive_train = measure(lambda:train(recurrent_reference))
        fused_train = measure(lambda:train(fused_recurrent_slot))
        rows.append({'B_T_H_M_V':shape,'torch_fwd_ms':naive,'triton_fwd_ms':fused,
                     'torch_fwd_bwd_ms':naive_train,'triton_fwd_bwd_ms':fused_train})
    xs,slots = model_inputs('cuda',t=32,m=8)
    with torch.no_grad():
        geometry = measure(lambda:prepare_geometry(xs[0],xs[1],slots,xs[5]))
        end_to_end = measure(lambda:call_model(xs,slots,'triton'))
        torch_total = measure(lambda:call_model(xs,slots,'torch'))
    REPORT['benchmark'] = {'core':rows,'end_to_end':{'B_T_H_K_V_M':[2,32,2,7,11,8],
                           'geometry_ms':geometry,'triton_total_ms':end_to_end,'torch_total_ms':torch_total},
                           'timing':'triton.testing.do_bench, warmup=30ms, rep=100ms; compilation excluded',
                           'baseline':'PyTorch reference for the same operator; not GSA/GDN model throughput'}
    print('BENCHMARK',json.dumps(REPORT['benchmark']),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    if not args.cpu:
        assert torch.cuda.is_available(), 'GPU validation requires CUDA'
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory=True
    import google.protobuf
    from google.protobuf.internal.api_implementation import Type
    REPORT['environment']={'python':platform.python_version(),'torch':torch.__version__,
                           'cuda':torch.version.cuda,'protobuf':google.protobuf.__version__,
                           'protobuf_backend':Type(),'device':'cpu' if args.cpu else torch.cuda.get_device_name(),
                           'tf32':False,'nan_poisoning':True}
    if not args.cpu:
        import triton
        REPORT['environment']['triton']=triton.__version__
        REPORT['environment']['nvidia_smi']=subprocess.check_output(
            ['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip()
    REPORT['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(ROOT.rglob('*.py'))}
    print('ENVIRONMENT',json.dumps(REPORT['environment']),flush=True)
    started=time.monotonic()
    try:
        validate_model('cpu' if args.cpu else 'cuda')
        if not args.cpu:
            validate_core()
            train_smoke()
            if args.benchmark:
                benchmark()
        REPORT['status']='passed'
    except Exception as exc:
        REPORT['status']='failed'
        REPORT['error']=repr(exc)
        raise
    finally:
        REPORT['elapsed_seconds']=time.monotonic()-started
        dest=ROOT/'evidence'/('cpu.json' if args.cpu else 'gpu.json')
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(json.dumps(REPORT,indent=2)+'\n')
        print('REPORT',str(dest),'checks',len(REPORT['checks']),'status',REPORT['status'],flush=True)


if __name__=='__main__':
    main()
