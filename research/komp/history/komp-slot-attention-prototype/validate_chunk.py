"""Chunk acceptance against the dense recurrence, including zero-gate gradients."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
import triton

import validate as base
from komp_slot.chunk import _UnitLowerInverse, chunk_coefficients, chunk_slot, _ChunkScan
from komp_slot.kernels import fused_recurrent_slot
from komp_slot.reference import recurrent_reference

ROOT = Path(__file__).parent
REPORT = base.REPORT


def core(device):
    backend = 'torch' if device == 'cpu' else 'triton'
    dtype = torch.float64 if device == 'cpu' else torch.float32
    cases = [(1,1,1,1,1), (2,17,3,5,37), (1,33,2,16,64),
             (2,65,2,32,17), (1,129,1,64,65)]
    for c in (16,32):
        for n, shape in enumerate(cases):
            xs, slots = base.core_inputs(shape, device, n)
            xs = [x.to(dtype) for x in xs]
            xs[4][:,0] = 0
            xs[4][:,-1] = 1
            if shape[1] > c:
                xs[4][:,c-1] = 0
                xs[4][:,c] = 0
            xs[5][:,0] = 1
            if n == 1:
                slots.fill_(shape[3]-1)
            a = base.clone_grad(xs)
            b = [x.double().detach().requires_grad_() for x in xs]
            fn = lambda *args: chunk_slot(*args, chunk_size=c, backend=backend)
            out, final = base.call_core(fn, a, slots)
            ro, rf = base.call_core(recurrent_reference,b,slots)
            atol, rtol = (2e-11,2e-10) if device == 'cpu' else (3e-5,3e-4)
            tag = f'chunk{c}/case{n}'
            base.check(tag+'/output',out,ro,atol,rtol)
            base.check(tag+'/state',final,rf,atol,rtol)
            go,gf = torch.randn_like(out),torch.randn_like(final)
            ga = torch.autograd.grad((out,final),a,(go,gf))
            gb = torch.autograd.grad((ro,rf),b,(go.double(),gf.double()))
            for i,(got,ref) in enumerate(zip(ga,gb)):
                base.check(tag+f'/grad{i}',got,ref,atol*2,rtol*2)
            if device == 'cuda':
                fo,ff = base.call_core(fused_recurrent_slot,xs,slots)
                base.check(tag+'/fused_output',out,fo)
                base.check(tag+'/fused_state',final,ff)
            print('PASS',tag,shape,flush=True)
        xs,slots = base.core_inputs((1,19,2,5,33),device,9)
        xs = [x.to(dtype) for x in xs]
        for which in (0,1):
            a,b = base.clone_grad(xs),base.clone_grad(xs)
            co=base.call_core(fn,a,slots)[which]
            ro=base.call_core(recurrent_reference,b,slots)[which]
            ga=torch.autograd.grad(co.square().sum(),a,allow_unused=True)
            gb=torch.autograd.grad(ro.square().sum(),b,allow_unused=True)
            for i,(got,ref) in enumerate(zip(ga,gb)):
                got=torch.zeros_like(a[i]) if got is None else got
                base.check(f'chunk{c}/cotangent{which}/grad{i}',got,
                           torch.zeros_like(got) if ref is None else ref,atol*2,rtol*2)
        noncontig=[torch.stack((x,x),-1)[...,0] for x in xs]
        out,final=base.call_core(fn,noncontig,slots)
        ro,rf=base.call_core(recurrent_reference,xs,slots)
        base.check(f'chunk{c}/strided/output',out,ro,atol,rtol)
        base.check(f'chunk{c}/strided/state',final,rf,atol,rtol)


def inverse_check():
    for n in (32,64):
        for repeated in (False,True):
            torch.manual_seed(n)
            low=torch.randn(2,3,n,n,device='cuda')*.03
            if repeated:
                low.fill_(-.95)
            low=low.tril(-1).requires_grad_()
            ref=low.detach().double().requires_grad_()
            eye=torch.eye(n,device='cuda',dtype=torch.float64).expand_as(ref)
            out=_UnitLowerInverse.apply(low)
            expected=torch.linalg.solve_triangular(eye-ref,eye,upper=False,unitriangular=True)
            base.check(f'inverse{n}/repeat{repeated}/forward',out,expected)
            grad=torch.randn_like(out)
            got,=torch.autograd.grad(out,low,grad)
            want,=torch.autograd.grad(expected,ref,grad.double())
            base.check(f'inverse{n}/repeat{repeated}/backward',got,want,6e-5,6e-4)


def model():
    original=base.projected_slot_attention
    for c in (16,32):
        def chunk_model(*args,**kwargs):
            if kwargs.get('backend','triton')=='triton':
                kwargs['backend']='chunk'
                kwargs['chunk_size']=c
            return original(*args,**kwargs)
        base.projected_slot_attention=chunk_model
        base.validate_model('cuda')
        # Existing model suite tests short streams; also cross several chunk boundaries.
        xs,slots=base.model_inputs('cuda',t=67)
        a,b=base.clone_grad(xs),base.clone_grad(xs)
        out,state=base.call_model(a,slots,'triton')
        ro,rs=base.call_model(b,slots,'torch')
        base.check(f'full{c}/output',out,ro)
        base.check(f'full{c}/state',state[1],rs[1])
        ga=torch.autograd.grad(out.square().sum()+state[1].square().sum(),a)
        gb=torch.autograd.grad(ro.square().sum()+rs[1].square().sum(),b)
        for i,(got,ref) in enumerate(zip(ga,gb)):
            base.check(f'full{c}/grad{i}',got,ref,2e-4,1e-3)
    base.train_smoke()
    base.projected_slot_attention=original


def benchmark():
    rows=[]
    measure=lambda fn: triton.testing.do_bench(fn,warmup=30,rep=100)
    for shape in [(2,256,4,16,64),(2,1024,4,16,64),(2,2048,4,32,64)]:
        xs,slots=base.core_inputs(shape,'cuda')
        with torch.no_grad():
            recurrent=measure(lambda:base.call_core(fused_recurrent_slot,xs,slots))
        gradxs=base.clone_grad(xs)
        def train(fn):
            o,s=base.call_core(fn,gradxs,slots)
            torch.autograd.grad(o.sum()+s.sum(),gradxs)
        recurrent_train=measure(lambda:train(fused_recurrent_slot))
        for c in (16,32):
            fn=lambda *args:chunk_slot(*args,chunk_size=c)
            a,b=base.clone_grad(xs),base.clone_grad(xs)
            co,cs=base.call_core(fn,a,slots)
            fo,fs=base.call_core(fused_recurrent_slot,b,slots)
            tag=f'long{shape[1]}/chunk{c}'
            base.check(tag+'/output',co,fo)
            base.check(tag+'/state',cs,fs)
            ga=torch.autograd.grad(co.square().sum()+cs.square().sum(),a)
            gb=torch.autograd.grad(fo.square().sum()+fs.square().sum(),b)
            for i,(got,ref) in enumerate(zip(ga,gb)):
                base.check(tag+f'/grad{i}',got,ref,6e-5,6e-4)
            with torch.no_grad():
                total=measure(lambda:base.call_core(fn,xs,slots))
                coeff=lambda:chunk_coefficients(*xs[:3],xs[4],xs[5],slots,c)
                prep=measure(coeff)
                maps=coeff()
                b,t,h,m,v=shape
                values=xs[3].transpose(1,2).reshape(b,h,t//c,c,v).contiguous()
                scan=measure(lambda:_ChunkScan.apply(*maps,values,xs[6],False))
            train_ms=measure(lambda:train(fn))
            rows.append({'B_T_H_M_V':shape,'chunk_size':c,'recurrent_fwd_ms':recurrent,
                         'chunk_fwd_ms':total,'coefficient_ms':prep,'scan_ms':scan,
                         'recurrent_fwd_bwd_ms':recurrent_train,'chunk_fwd_bwd_ms':train_ms})
    REPORT['chunk_benchmark']={'rows':rows,'timing':'warmup 30ms, rep 100ms, excludes compilation',
                              'scope':'Core includes chunk map construction, excludes key/Gram geometry'}
    print('BENCHMARK',json.dumps(REPORT['chunk_benchmark']),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    device='cpu' if args.cpu else 'cuda'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory=True
    import google.protobuf
    from google.protobuf.internal.api_implementation import Type
    REPORT['environment']={'python':platform.python_version(),'torch':torch.__version__,
                           'triton':triton.__version__,'cuda':torch.version.cuda,
                           'protobuf':google.protobuf.__version__,'protobuf_backend':Type(),
                           'device':device if args.cpu else torch.cuda.get_device_name(),
                           'tf32':False,'nan_poisoning':True}
    REPORT['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(ROOT.rglob('*.py'))}
    started=time.monotonic()
    try:
        core(device)
        if not args.cpu:
            inverse_check()
            model()
            if args.benchmark:
                benchmark()
        REPORT['status']='passed'
    except Exception as exc:
        REPORT['status']='failed'
        REPORT['error']=repr(exc)
        raise
    finally:
        REPORT['elapsed_seconds']=time.monotonic()-started
        dest=ROOT/'evidence'/('chunk-cpu.json' if args.cpu else 'chunk-gpu.json')
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(json.dumps(REPORT,indent=2)+'\n')
        print('REPORT',len(REPORT['checks']),REPORT['status'],flush=True)


if __name__=='__main__':
    main()
