"""Acceptance and direct mixed/all-Triton comparisons; no tolerance changes."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
import triton
import validate as base
import validate_chunk as chunks
from komp_slot import projected_slot_attention
from komp_slot.geometry import triton_geometry
from komp_slot.geometry_chunk import triton_geometry_chunked
from komp_slot.fast_chunk import fast_chunk
from komp_slot.triton_chunk import triton_chunk,prepare
from komp_slot.chunk import chunk_coefficients,chunk_slot
from komp_slot.kernels import fused_recurrent_slot
from komp_slot.reference import prepare_geometry

ROOT=Path(__file__).parent
REPORT=base.REPORT


def coefficients():
    for c in (16,32):
        for shape in [(1,1,1,1,1),(2,17,3,5,37),(1,65,2,32,64),(1,33,1,64,65)]:
            xs,slots=base.core_inputs(shape,'cuda')
            xs[4][:,0]=0
            xs[4][:,-1]=1
            slots.fill_(shape[3]-1)
            got=prepare(*xs[:3],xs[4],xs[5],slots,c)
            ref=chunk_coefficients(*xs[:3],xs[4],xs[5],slots,c,'torch')
            for name,a,b in zip(('OS','OV','FS','FV'),got,ref):
                base.check(f'coeff{c}/{shape}/{name}',a,b)
    print('PASS coefficients',flush=True)


def geometry():
  for name,fn in (('v1',triton_geometry),('chunked',triton_geometry_chunked)):
    for m,t,dtype in [(1,1,torch.float32),(5,19,torch.float32),(16,33,torch.float32),
                       (32,35,torch.float32),(64,7,torch.float32),(5,19,torch.bfloat16)]:
        xs,slots=base.model_inputs('cuda',dtype,t,m)
        q,k,initial=[xs[i].detach().requires_grad_() for i in (0,1,5)]
        qr,kr,ir=[x.detach().clone().requires_grad_() for x in (q,k,initial)]
        got=fn(q,k,slots,initial)
        ref=prepare_geometry(qr,kr,slots,ir)
        tag=f'geometry/{name}/{m}/{t}/{dtype}'
        for i,(a,b) in enumerate(zip(got,ref)):
            base.check(tag+f'/out{i}',a,b)
        weights=[torch.randn_like(x) for x in got]
        ga=torch.autograd.grad(got,(q,k,initial),weights)
        gb=torch.autograd.grad(ref,(qr,kr,ir),weights)
        for i,(a,b) in enumerate(zip(ga,gb)):
            base.check(tag+f'/grad{i}',a.float(),b.float(),3e-3 if dtype==torch.bfloat16 else 2e-4,
                       1e-2 if dtype==torch.bfloat16 else 1e-3)
        print('PASS',tag,flush=True)


def models():
    original=base.projected_slot_attention
    for c in (16,32):
        def call(*args,**kwargs):
            if kwargs.get('backend','triton')=='triton':
                kwargs['backend']='all_triton'; kwargs['chunk_size']=c
            return original(*args,**kwargs)
        base.projected_slot_attention=call
        base.validate_model('cuda')
    base.train_smoke()
    base.projected_slot_attention=original


def extended():
    for kind in ('nan','gate','slot'):
        xs,slots=base.model_inputs('cuda',t=3,m=2)
        if kind=='nan':
            xs[0][0,0,0,0]=float('nan')
        elif kind=='gate':
            xs[3][0,0,0]=1.1
        else:
            slots[0,0,0]=-1
        try:
            base.call_model(xs,slots,'all_triton')
        except ValueError:
            REPORT['checks'].append({'name':'input_rejected/'+kind,'passed':True})
        else:
            raise AssertionError('invalid input accepted: '+kind)
    for c in (16,32):
        xs,slots=base.model_inputs('cuda',t=67)
        a,b=base.clone_grad(xs),base.clone_grad(xs)
        def call(xs,ss):
            return projected_slot_attention(*xs[:5],ss,(xs[5],xs[6]),backend='all_triton',chunk_size=c)
        full,sf=call(a,slots)
        first,s1=call([x[:,:29] for x in b[:5]]+b[5:],slots[:,:29])
        second,s2=call([x[:,29:] for x in b[:5]]+list(s1),slots[:,29:])
        joined=torch.cat((first,second),1)
        base.check(f'long_stream{c}/output',joined,full)
        base.check(f'long_stream{c}/state',s2[1],sf[1])
        ga=torch.autograd.grad(full.square().sum()+sf[1].square().sum(),a)
        gb=torch.autograd.grad(joined.square().sum()+s2[1].square().sum(),b)
        for i,(got,ref) in enumerate(zip(ga,gb)):
            base.check(f'long_stream{c}/grad{i}',got,ref,2e-4,1e-3)
    for dtype in (torch.float16,torch.float32):
        xs,slots=base.model_inputs('cuda',dtype,t=35,m=8)
        # Non-power-of-two QK dimensions and a longer chunk boundary.
        for i in (0,1,5):
            xs[i]=torch.randn(*xs[i].shape[:-1],33,device='cuda',dtype=dtype)
        a,b=base.clone_grad(xs),base.clone_grad(xs)
        out,state=base.call_model(a,slots,'all_triton')
        ro,rs=base.call_model(b,slots,'torch')
        base.check(f'wide/{dtype}/output',out,ro)
        ga=torch.autograd.grad((out,*state),a,[torch.ones_like(x) for x in (out,*state)])
        gb=torch.autograd.grad((ro,*rs),b,[torch.ones_like(x) for x in (ro,*rs)])
        for i,(got,ref) in enumerate(zip(ga,gb)):
            base.check(f'wide/{dtype}/grad{i}',got.float(),ref.float(),
                       3e-3 if dtype==torch.float16 else 2e-4,1e-2 if dtype==torch.float16 else 1e-3)


def profile_path():
    from torch.profiler import profile,ProfilerActivity
    xs,slots=base.model_inputs('cuda',t=32,m=8)
    xs=base.clone_grad(xs)
    weights=[torch.randn_like(xs[2]),torch.randn_like(xs[5]),torch.randn_like(xs[6])]
    def run():
        o,s=base.call_model(xs,slots,'all_triton')
        torch.autograd.grad((o,*s),xs,weights)
    run()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
        run()
    keys=sorted({e.key for e in prof.key_averages()})
    forbidden={'aten::mm','aten::bmm','aten::matmul','aten::linalg_solve','aten::linalg_solve_ex',
               'aten::exp','aten::sum','aten::mul','aten::add','aten::div','aten::pow',
               'aten::linalg_vector_norm','aten::constant_pad_nd','aten::one_hot','aten::cumsum','aten::cumprod'}
    assert not forbidden.intersection(keys),forbidden.intersection(keys)
    REPORT['profile']={'events':keys,'forbidden_tensor_math':sorted(forbidden),'status':'passed'}
    print('PASS profiler: no ATen tensor math in forward/backward',flush=True)


def benchmark():
    bench=lambda f:triton.testing.do_bench(f,warmup=30,rep=100)
    rows=[]
    for shape in [(2,256,4,16,64),(2,1024,4,16,64),(2,2048,4,32,64)]:
        xs,slots=base.core_inputs(shape,'cuda')
        for c in (16,32):
            row={'shape':shape,'chunk_size':c}
            a,b=base.clone_grad(xs),base.clone_grad(xs)
            co,cs=base.call_core(lambda *args:triton_chunk(*args,chunk_size=c),a,slots)
            fo,fs=base.call_core(fused_recurrent_slot,b,slots)
            base.check(f'long{shape[1]}/{c}/output',co,fo)
            base.check(f'long{shape[1]}/{c}/state',cs,fs)
            ga=torch.autograd.grad(co.square().sum()+cs.square().sum(),a)
            gb=torch.autograd.grad(fo.square().sum()+fs.square().sum(),b)
            for i,(got,ref) in enumerate(zip(ga,gb)):
                base.check(f'long{shape[1]}/{c}/grad{i}',got,ref,6e-5,6e-4)
            for name,fn in [('recurrent',fused_recurrent_slot),
                            ('mixed_chunk',lambda *a:chunk_slot(*a,chunk_size=c)),
                            ('triton_chunk',lambda *a:triton_chunk(*a,chunk_size=c))]:
                with torch.no_grad():
                    row[name+'_fwd_ms']=bench(lambda:base.call_core(fn,xs,slots))
                gradxs=base.clone_grad(xs)
                def train():
                    o,s=base.call_core(fn,gradxs,slots)
                    torch.autograd.grad(o.sum()+s.sum(),gradxs)
                row[name+'_fwd_bwd_ms']=bench(train)
            with torch.no_grad():
                row['coefficient_ms']=bench(lambda:prepare(*xs[:3],xs[4],xs[5],slots,c))
            rows.append(row)
            print('BENCH CORE',json.dumps(row),flush=True)
    full=[]
    for t,m in [(32,8),(256,16),(1024,16)]:
        xs,slots=base.model_inputs('cuda',t=t,m=m)
        row={'shape':[2,t,2,7,11,m]}
        a,b=base.clone_grad(xs),base.clone_grad(xs)
        co,cs=base.call_model(a,slots,'all_triton')
        fo,fs=base.call_model(b,slots,'chunk')
        base.check(f'full_long{t}/output',co,fo)
        base.check(f'full_long{t}/state',cs[1],fs[1])
        ga=torch.autograd.grad(co.square().sum()+cs[1].square().sum(),a)
        gb=torch.autograd.grad(fo.square().sum()+fs[1].square().sum(),b)
        for i,(got,ref) in enumerate(zip(ga,gb)):
            base.check(f'full_long{t}/grad{i}',got,ref,2e-4,1e-3)
        for backend in ('chunk','all_triton'):
            with torch.no_grad():
                row[backend+'_fwd_ms']=bench(lambda:base.call_model(xs,slots,backend))
            gradxs=base.clone_grad(xs)
            def train():
                o,s=base.call_model(gradxs,slots,backend)
                torch.autograd.grad(o.sum()+s[1].sum(),gradxs)
            row[backend+'_fwd_bwd_ms']=bench(train)
        full.append(row)
        print('BENCH FULL',json.dumps(row),flush=True)
    REPORT['benchmark']={'core':rows,'full':full,'timing':'warmup 30ms, rep 100ms; excludes compilation'}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--benchmark',action='store_true'); parser.add_argument('--output',default=None)
    args=parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory=True
    import google.protobuf
    from google.protobuf.internal.api_implementation import Type
    REPORT['environment']={'python':platform.python_version(),'torch':torch.__version__,
        'triton':triton.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
        'protobuf':google.protobuf.__version__,'protobuf_backend':Type(),
        'chunk_dot':'tf32x3','geometry_dot':'ieee','nan_poisoning':True}
    REPORT['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(ROOT.rglob('*.py'))}
    started=time.monotonic()
    try:
        coefficients()
        chunks.chunk_slot=lambda *a,**kw:triton_chunk(*a,chunk_size=kw.get('chunk_size',16))
        chunks.core('cuda')
        chunks.chunk_slot=lambda *a,**kw:fast_chunk(*a,chunk_size=kw.get('chunk_size',16))
        chunks.core('cuda')
        geometry()
        models()
        extended()
        profile_path()
        if args.benchmark:
            benchmark()
        REPORT['status']='passed'
    except Exception as exc:
        REPORT['status']='failed';REPORT['error']=repr(exc)
        raise
    finally:
        REPORT['elapsed_seconds']=time.monotonic()-started
        dest=Path(args.output) if args.output else ROOT/'evidence'/'all-triton-gpu.json';dest.parent.mkdir(exist_ok=True)
        dest.write_text(json.dumps(REPORT,indent=2)+'\n')
        print('REPORT',len(REPORT['checks']),REPORT['status'],flush=True)


if __name__=='__main__':
    main()
