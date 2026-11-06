"""Pinned FLA GSA/GDN versus KOMP: equal state storage, eager and CUDA graph."""
import gc
import hashlib
import importlib.machinery
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

import torch
import torch.nn.functional as F
import triton

ROOT=Path(__file__).parent
FLA_ROOT=Path(os.environ.get('FLA_SOURCE',ROOT/'fla-source'))
# Load unchanged operator modules without optional transformer-model registrations.
# The source files and all native Triton kernels are the pinned upstream checkout.
for name,rel in [('fla','fla'),('fla.ops','fla/ops'),('fla.modules','fla/modules')]:
    pkg=types.ModuleType(name)
    pkg.__path__=[str(FLA_ROOT/rel)]
    pkg.__file__=str(FLA_ROOT/rel/'__init__.py')
    pkg.__spec__=importlib.machinery.ModuleSpec(name,loader=None,is_package=True)
    pkg.__version__='0.6.0'
    sys.modules[name]=pkg
from fla.ops.gsa import chunk_gsa
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.gsa.naive import naive_recurrent_gsa
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
from fla.utils import assert_close,get_abs_err,get_err_ratio
from komp_slot import projected_slot_attention
from komp_slot.geometry import triton_geometry
from komp_slot.triton_chunk import triton_chunk
from komp_slot.reference import prepare_geometry,recurrent_reference

REPORT={'rows':[],'checks':[]}
DEST=ROOT/'evidence'/'fla-comparison.json'


def save():
    DEST.parent.mkdir(exist_ok=True)
    DEST.write_text(json.dumps(REPORT,indent=2)+'\n')


def inputs(t,dtype,m=32,b=2,h=4,d=64,v=64):
    torch.manual_seed(981+t+m)
    rand=lambda *s:torch.randn(*s,device='cuda',dtype=torch.float32)
    q=rand(b,t,h,d).to(dtype); k=rand(b,t,h,d).to(dtype); value=(rand(b,t,h,v)*.2).to(dtype)
    alpha=torch.full((b,t,h),.98,device='cuda')
    beta=torch.full_like(alpha,.2)
    g=alpha.log()
    gs=g[...,None].expand(b,t,h,m).contiguous()
    s=(1-gs.exp()).to(dtype)
    ki=rand(b,h,m,d)
    vi=rand(b,h,m,v)*.05
    hk=ki.transpose(-1,-2).contiguous()*.05
    gd=rand(b,h,d,v)*.05
    slots=torch.arange(t,device='cuda')[None,:,None].expand(b,t,h).remainder(m).contiguous()
    return {'komp':[q,k,value,alpha,beta,ki,vi], 'gsa':[q,k,value,s,gs,hk,vi],
            'gdn':[q,k,value,g,beta,gd]},slots


def clone(xs):
    return [x.detach().clone().requires_grad_() for x in xs]


def run(name,xs,slots,reference=False):
    if name=='komp_public':
        o,(ks,vs)=projected_slot_attention(*xs[:5],slots,(xs[5],xs[6]))
        return o,ks,vs
    if name=='komp':
        geo=prepare_geometry if reference else triton_geometry
        core=recurrent_reference if reference else triton_chunk
        h,p,r,ks=geo(xs[0],xs[1],slots,xs[5])
        o,vs=core(h,p,r,xs[2],xs[3],xs[4],slots,xs[6])
        return o,ks,vs
    if name=='komp_core':
        return triton_chunk(*xs[:6],slots,xs[6])
    if name=='gsa':
        fn=naive_recurrent_gsa if reference else chunk_gsa
        o,state=fn(*xs[:5],initial_state=(xs[5],xs[6]),output_final_state=True)
        return o,*state
    if name=='gdn':
        if reference:
            q,k=[F.normalize(x.float(),dim=-1).to(x.dtype) for x in xs[:2]]
            return naive_recurrent_gated_delta_rule(q,k,xs[2],xs[4],xs[3],initial_state=xs[5],output_final_state=True)
        return chunk_gated_delta_rule(*xs[:5],initial_state=xs[5],output_final_state=True,
                                     use_qk_l2norm_in_kernel=True)
    raise ValueError(name)


def check(name,ref,got,tol):
    assert bool(torch.isfinite(got).all()),name
    assert_close(name,ref.float(),got.float(),tol)
    REPORT['checks'].append({'name':name,'max_abs':get_abs_err(ref.float(),got.float()),
                             'rms_relative':get_err_ratio(ref.float(),got.float()),'rms_tolerance':tol})


def validate():
    for dtype in (torch.float32,torch.bfloat16):
        data,slots=inputs(65,dtype,b=1,h=2)
        for name in ('gsa','gdn','komp'):
            a,b=clone(data[name]),clone(data[name])
            out,ref=run(name,a,slots),run(name,b,slots,True)
            weights=[torch.randn_like(x) for x in out]
            tol=.005 if dtype==torch.float32 else .01
            for i,(x,y) in enumerate(zip(out,ref)):
                check(f'{name}/{dtype}/out{i}',y,x,tol)
            ga=torch.autograd.grad(out,a,weights)
            gb=torch.autograd.grad(ref,b,[w.to(y.dtype) for w,y in zip(weights,ref)])
            for i,(x,y) in enumerate(zip(ga,gb)):
                # Upstream tests/ops/test_gdn.py:test_chunk allows 2% for dg/dbeta.
                grad_tol=.02 if name=='gdn' and dtype==torch.bfloat16 and i in (3,4) else tol
                check(f'{name}/{dtype}/grad{i}',y,x,grad_tol)
            print('PASS reference',name,str(dtype),flush=True)
    save()


def measure(fn):
    # Warm/autotune before either timer; graph compilation/capture is excluded.
    for _ in range(3):fn()
    torch.cuda.synchronize()
    eager=[]
    for _ in range(3):
        eager.append(float(triton.testing.do_bench(fn,warmup=20,rep=80)))
    graph=[]
    for _ in range(3):
        graph.append(float(triton.testing.do_bench_cudagraph(fn,rep=80)))
    return {'eager_ms':statistics.median(eager),'graph_ms':statistics.median(graph),
            'eager_trials_ms':eager,'graph_trials_ms':graph}


def benchmark_case(t,dtype,m=32):
    data,slots=inputs(t,dtype,m)
    with torch.no_grad():
        h,p,r,_=triton_geometry(data['komp'][0],data['komp'][1],slots,data['komp'][5])
    data['komp_core']=[h,p,r,data['komp'][2],data['komp'][3],data['komp'][4],data['komp'][6]]
    row={'B':2,'T':t,'H':4,'Dk':64,'Dv':64,'M':m,'input_dtype':str(dtype),
         'state_elements_per_head':{'komp':m*128,'gsa':m*128,'gdn':4096},'operators':{}}
    for name in ('gdn','gsa','komp','komp_core'):
        xs=data[name]
        print('START',t,m,str(dtype),name,flush=True)
        with torch.no_grad():
            result=run(name,xs,slots)
            for x in result:assert bool(torch.isfinite(x).all())
            fwd=measure(lambda:run(name,xs,slots))
        gradxs=clone(xs)
        weights=[torch.randn_like(x)*.01 for x in result]
        del result
        def train():
            out=run(name,gradxs,slots)
            return torch.autograd.grad(out,gradxs,weights)
        grads=train()
        assert all(bool(torch.isfinite(x).all()) for x in grads)
        del grads
        both=measure(train)
        row['operators'][name]={'forward':fwd,'forward_backward':both}
        print('RESULT',t,m,str(dtype),name,json.dumps(row['operators'][name]),flush=True)
        gc.collect()
    # Public API includes finite/range checks and one CPU synchronization, so no graph capture.
    with torch.no_grad():
        xs=data['komp']
        for _ in range(3):run('komp_public',xs,slots)
        public=[float(triton.testing.do_bench(lambda:run('komp_public',xs,slots),warmup=20,rep=80)) for _ in range(3)]
    row['public_komp_forward_eager_ms']=statistics.median(public)
    REPORT['rows'].append(row)
    save()


def main():
    import google.protobuf
    from google.protobuf.internal.api_implementation import Type
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(False)
    torch.utils.deterministic.fill_uninitialized_memory=False
    REPORT['environment']={'python':platform.python_version(),'torch':torch.__version__,
        'triton':triton.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
        'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
        'protobuf':google.protobuf.__version__,'protobuf_backend':Type(),
        'fla_sha':'97874af2400916e2885c33ea30559a70ff686b66',
        'fla_import':'operator namespaces only; unchanged upstream operator and utility files',
        'fla_dispatch':'native Triton (FLA_DISABLE_BACKEND_DISPATCH=1)',
        'tensor_cache':'disabled','nan_poisoning':False}
    REPORT['method']={'timing':'Median of 3 trials; eager warmup=20ms rep=80ms; CUDA graph rep=80ms',
        'validation':'FP32 RMS .005; BF16 RMS .01 except GDN dg/dbeta .02 per upstream test_chunk; r2 .01 gate failure retained',
        'scope':'Operator only; no input projections, output projection, conv, optimizer, routing network or gate activation',
        'komp':'Full includes normalization, route history, kernel Gram solve, chunk coeff/scan; no host numerical checks',
        'komp_core':'Precomputed h,p,r; geometry and its gradients excluded; diagnostic only',
        'gsa':'Native chunk_gsa, chunk64 checkpoint2; includes two memory stages and slot softmax',
        'gdn':'Native chunk_gated_delta_rule, chunk64, fused QK normalization',
        'komp_chunk':16,'states':'FP32, nonzero initial states, returned final states with gradients',
        'features':'Shared raw q/k/v, FP32 gates; independent s/g leaves for GSA',
        'precision':'FP32 and BF16 inputs tested; KOMP internally FP32 with tf32x3 block dots/IEEE Gram, FLA native precision',
        'equivalence':'Different algorithms; equal storage is not a claim of equal capacity or task quality'}
    files=[Path(__file__),*sorted((ROOT/'komp_slot').glob('*.py'))]
    REPORT['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    REPORT['fla_source_sha256']={str(p.relative_to(FLA_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted((FLA_ROOT/'fla').rglob('*.py'))}
    started=time.monotonic()
    try:
        validate()
        for t in (256,1024,2048,4096):benchmark_case(t,torch.bfloat16)
        benchmark_case(1024,torch.float32)
        for m in (16,64):benchmark_case(1024,torch.bfloat16,m)
        REPORT['status']='passed'
    except Exception as exc:
        REPORT['status']='failed';REPORT['error']=repr(exc)
        raise
    finally:
        REPORT['elapsed_seconds']=time.monotonic()-started
        save()
        print('REPORT',REPORT['status'],'rows',len(REPORT['rows']),flush=True)


if __name__=='__main__':main()
