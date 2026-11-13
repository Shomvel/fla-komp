"""Frozen-source single-GPU numerical checks and bounded mixer benchmarks."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
import triton

from komp_slot.value_decay import block_komp_value_decay, ValueDecayGate

ROOT=Path(__file__).resolve().parent
checks=[]


def inputs(b,t,h,k,v,m,dtype,seed=42):
    torch.manual_seed(seed)
    a=[torch.randn(b,t,h,k,device='cuda'),torch.randn(b,t,h,k,device='cuda'),
       torch.randn(b,t,h,v,device='cuda'),-torch.rand(b,t,h,v,device='cuda')*.1,
       torch.rand(b,t,h,device='cuda'),torch.randn(b,h,m,k,device='cuda'),
       torch.randn(b,h,m,v,device='cuda')*.1]
    return [x.to(dtype) if i in (0,1,2) else x for i,x in enumerate(a)]


def run(xs,c,backend):
    o,(k,v)=block_komp_value_decay(*xs[:5],(xs[5],xs[6]),chunk_size=c,backend=backend)
    return o,k,v


def check(name,a,b,atol=3e-4,rtol=3e-3):
    a,b=a.double(),b.double()
    delta=(a-b).abs();scale=b.square().mean().sqrt()
    rel=float(delta.square().mean().sqrt()/scale.clamp_min(1e-12))
    item=dict(name=name,max_abs=float(delta.max()),rms_relative=rel,atol=atol,rtol=rtol)
    checks.append(item)
    assert bool(torch.isfinite(a).all()),item
    assert float(delta.max())<=atol or rel<=rtol,item


def validation():
    cases=[(1,1,1,3,2,1,1),(1,9,2,5,4,3,1),(2,17,2,5,7,3,4),
           (1,7,2,4,4,4,3),(1,129,2,64,64,32,128),(1,65,2,64,32,32,1)]
    for ci,(b,t,h,k,v,m,c) in enumerate(cases):
        for dtype in (torch.float32,torch.bfloat16):
            xs=inputs(b,t,h,k,v,m,dtype,42+ci)
            aa=[x.detach().requires_grad_() for x in xs]
            rr=[x.detach().double().requires_grad_() for x in xs]
            a=run(aa,c,'triton');r=run(rr,c,'recurrent')
            tag=f'{b}x{t}x{h}/K{k}V{v}M{m}C{c}/{dtype}'
            tol=.025 if dtype==torch.bfloat16 else .003
            for n,x,y in zip(('out','keys','coef'),a,r):check(tag+'/'+n,x,y,rtol=tol)
            w=[torch.randn_like(x) for x in a]
            ga=torch.autograd.grad(a,aa,w);gr=torch.autograd.grad(r,rr,[x.double() for x in w])
            for n,x,y in zip(('dq','dk','dv','dg','db','dk0','dc0'),ga,gr):check(tag+'/'+n,x,y,rtol=tol)
            print('PASS',tag,flush=True)
    for c in (1,4):
        for g in (0.,-1000.):
            xs=inputs(1,9,2,5,7,3,torch.float32)
            xs[3].fill_(g);xs[4][:,1]=0;xs[4][:,2]=1;xs[0][:,0]=0;xs[1][:,1]=0
            a=[x.detach().requires_grad_() for x in xs];r=[x.detach().double().requires_grad_() for x in xs]
            aa=run(a,c,'triton');rr=run(r,c,'recurrent')
            for i,(x,y) in enumerate(zip(aa,rr)):check(f'extreme/{g}/C{c}/{i}',x,y)
            wa=[torch.randn_like(x) for x in aa]
            ga=torch.autograd.grad(aa,a,wa);gr=torch.autograd.grad(rr,r,[x.double() for x in wa])
            for i,(x,y) in enumerate(zip(ga,gr)):check(f'extreme/{g}/C{c}/grad{i}',x,y,rtol=.006)
    # Value channel gates identical -> scalar decay limit already covered by
    # CPU suite; here test future independence and boundary streaming on CUDA.
    xs=inputs(1,9,2,5,7,3,torch.float32)
    for c in (1,4):
        base=run(xs,c,'triton');edited=[x.clone() for x in xs]
        for i in (0,1,2):edited[i][:,4:]+=10
        edited[3][:,4:]=-3;edited[4][:,4:]=0
        check(f'causal/C{c}',run(edited,c,'triton')[0][:,:4],base[0][:,:4],atol=1e-6,rtol=1e-6)
        a=run([x[:,:4] for x in xs[:5]]+xs[5:],c,'triton')
        b=run([x[:,4:] for x in xs[:5]]+[a[1],a[2]],c,'triton')
        check(f'stream/C{c}',torch.cat((a[0],b[0]),1),base[0],atol=1e-5,rtol=1e-5)
        check(f'stream_coef/C{c}',b[2],base[2],atol=1e-5,rtol=1e-5)
    # Small real autograd/optimizer chain (synthetic IDs, not pretraining).
    for c in (1,4):
        torch.manual_seed(42)
        emb=torch.nn.Embedding(31,12,device='cuda');proj=torch.nn.Linear(12,24,device='cuda')
        gate=ValueDecayGate(12,2,4,rank=3).cuda();head=torch.nn.Linear(8,31,device='cuda')
        params=[*emb.parameters(),*proj.parameters(),*gate.parameters(),*head.parameters()]
        opt=torch.optim.AdamW(params,lr=3e-4);ids=torch.randint(0,31,(1,10),device='cuda');losses=[]
        for _ in range(3):
            opt.zero_grad();x=emb(ids[:,:-1]);q,k,v=[y.reshape(1,9,2,4) for y in proj(x).chunk(3,-1)]
            beta=torch.full((1,9,2),.2,device='cuda');keys=torch.randn(1,2,3,4,device='cuda')
            o,_=block_komp_value_decay(q,k,v,gate(x),beta,(keys,torch.zeros_like(keys)),chunk_size=c,backend='triton')
            loss=torch.nn.functional.cross_entropy(head(o.flatten(-2)).flatten(0,1),ids[:,1:].flatten())
            loss.backward();assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
            opt.step();losses.append(float(loss.detach()))
        checks.append({'name':f'LM_Adam/C{c}','losses':losses,'passed':True})


def benchmark():
    results=[]
    for c in (1,128):
        for b,t,h in ((1,256,4),(16,2048,16)):
            # Actual single-layer B16 training shape. Geometry temporary size
            # is independent of V, but Exact still has per-token Gram solves.
            xs=[x.requires_grad_() for x in inputs(b,t,h,64,64,32,torch.bfloat16)]
            def iteration():
                for x in xs:x.grad=None
                o,k,s=run(xs,c,'triton')
                (o.float().square().mean()+s.square().mean()+k.square().mean()).backward()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t0=time.monotonic()
            iteration();torch.cuda.synchronize();cold=time.monotonic()-t0
            times=[]
            for _ in range(3):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();iteration();end.record();end.synchronize();times.append(start.elapsed_time(end))
            assert all(x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in xs)
            row={'shape':[b,t,h,64,64,32],'chunk':c,'backend':'hybrid_triton','cold_seconds':cold,
                 'forward_backward_ms':times,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                 'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            results.append(row);print('BENCH',json.dumps(row),flush=True)
            del xs;torch.cuda.empty_cache()
    return results


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    import google.protobuf
    from google.protobuf.internal.api_implementation import Type
    report={'status':'running','checks':checks,'torch':torch.__version__,'triton':triton.__version__,
            'python':platform.python_version(),'protobuf':google.protobuf.__version__,'protobuf_backend':Type(),
            'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,
            'scope':'hybrid geometry PyTorch + state forward/backward Triton; single GPU synthetic validation'}
    start=time.monotonic()
    try:
        validation();report['numerical_status']='passed'
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
        report['benchmarks']=benchmark();report['status']='passed'
    except Exception as e:
        report['status']='failed';report['error']=repr(e);raise
    finally:
        report['seconds']=time.monotonic()-start
        report['source_sha256']={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
            ['komp_slot/value_decay.py','komp_slot/value_decay_triton.py','komp_slot/block_reference.py','validate_value_decay_triton.py']}
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
