"""Slot pre-decay acceptance against FP64 recurrence and scalar degeneration."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import triton
import validate as base
from komp_slot import projected_slot_attention
from komp_slot.reference import recurrent_reference
from komp_slot.triton_chunk import triton_chunk
from komp_slot.fast_chunk import fast_chunk

CORE=fast_chunk

ROOT=Path(__file__).resolve().parent
REPORT=base.REPORT


def slot_inputs(shape,dtype=torch.float32,zero=False):
    xs,slots=base.core_inputs(shape,'cuda',seed=731)
    b,t,h,m,v=shape
    xs[4]=torch.rand((b,t,h,m),device='cuda')*.15+.8
    if zero:
        xs[4][:,0]=0
        xs[4][:,-1]=1
        xs[4][:,t//2,:,0]=0
        slots.fill_(m-1)
    # Geometry stays FP32; the value input can be BF16 as in the small LM.
    xs[3]=xs[3].to(dtype)
    return xs,slots


def call(xs,slots,c=16):
    return CORE(*xs[:6],slots,xs[6],chunk_size=c)


def core():
    cases=[(1,1,1,1,1),(2,17,3,5,37),(1,65,2,32,64),
           (1,33,1,64,65),(2,192,2,32,64),(2,2048,4,32,64)]
    for c in (16,32):
        for i,shape in enumerate(cases):
            xs,slots=slot_inputs(shape,torch.bfloat16 if i==4 else torch.float32,zero=i%2==0)
            a=base.clone_grad(xs);b=[x.double().requires_grad_() for x in xs]
            got=call(a,slots,c);ref=base.call_core(recurrent_reference,b,slots)
            tag=f'core/c{c}/{shape}'
            for j,(x,y) in enumerate(zip(got,ref)):base.check(tag+f'/out{j}',x,y)
            weights=[torch.randn_like(x) for x in got]
            ga=torch.autograd.grad(got,a,weights)
            gb=torch.autograd.grad(ref,b,[w.double() for w in weights])
            for j,(x,y) in enumerate(zip(ga,gb)):
                base.check(tag+f'/grad{j}',x.float(),y.float(),3e-3 if xs[j].dtype==torch.bfloat16 else 8e-5,
                           1e-2 if xs[j].dtype==torch.bfloat16 else 8e-4)
            print('PASS',tag,flush=True)


def scalar_equivalence():
    for c in (16,32):
        xs,slots=base.core_inputs((2,67,2,32,37),'cuda',731)
        a,b=base.clone_grad(xs),base.clone_grad(xs)
        vector=b.copy();vector[4]=b[4][...,None].expand(*b[4].shape,32)
        got,ref=call(vector,slots,c),call(a,slots,c)
        weights=[torch.randn_like(x) for x in got]
        for j,(x,y) in enumerate(zip(got,ref)):base.check(f'scalar/c{c}/out{j}',x,y)
        ga=torch.autograd.grad(got,b,weights);gb=torch.autograd.grad(ref,a,weights)
        for j,(x,y) in enumerate(zip(ga,gb)):base.check(f'scalar/c{c}/grad{j}',x,y,8e-5,8e-4)
    print('PASS scalar equivalence (including summed slot da)',flush=True)


def finite_difference_and_order():
    xs,slots=slot_inputs((1,2,1,3,2),zero=False)
    ref=[x.double().detach() for x in xs]
    alpha=ref[4].clone().requires_grad_()
    def fn(a):
        inp=ref.copy();inp[4]=a
        return base.call_core(recurrent_reference,inp,slots)
    assert torch.autograd.gradcheck(fn,(alpha,),eps=1e-6,atol=1e-7,rtol=1e-5)
    REPORT['checks'].append({'name':'fp64_alpha_finite_difference','passed':True})
    # Zero decay must discard all old information but retain the current write.
    inp=[x[:,:1] for x in xs[:6]]+[xs[6]]
    inp[4]=torch.zeros_like(inp[4]);ss=slots[:,:1]
    out,final=call(inp,ss)
    e=torch.nn.functional.one_hot(ss[:,0],3).float()
    expected=inp[5][:,0,:,None,None]*e[...,None]*inp[3][:,0,:,None,:]
    base.check('zero_decay/current_write_survives',final,expected)
    base.check('zero_decay/current_read',out[:,0],(inp[2][:,0,...,None]*expected).sum(-2))
    print('PASS FP64 finite differences and pre-decay write order',flush=True)


def streaming_and_permutation():
    xs,slots=slot_inputs((1,67,2,5,37),zero=True)
    a,b=base.clone_grad(xs),base.clone_grad(xs)
    full=call(a,slots)
    first=call([x[:,:29] for x in b[:6]]+[b[6]],slots[:,:29])
    second=call([x[:,29:] for x in b[:6]]+[first[1]],slots[:,29:])
    joined=(torch.cat((first[0],second[0]),1),second[1])
    for j,(x,y) in enumerate(zip(joined,full)):base.check(f'stream/out{j}',x,y)
    weights=[torch.randn_like(x) for x in full]
    ga=torch.autograd.grad(joined,b,weights);gb=torch.autograd.grad(full,a,weights)
    for j,(x,y) in enumerate(zip(ga,gb)):base.check(f'stream/grad{j}',x,y,8e-5,8e-4)
    perm=torch.tensor([2,4,0,1,3],device='cuda');inv=torch.argsort(perm)
    px=xs.copy()
    for j in (0,1,2,4):px[j]=xs[j][...,perm]
    px[6]=xs[6][:,:,perm]
    po,pf=call(px,inv[slots])
    ro,rf=call(xs,slots)
    base.check('perm/output',po,ro)
    base.check('perm/state',pf,rf[:,:,perm])
    nc=[torch.stack((x,x),-1)[...,0] for x in xs]
    no,nf=call(nc,slots)
    base.check('noncontiguous/output',no,ro);base.check('noncontiguous/state',nf,rf)
    for which in (0,1):
        a,b=base.clone_grad(xs),base.clone_grad(xs)
        go=call(a,slots)[which];ro=base.call_core(recurrent_reference,b,slots)[which]
        ga=torch.autograd.grad(go.square().sum(),a)
        gb=torch.autograd.grad(ro.square().sum(),b,allow_unused=True)
        for j,(x,y) in enumerate(zip(ga,gb)):
            base.check(f'cotangent{which}/grad{j}',x,torch.zeros_like(x) if y is None else y,8e-5,8e-4)
    print('PASS streaming, permutation, layouts, absent cotangents',flush=True)


def full_model():
    for dtype in (torch.float32,torch.bfloat16):
        xs,slots=base.model_inputs('cuda',dtype,t=35,m=8)
        xs[3]=torch.rand((*xs[3].shape,8),device='cuda')*.15+.8
        xs[3][:,0]=0
        a,b=base.clone_grad(xs),base.clone_grad(xs)
        oo,ss=base.call_model(a,slots,'all_triton')
        ro,rs=base.call_model(b,slots,'torch')
        got,ref=(oo,*ss),(ro,*rs)
        for j,(x,y) in enumerate(zip(got,ref)):base.check(f'full/{dtype}/out{j}',x,y)
        weights=[torch.randn_like(x) for x in got]
        ga=torch.autograd.grad(got,a,weights);gb=torch.autograd.grad(ref,b,weights)
        for j,(x,y) in enumerate(zip(ga,gb)):
            base.check(f'full/{dtype}/grad{j}',x.float(),y.float(),3e-3 if dtype==torch.bfloat16 else 2e-4,
                       1e-2 if dtype==torch.bfloat16 else 1e-3)
    # Vector alpha is larger than beta and slots: last-element validation checks
    # the distinct length contract, including slot/beta errors beyond one block.
    xs,slots=base.model_inputs('cuda',t=257,m=8)
    xs[3]=torch.full((*xs[3].shape,8),.98,device='cuda')
    base.call_model(xs,slots,'all_triton')
    for name in ('alpha_nan','alpha_range','beta_range','slot_range','shape','legacy'):
        ys=[x.clone() for x in xs];ss=slots.clone();backend='all_triton'
        if name=='alpha_nan':ys[3].reshape(-1)[-1]=float('nan')
        if name=='alpha_range':ys[3].reshape(-1)[-1]=1.1
        if name=='beta_range':ys[4].reshape(-1)[-1]=1.1
        if name=='slot_range':ss.reshape(-1)[-1]=8
        if name=='shape':ys[3]=ys[3][...,:7]
        if name=='legacy':backend='chunk'
        try:base.call_model(ys,ss,backend)
        except ValueError:REPORT['checks'].append({'name':'rejected/'+name,'passed':True})
        else:raise AssertionError('accepted '+name)
    print('PASS full model and input contracts',flush=True)


def benchmark():
    rows=[]
    for shape in ((64,192,2,32,64),(2,2048,4,32,64)):
        xs,slots=base.core_inputs(shape,'cuda',731)
        for name in ('scalar','slot_broadcast','slot_random'):
            data=[x.clone() for x in xs]
            if name=='slot_broadcast':data[4]=xs[4][...,None].expand(*xs[4].shape,shape[3]).contiguous()
            if name=='slot_random':data[4]=torch.rand((*xs[4].shape,shape[3]),device='cuda')*.15+.8
            row={'shape':shape,'variant':name}
            with torch.no_grad():
                fn=lambda:call(data,slots)
                for _ in range(3):fn()
                row['forward_graph_ms']=triton.testing.do_bench_cudagraph(fn,rep=80)
                row['forward_eager_ms']=triton.testing.do_bench(fn,warmup=20,rep=80)
            data=base.clone_grad(data)
            def train():
                out=call(data,slots)
                return torch.autograd.grad(out,data,[torch.ones_like(x) for x in out])
            for _ in range(3):train()
            row['forward_backward_graph_ms']=triton.testing.do_bench_cudagraph(train,rep=80)
            row['forward_backward_eager_ms']=triton.testing.do_bench(train,warmup=20,rep=80)
            rows.append(row);print('BENCH',json.dumps(row),flush=True)
    REPORT['benchmark']=rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--benchmark',action='store_true')
    p.add_argument('--core',choices=('fast','v1'),default='fast');p.add_argument('--output',default=None);args=p.parse_args()
    global CORE
    CORE=fast_chunk if args.core=='fast' else triton_chunk
    REPORT['core']=args.core
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.utils.deterministic.fill_uninitialized_memory=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    REPORT['environment']={'torch':str(torch.__version__),'triton':triton.__version__,
                           'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda}
    files=[Path(__file__),*[ROOT/'komp_slot'/f for f in ('triton_chunk.py','fast_chunk.py','scan.py','geometry_chunk.py','operator.py','utils.py','reference.py')]]
    REPORT['source_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    started=time.monotonic()
    try:
        core();scalar_equivalence();finite_difference_and_order();streaming_and_permutation();full_model()
        if args.benchmark:
            torch.use_deterministic_algorithms(False)
            torch.utils.deterministic.fill_uninitialized_memory=False
            benchmark()
        REPORT['status']='passed'
    except Exception as exc:
        REPORT['status']='failed';REPORT['error']=repr(exc);raise
    finally:
        REPORT['elapsed_seconds']=time.monotonic()-started
        dest=Path(args.output) if args.output else ROOT/'evidence/slot-decay-gpu.json';dest.parent.mkdir(exist_ok=True)
        dest.write_text(json.dumps(REPORT,indent=2)+'\n')
        print('REPORT',REPORT['status'],len(REPORT['checks']),flush=True)


if __name__=='__main__':main()
