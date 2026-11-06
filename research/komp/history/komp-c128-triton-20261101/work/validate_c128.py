"""C128 fwd/bwd against FP64 token/flex references; causality, tail, slot gates and old-size regression."""
import argparse,hashlib,importlib.util,json,platform,time
from pathlib import Path
import torch,triton
import komp_slot.block_fast as fast
from komp_slot.block_reference import block_komp_reference
from komp_slot.block_flex import block_komp_flex
from validate_block_komp import inputs,rms
R=Path(__file__).parent.parent
REPORT=dict(checks=[],cases=[],environment=dict(torch=str(torch.__version__),triton=triton.__version__,python=platform.python_version(),gpu=torch.cuda.get_device_name()))

def check(name,a,b,tol):
    err=rms(a,b)
    REPORT['checks'].append(dict(name=name,rms_relative=err,max_abs=float((a.double()-b.double()).abs().max()),tolerance=tol))
    assert bool(torch.isfinite(a).all()) and err<=tol,(name,err,tol)

def run(fn,x,c=128,kw=None):
    o,(k,v)=fn(*x[:5],(x[5],x[6]),chunk_size=c,**(kw or {}));return o,k,v

def case(shape,dtype,slot,evict,seed,near=False):
    b,t,h,dk,dv,m=shape
    x=inputs(b,t,h,dk,dv,m,dtype,seed,slot=slot,zero=seed%2==0)
    if near:
        center=torch.randn(1,1,h,dk,device='cuda')
        x[1]=(center+.02*x[1].float()).to(dtype);x[5]=center.transpose(1,2)+.02*x[5]
    kw={}
    if evict=='learned':
        kw=dict(evict=evict,scores=torch.randint(0,3,(b,t,h),device='cuda').float(),initial_scores=torch.randint(0,3,(b,h,m),device='cuda').float())
    a=[v.detach().clone().requires_grad_() for v in x];z=[v.detach().double().requires_grad_() for v in x]
    ref_fn=block_komp_flex if evict=='learned' else block_komp_reference
    ref_kw={k:v.double() if isinstance(v,torch.Tensor) else v for k,v in kw.items()}
    actual=run(fast.block_komp_fast,a,kw=kw);ref=run(ref_fn,z,kw=ref_kw)
    weights=[torch.randn_like(v) for v in actual]
    ga=torch.autograd.grad(actual,a,weights);gz=torch.autograd.grad(ref,z,[v.double() for v in weights])
    tol=.005 if dtype==torch.float32 else .01
    tag=f'{shape}/{dtype}/slot{slot}/{evict}/near{near}'
    for name,v,w in zip(('out','keys','values','dq','dk','dv','dg','dbeta','dk0','dv0'),(*actual,*ga),(*ref,*gz)):
        check(tag+'/'+name,v,w,tol)
    REPORT['cases'].append(tag)
    (R/'evidence/c128-progress.json').write_text(json.dumps(REPORT,indent=2))
    print('PASS',tag,flush=True)

def causal():
    for evict in ('fifo','learned'):
        for dtype in (torch.float32,torch.bfloat16):
            x=inputs(1,385,2,64,64,32,dtype,131,slot=True)
            kw=dict(evict='learned',scores=torch.randn(1,385,2,device='cuda'),initial_scores=torch.zeros(1,2,32,device='cuda')) if evict=='learned' else {}
            baseline=run(fast.block_komp_fast,x,kw=kw)[0]
            for cut in (31,63,127,128,129,255,256,257):
                altered=[a.clone() for a in x]
                for a in altered[:3]:a[:,cut:]+=1
                altered[3][:,cut:]-=.1;altered[4][:,cut:]*=.5
                changed=dict(kw)
                if kw:changed['scores']=kw['scores'].clone();changed['scores'][:,cut:]+=100
                y=run(fast.block_komp_fast,altered,kw=changed)[0]
                check(f'causal/{evict}/{dtype}/{cut}',y[:,:cut],baseline[:,:cut],0.)
            a=[v.detach().clone().requires_grad_() for v in x]
            o=run(fast.block_komp_fast,a,kw=kw)[0]
            grad=torch.autograd.grad(o[:,127].float().square().sum(),a[:5])
            for name,g in zip(('q','k','v','g','beta'),grad):
                assert not bool(torch.count_nonzero(g[:,128:])),(evict,name)
            REPORT['checks'].append(dict(name=f'future_gradient/{evict}/{dtype}',passed=True))
    for slot in (False,True):
        x=inputs(1,385,2,64,64,32,torch.float32,171,slot=slot)
        full=run(fast.block_komp_fast,x)
        a=fast.block_komp_fast(*[v[:,:256] for v in x[:5]],(x[5],x[6]),chunk_size=128)
        b=fast.block_komp_fast(*[v[:,256:] for v in x[:5]],a[1],chunk_size=128)
        check(f'stream/{slot}/out',torch.cat((a[0],b[0]),1),full[0],.001)
        check(f'stream/{slot}/values',b[1][1],full[2],.001)
    print('PASS causality and streaming',flush=True)

def legacy():
    spec=importlib.util.spec_from_file_location('komp_slot.before',R/'evidence/block_fast.before.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    for c in (16,32,64):
        for slot in (False,True):
            x=inputs(1,200,2,64,64,32,torch.bfloat16,621,slot=slot)
            a=[v.detach().clone().requires_grad_() for v in x];z=[v.detach().clone().requires_grad_() for v in x]
            o=run(fast.block_komp_fast,a,c);r=run(old.block_komp_fast,z,c)
            w=[torch.randn_like(v) for v in o]
            g=torch.autograd.grad(o,a,w);gr=torch.autograd.grad(r,z,w)
            for i,(u,v) in enumerate(zip((*o,*g),(*r,*gr))):check(f'legacy/{c}/{slot}/{i}',u,v,0.)
    print('PASS legacy bitwise',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output',default='c128-validation.json');args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    shapes=[(1,257,2,64,64,32)] if args.smoke else [(1,1,1,8,8,4),(1,127,2,33,17,5),(1,128,2,64,64,32),(2,129,2,32,48,48),(1,257,2,64,64,64),(1,385,2,128,128,32)]
    try:
        for i,shape in enumerate(shapes):
            for slot in (False,True):
                for dtype in (torch.float32,torch.bfloat16):
                    for evict in (('fifo',) if args.smoke else ('fifo','learned')):
                        case(shape,dtype,slot,evict,43+i)
        if not args.smoke:
            for slot in (False,True):case((1,257,2,64,64,32),torch.float32,slot,'fifo',177,True)
            causal();legacy()
        REPORT['status']='passed'
    except BaseException as exc:REPORT.update(status='failed',error=repr(exc));raise
    finally:
        REPORT['sources']={str(p.relative_to(R/'work')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'work/komp_slot').glob('*.py')}
        (R/'evidence'/args.output).write_text(json.dumps(REPORT,indent=2));print('REPORT',REPORT['status'],len(REPORT['checks']),flush=True)
