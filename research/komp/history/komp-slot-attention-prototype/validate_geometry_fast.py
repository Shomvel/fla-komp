"""Opt-in geometry acceptance against unchanged baseline and FP64/FP32 oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import torch
import triton
from komp_slot.geometry import triton_geometry_serial,triton_geometry_fast
from komp_slot.reference import prepare_geometry

ROOT=Path(__file__).resolve().parent
MODES={'baseline':triton_geometry_serial,
       'gather_ieee':lambda *a,**kw:triton_geometry_fast(*a,**kw,dot_precision='ieee'),
       'gather_tf32x3':triton_geometry_fast}


def inputs(t,m,d=64,dtype=torch.float32,b=1,h=2,case='random'):
    torch.manual_seed(1731+t+m)
    q=torch.randn(b,t,h,d,device='cuda',dtype=dtype)
    k=torch.randn_like(q)
    initial=torch.randn(b,h,m,d,device='cuda',dtype=dtype)
    if case=='near_duplicate':
        center=torch.randn(b,h,1,d,device='cuda')
        initial=(center+0.02*initial.float()).to(dtype)
        k=(center.transpose(1,2)+0.02*k.float()).to(dtype)
    if case=='zero':q.zero_();k.zero_();initial.zero_()
    if case=='noncontiguous':
        q=q.transpose(1,2).contiguous().transpose(1,2)
        k=k.transpose(1,2).contiguous().transpose(1,2)
        initial=initial.transpose(-1,-2).contiguous().transpose(-1,-2)
    slots=torch.randint(m,(b,t,h),device='cuda')
    if case=='repeated':slots.fill_(m-1)
    return (q,k,initial),slots


def clone(xs,dtype=None):
    return [x.detach().to(dtype or x.dtype).clone().requires_grad_() for x in xs]


def call(fn,xs,slots):return fn(xs[0],xs[1],slots,xs[2])


def errors(got,ref):
    x,y=got.double(),ref.double()
    return {'max_abs':float((x-y).abs().max()),'rms_relative':float((x-y).square().mean().sqrt()/(y.square().mean().sqrt()+1e-12))}


def check(tag,got,ref,report,atol=2e-4,rtol=1e-3):
    assert bool(torch.isfinite(got).all()),tag
    report['checks'].append({'name':tag,**errors(got,ref),'atol':atol,'rtol':rtol})
    torch.testing.assert_close(got.float(),ref.float(),atol=atol,rtol=rtol,msg=tag)


def validate(report):
    cases=[(1,1,7,torch.float32,'random'),(19,5,33,torch.float32,'noncontiguous'),
           (33,16,64,torch.float32,'repeated'),(65,32,64,torch.float32,'random'),
           (17,64,64,torch.float32,'random'),(35,32,64,torch.bfloat16,'random'),
           (192,32,64,torch.float32,'random'),(7,8,16,torch.float32,'zero')]
    for t,m,d,dtype,case in cases:
        xs,slots=inputs(t,m,d,dtype,case=case)
        xr=clone(xs);ref=call(prepare_geometry,xr,slots)
        weights=[torch.randn_like(x) for x in ref]
        rg=torch.autograd.grad(ref,xr,weights)
        for mode,fn in MODES.items():
            x=clone(xs);out=call(fn,x,slots);grads=torch.autograd.grad(out,x,weights)
            tag=f'{mode}/{t}/{m}/{dtype}/{case}'
            for i,(a,b) in enumerate(zip(out,ref)):check(tag+f'/out{i}',a,b,report)
            for i,(a,b) in enumerate(zip(grads,rg)):
                check(tag+f'/grad{i}',a,b,report,3e-3 if dtype==torch.bfloat16 else 2e-4,
                      1e-2 if dtype==torch.bfloat16 else 1e-3)
        print('PASS geometry',t,m,str(dtype),case,flush=True)
    # Stress conditioning separately: compare all implementations to an FP64
    # oracle, and require each candidate not to regress the baseline error.
    xs,slots=inputs(9,16,32,case='near_duplicate')
    xr=clone(xs,torch.float64);ref=call(prepare_geometry,xr,slots)
    weights=[torch.randn_like(x).float() for x in ref]
    rg=torch.autograd.grad(ref,xr,[w.double() for w in weights])
    baseline=None
    for mode,fn in MODES.items():
        x=clone(xs);out=call(fn,x,slots);grads=torch.autograd.grad(out,x,weights)
        measured=[]
        for i,(a,b) in enumerate(zip((*out,*grads),(*ref,*rg))):
            assert bool(torch.isfinite(a).all())
            measured.append(errors(a,b))
        if mode=='baseline':baseline=measured
        else:
            for i,(err,base) in enumerate(zip(measured,baseline)):
                assert err['rms_relative']<=max(1e-3,base['rms_relative']*1.25), (mode,i,err,base)
        report['conditioning'].append({'mode':mode,'errors_vs_fp64':measured,
            'criterion':'RMS relative <= max(.001, 1.25 * baseline error); separate stress, not normal acceptance tolerance'})
    # Causal streaming across arbitrary split preserves outputs and final keys.
    xs,slots=inputs(67,32)
    for mode,fn in MODES.items():
        x=clone(xs);full=call(fn,x,slots)
        first=fn(x[0][:,:29],x[1][:,:29],slots[:,:29],x[2])
        second=fn(x[0][:,29:],x[1][:,29:],slots[:,29:],first[3])
        for i in range(3):check(mode+f'/stream/out{i}',torch.cat((first[i],second[i]),1),full[i],report)
        check(mode+'/stream/final',second[3],full[3],report)
        gf=torch.autograd.grad(sum(o.square().sum() for o in full),x,retain_graph=True)
        gs=torch.autograd.grad(sum(torch.cat((first[i],second[i]),1).square().sum() for i in range(3))+second[3].square().sum(),x)
        for i,(a,b) in enumerate(zip(gf,gs)):check(mode+f'/stream/grad{i}',a,b,report)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'evidence/geometry-fast-validation.json'));args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    report={'checks':[],'conditioning':[],'environment':{'torch':str(torch.__version__),'triton':triton.__version__,
        'python':platform.python_version(),'gpu':torch.cuda.get_device_name()},
        'sources':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),ROOT/'komp_slot/geometry.py')}}
    try:validate(report);report['status']='passed'
    except Exception as e:report['status']='failed';report['error']=repr(e);raise
    finally:
        dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2)+'\n')
        print('REPORT',report['status'],len(report['checks']),flush=True)


if __name__=='__main__':main()
