"""Exact route-scan acceptance, geometry gradients, and isolated speed measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import torch
import triton
from komp_slot.geometry import routes_kernel,routes_parallel_into,triton_geometry,triton_geometry_serial,triton_geometry_parallel
from validate_geometry_fast import inputs,clone,call,check

ROOT=Path(__file__).resolve().parent


def route_outputs(slots,m):
    b,t,h=slots.shape
    return (torch.empty((b,t,h,m),device=slots.device,dtype=torch.int32),
            torch.empty((b,t,h),device=slots.device,dtype=torch.int32),
            torch.empty((b,h,m),device=slots.device,dtype=torch.int32))


def sequential(slots,m,out):
    b,t,h=slots.shape
    routes_kernel[(b*h,)](slots,*out,t,h,m,triton.next_power_of_2(m))


def oracle(slots,m):
    b,t,h=slots.shape;src=torch.arange(m).expand(b,h,m).clone();history=[];removed=[]
    for i in range(t):
        selected=slots[:,i,:,None]
        removed.append(src.gather(-1,selected).squeeze(-1).clone())
        src=src.scatter(-1,selected,m+i)
        history.append(src.clone())
    return torch.stack(history,1),torch.stack(removed,1),src


def timed(fn):
    fn();torch.cuda.synchronize()
    values=[float(triton.testing.do_bench_cudagraph(fn,rep=80)) for _ in range(3)]
    return {'graph_ms':statistics.median(values),'trials_ms':values}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'evidence/routes-parallel.json'));p.add_argument('--benchmark',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    report={'checks':[],'routes':[],'benchmarks':[],'environment':{'torch':str(torch.__version__),'triton':triton.__version__,'gpu':torch.cuda.get_device_name()},
            'geometry_sha256':hashlib.sha256((ROOT/'komp_slot/geometry.py').read_bytes()).hexdigest()}
    try:
        for t,m in [(1,1),(127,5),(128,16),(129,32),(257,64),(4097,32)]:
            for pattern in ('random','repeated','cyclic'):
                torch.manual_seed(8361+t+m)
                slots=torch.randint(m,(2,t,3),device='cuda')
                if pattern=='repeated':slots.fill_(m-1)
                if pattern=='cyclic':slots.copy_(torch.arange(t,device='cuda')[None,:,None]%m)
                a=route_outputs(slots,m);b=route_outputs(slots,m)
                sequential(slots,m,a);routes_parallel_into(slots,*b,m)
                cpu=oracle(slots.cpu(),m)
                for name,x,y,z in zip(('source','removed','last'),a,b,cpu):
                    assert torch.equal(x,y),(t,m,pattern,name,'sequential')
                    assert torch.equal(y.cpu().long(),z),(t,m,pattern,name,'oracle')
                report['routes'].append({'T':t,'M':m,'pattern':pattern,'exact':True})
            print('PASS exact routes',t,m,flush=True)
        for t,m in [(1,1),(129,5),(257,32),(65,64)]:
            xs,slots=inputs(t,m)
            a=clone(xs);b=clone(xs)
            out=call(triton_geometry_parallel,a,slots);ref=call(triton_geometry_serial,b,slots)
            weights=[torch.randn_like(x) for x in out]
            ga=torch.autograd.grad(out,a,weights);gb=torch.autograd.grad(ref,b,weights)
            for i,(x,y) in enumerate(zip((*out,*ga),(*ref,*gb))):check(f'parallel_route/{t}/{m}/{i}',x,y,report)
            print('PASS geometry gradients',t,m,flush=True)
        # Lock the auto-dispatch boundary and both sides of the slot cutoff.
        for t,m in [(191,16),(192,16),(192,15),(257,32)]:
            xs,slots=inputs(t,m)
            a=clone(xs);b=clone(xs)
            out=call(triton_geometry,a,slots);ref=call(triton_geometry_serial,b,slots)
            weights=[torch.randn_like(x) for x in out]
            ga=torch.autograd.grad(out,a,weights);gb=torch.autograd.grad(ref,b,weights)
            for i,(x,y) in enumerate(zip((*out,*ga),(*ref,*gb))):check(f'auto/{t}/{m}/{i}',x,y,report)
            print('PASS auto threshold',t,m,flush=True)
        # Full T257 uses parallel history; split129/128 uses serial on each part.
        # Compare both values and all input gradients against unsplit serial.
        xs,slots=inputs(257,32)
        a=clone(xs);b=clone(xs)
        ref=call(triton_geometry_serial,b,slots)
        first=triton_geometry(a[0][:,:129],a[1][:,:129],slots[:,:129],a[2])
        second=triton_geometry(a[0][:,129:],a[1][:,129:],slots[:,129:],first[3])
        joined=tuple(torch.cat((first[i],second[i]),1) for i in range(3))+(second[3],)
        weights=[torch.randn_like(x) for x in joined]
        ga=torch.autograd.grad(joined,a,weights);gb=torch.autograd.grad(ref,b,weights)
        for i,(x,y) in enumerate(zip((*joined,*ga),(*ref,*gb))):check(f'auto/stream257/{i}',x,y,report)
        print('PASS auto streaming split across route-dispatch threshold',flush=True)
        if args.benchmark:
            for b,t,h,m in [(2,192,4,32),(2,1024,4,16),(2,2048,4,32),(2,4096,4,32),(64,192,2,32)]:
                slots=torch.randint(m,(b,t,h),device='cuda')
                out=route_outputs(slots,m)
                row={'B':b,'T':t,'H':h,'M':m}
                row['sequential']=timed(lambda:sequential(slots,m,out))
                row['parallel']=timed(lambda:routes_parallel_into(slots,*out,m))
                row['speedup']=row['sequential']['graph_ms']/row['parallel']['graph_ms']
                report['benchmarks'].append(row);print('BENCH',json.dumps(row),flush=True)
        report['status']='passed'
    except Exception as e:report['status']='failed';report['error']=repr(e);raise
    finally:
        dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2)+'\n')
        print('REPORT',report['status'],flush=True)


if __name__=='__main__':main()
