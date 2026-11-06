"""Same-GPU baseline/gather/TF32x3 geometry timing, including complete recurrence."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import torch
import triton
from validate_geometry_fast import inputs,clone,call,MODES
from komp_slot.triton_chunk import triton_chunk
from komp_slot.geometry import triton_geometry_parallel

ROOT=Path(__file__).resolve().parent


def timeit(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize()
    eager=[float(triton.testing.do_bench(fn,warmup=20,rep=80)) for _ in range(3)]
    graph=[float(triton.testing.do_bench_cudagraph(fn,rep=80)) for _ in range(3)]
    return {'eager_ms':statistics.median(eager),'graph_ms':statistics.median(graph),
            'eager_trials_ms':eager,'graph_trials_ms':graph}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'evidence/geometry-fast-benchmark.json'));p.add_argument('--quick',action='store_true');p.add_argument('--parallel',action='store_true');args=p.parse_args()
    modes=({'baseline':MODES['baseline'],'parallel_ieee':triton_geometry_parallel,
            'parallel_gather_tf32x3':lambda *a:triton_geometry_parallel(*a,gather=True,dot_precision='tf32x3')} if args.parallel else MODES)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rows=[];report={'rows':rows,'environment':{'torch':str(torch.__version__),'triton':triton.__version__,'gpu':torch.cuda.get_device_name()},
        'geometry_sha256':hashlib.sha256((ROOT/'komp_slot/geometry.py').read_bytes()).hexdigest(),
        'method':'Median of 3, eager warmup20 rep80ms and CUDA Graph rep80ms; full includes all geometry and chunk recurrence'}
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    def save():dest.write_text(json.dumps(report,indent=2)+'\n')
    cases=[(256,32),(2048,32)] if args.quick else [(192,32),(1024,16),(1024,32),(2048,32),(1024,64)]
    try:
        for t,m in cases:
            xs,slots=inputs(t,m,dtype=torch.bfloat16,b=2,h=4)
            row={'B':2,'T':t,'H':4,'D':64,'M':m,'input_dtype':'bfloat16','modes':{}}
            for mode,fn in modes.items():
                print('START',t,m,mode,flush=True)
                with torch.no_grad():
                    gout=call(fn,xs,slots)
                    assert all(bool(torch.isfinite(o).all()) for o in gout)
                    gf=timeit(lambda:call(fn,xs,slots))
                gx=clone(xs);gw=[torch.randn_like(o)*.01 for o in gout]
                def geometry_train():return torch.autograd.grad(call(fn,gx,slots),gx,gw)
                gb=timeit(geometry_train)
                value=torch.randn(2,t,4,64,device='cuda',dtype=torch.bfloat16)*.2
                alpha=torch.full((2,t,4),.98,device='cuda');beta=torch.full_like(alpha,.2)
                initial=torch.randn(2,4,m,64,device='cuda')*.05
                fx=list(xs)+[value,alpha,beta,initial]
                def full(z):
                    h,p,r,k=call(fn,z[:3],slots)
                    out,s=triton_chunk(h,p,r,z[3],z[4],z[5],slots,z[6])
                    return out,k,s
                with torch.no_grad():ff=timeit(lambda:full(fx));fout=full(fx)
                gradx=clone(fx);fw=[torch.randn_like(o)*.01 for o in fout]
                def full_train():return torch.autograd.grad(full(gradx),gradx,fw)
                fb=timeit(full_train)
                row['modes'][mode]={'geometry_forward':gf,'geometry_forward_backward':gb,'full_forward':ff,'full_forward_backward':fb}
                print('RESULT',t,m,mode,json.dumps(row['modes'][mode]),flush=True)
            rows.append(row);save()
        report['status']='passed'
    except Exception as e:report['status']='failed';report['error']=repr(e);raise
    finally:save()


if __name__=='__main__':main()
