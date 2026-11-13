"""Same-device native slot/scalar, hybrid value, native-reuse value comparison."""
import argparse, hashlib, json, platform, statistics, time
from pathlib import Path
import torch, triton
import validate_value_decay_triton as suite
from komp_slot.exact_komp import exact_komp
from komp_slot.block_fast import block_komp_fast
from komp_slot.value_decay_hybrid import block_komp_value_decay_triton as hybrid
from komp_slot.value_decay_triton import block_komp_value_decay_triton as native


def extended_validation():
    suite.validation()
    for c,t in ((1,33),(16,35),(32,65),(64,67),(128,257)):
        for dtype in (torch.float32,torch.bfloat16):
            xs=suite.inputs(1,t,2,64,64,32,dtype,71)
            aa=[x.detach().requires_grad_() for x in xs]
            rr=[x.detach().double().requires_grad_() for x in xs]
            a=suite.run(aa,c,'triton');r=suite.run(rr,c,'recurrent')
            tag=f'native/C{c}/T{t}/{dtype}'
            tol=.025 if dtype==torch.bfloat16 else .003
            for i,(x,y) in enumerate(zip(a,r)):suite.check(tag+f'/out{i}',x,y,rtol=tol)
            w=[torch.randn_like(x) for x in a]
            ga=torch.autograd.grad(a,aa,w);gr=torch.autograd.grad(r,rr,[x.double() for x in w])
            for i,(x,y) in enumerate(zip(ga,gr)):suite.check(tag+f'/grad{i}',x,y,rtol=tol)
            print('PASS',tag,flush=True)
    # Nontrivial gate VJP at identical channels; large heterogeneous gates must
    # fall back safely, rather than clipping or replacing the gate by its mean.
    for gate in ('same','heterogeneous'):
        xs=suite.inputs(1,19,2,64,64,32,torch.float32,81)
        xs[3].fill_(-.005)
        if gate=='heterogeneous':xs[3][...,::2]=-1000
        aa=[x.detach().requires_grad_() for x in xs];rr=[x.detach().double().requires_grad_() for x in xs]
        a=suite.run(aa,1,'triton');r=suite.run(rr,1,'recurrent');w=[torch.randn_like(x) for x in a]
        for i,(x,y) in enumerate(zip(a,r)):suite.check(gate+f'/out{i}',x,y)
        ga=torch.autograd.grad(a,aa,w);gr=torch.autograd.grad(r,rr,[x.double() for x in w])
        for i,(x,y) in enumerate(zip(ga,gr)):suite.check(gate+f'/grad{i}',x,y)
        print('PASS',gate,flush=True)


def bench():
    rows=[]
    for c in (1,128):
        for gate in ('init','random'):
            for backend in ('slot','scalar','hybrid_value','native_value'):
                torch.cuda.empty_cache()
                xs=suite.inputs(16,2048,16,64,64,32,torch.bfloat16)
                if gate=='init':xs[3].fill_(-0.005012541823544286)
                if backend=='slot':xs[3]=xs[3][...,:32].contiguous()
                if backend=='scalar':xs[3]=xs[3].mean(-1)
                xs=[x.requires_grad_() for x in xs]
                def iteration():
                    for x in xs:x.grad=None
                    if backend in ('slot','scalar'):
                        if c==1:o,(k,s),_=exact_komp(*xs[:5],(xs[5],xs[6]))
                        else:o,(k,s)=block_komp_fast(*xs[:5],(xs[5],xs[6]),chunk_size=c)
                    else:
                        fn=hybrid if backend=='hybrid_value' else native
                        o,(k,s)=fn(*xs[:5],(xs[5],xs[6]),chunk_size=c)
                    (o.float().square().mean()+s.square().mean()+k.square().mean()).backward()
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t0=time.monotonic()
                for _ in range(3):iteration()
                torch.cuda.synchronize();cold=time.monotonic()-t0
                samples=[]
                for _ in range(7):
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record();iteration();b.record();b.synchronize();samples.append(a.elapsed_time(b))
                assert all(x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in xs)
                row=dict(chunk=c,gate=gate,backend=backend,shape=[16,2048,16,64,64,32],
                         ms=samples,median_ms=statistics.median(samples),cold_seconds=cold,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                rows.append(row);print('BENCH',json.dumps(row),flush=True)
                del xs
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    import google.protobuf
    from google.protobuf.internal.api_implementation import Type
    report=dict(status='running',checks=suite.checks,torch=torch.__version__,triton=triton.__version__,
                gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,python=platform.python_version(),
                protobuf=google.protobuf.__version__,protobuf_backend=Type())
    try:
        extended_validation();report['numerical_status']='passed'
        args.output.write_text(json.dumps(report,indent=2))
        report['benchmarks']=bench();report['status']='passed'
    except Exception as e:report['status']='failed';report['error']=repr(e);raise
    finally:
        report['sources']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('komp_slot').rglob('*.py')}
        args.output.write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
