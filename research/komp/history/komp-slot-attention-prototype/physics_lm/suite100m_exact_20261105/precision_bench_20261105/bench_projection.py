"""Projection precision diagnostics and warmed operator throughput, no training."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parent
SUITE=ROOT.parent
TUNING=SUITE/'exact_tuning_20261105'
if not TUNING.exists():TUNING=SUITE.parent/'exact_tuning_20261105'
sys.path.insert(0,str(TUNING))
import tune
import numerical
import torch
from bf16_projection import exact_komp as bf16
from bf16_projection_mixed import exact_komp as mixed
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

MODES=('fp64','fp32','bf16_projection','bf16_projection_mixed')

def invoke(mode,*args,**kwargs):
    kwargs['projection_dtype']='float64' if mode=='fp64' else 'float32'
    return {'fp64':tune.ORIGINAL_EXACT,'fp32':tune.ORIGINAL_EXACT,
            'bf16_projection':bf16,'bf16_projection_mixed':mixed}[mode](*args,**kwargs)


def metric(name,actual,expected,low=False,checks=None):
    a,b=actual.detach().double(),expected.detach().double()
    valid=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    rms=float((a-b).square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-6)) if valid else None
    maximum=float((a-b).abs().max())/max(1.,float(b.abs().max())) if valid else None
    rt,mt=(.025,.03) if low else (.005,.005)
    row={'name':name,'finite':valid,'rms_relative':rms,'normalized_max':maximum,
         'rms_tolerance':rt,'max_tolerance':mt,'passed':valid and rms<=rt and maximum<=mt}
    if checks is not None:checks.append(row)
    return row


def stress(report):
    original=numerical.exact_komp
    numerical.check=metric
    for mode in MODES:
        rows=[]
        numerical.exact_komp=lambda *args,**kwargs:invoke(mode,*args,**kwargs)
        candidate=tune.configuration('baseline').copy();candidate['name']=mode
        for dtype in (torch.float32,torch.bfloat16):
            for pattern in ('random','near','duplicate','strong'):
                numerical.run_case(candidate,pattern,dtype,rows)
        report['stress'][mode]={'verdict':'PASSED' if all(r['passed'] for r in rows) else 'FAILED',
                               'passed':sum(r['passed'] for r in rows),'total':len(rows),'checks':rows}
        save(report)
        print('STRESS',mode,report['stress'][mode]['passed'],len(rows),flush=True)
    numerical.exact_komp=original


def events(fn):
    samples=[]
    for trial in range(3):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(20):result=fn()
        end.record();end.synchronize();samples.append(start.elapsed_time(end)/20)
    return {'median_ms':statistics.median(samples),'trials_ms':samples}


def measure(fn):
    for i in range(5):result=fn()
    torch.cuda.synchronize();del result;gc.collect()
    before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    result=fn();torch.cuda.synchronize()
    peak=(torch.cuda.max_memory_allocated()-before)/2**20
    del result
    eager=events(fn)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(3):result=fn()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):result=fn()
    torch.cuda.synchronize()
    for i in range(5):graph.replay()
    timing=events(graph.replay)
    del result,graph;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
    return {'eager':eager,'cuda_graph':timing,'extra_peak_MiB':peak}


def inputs(t,h):
    torch.manual_seed(981)
    rand=lambda *shape:torch.randn(*shape,device='cuda')
    q,k,v=[rand(1,t,h,64).bfloat16() for _ in range(3)]
    g=torch.full((1,t,h),-.02,device='cuda');beta=torch.full_like(g,.2)
    k0,s0,h0=rand(1,h,32,64),.05*rand(1,h,32,64),.05*rand(1,h,64,64)
    return [x.requires_grad_() for x in (q,k,v,g,beta,k0,s0,h0)]


def run(mode,xs):
    q,k,v,g,beta,k0,s0,h0=xs
    if mode=='gdn':
        return chunk_gated_delta_rule(q,k,v,g,beta,initial_state=h0,output_final_state=True,
                                     use_qk_l2norm_in_kernel=True,chunk_size=64)
    out,(kf,sf),_=invoke(mode,q,k,v,g,beta,(k0,s0),eviction='fifo',compute_chunk=16)
    return out,kf,sf


def benchmark(report):
    for t,h in ((8192,2),(1024,8)):
        xs=inputs(t,h)
        for mode in (*MODES,'gdn'):
            print('BENCH_START',t,h,mode,flush=True)
            leaves=xs[:5]+([xs[7]] if mode=='gdn' else xs[5:7])
            with torch.no_grad():outputs=run(mode,xs)
            weights=[torch.randn_like(x)*.01 for x in outputs]
            row={'mode':mode,'B':1,'T':t,'H':h,'K':64,'V':64,'M':None if mode=='gdn' else 32,'input_dtype':'bfloat16'}
            forward=lambda:run(mode,xs)
            both=lambda:torch.autograd.grad(forward(),leaves,weights)
            grads=both()
            row['finite']=all(bool(torch.isfinite(x).all()) for x in (*outputs,*grads))
            row['numerical_acceptance']=report['stress'].get(mode,{}).get('verdict','native GDN comparator')
            if mode=='fp64':
                ref_outputs=[x.detach() for x in outputs];ref_grads=[x.detach() for x in grads]
            elif mode!='gdn':
                row['8k_or_1k_difference_vs_fp64']=[metric(str(i),a,b,True) for i,(a,b) in enumerate(zip((*outputs,*grads),(*ref_outputs,*ref_grads)))]
            del outputs,grads
            if row['finite']:
                with torch.no_grad():row['forward']=measure(forward)
                row['forward_backward']=measure(both)
            else:row['timing_skipped']='Nonfinite output or gradients on timing input'
            report['benchmark'].append(row);save(report)
            print('BENCH_RESULT',json.dumps(row),flush=True)
        del xs,weights,ref_outputs,ref_grads


def save(report):
    temp=ROOT/'operator-results.tmp.json';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(ROOT/'operator-results.json')


if __name__=='__main__':
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.utils.deterministic.fill_uninitialized_memory=False
    report={'status':'running','stress':{},'benchmark':[],
            'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),
            'variants':json.loads((ROOT/'variants.json').read_text()),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'method':'BF16 q/k/v, FP32 gate/state, FIFO M32 C16. Operator only; includes normalization/final states; gradients on output and final states. 5 warmups then 3x20 event timing, compilation excluded; eager and one-call CUDA graph. GDN native BF16 H/K/V matched, state capacity K*V=4096 vs KOMP keys+coefficients 4096 per head; algorithms differ. BF16 projection variants experimental, approximate backward, never numerical acceptance by speed.'}
    try:
        stress(report);benchmark(report);report['status']='completed';save(report)
    except BaseException as exc:
        report['status']='failed';report['error']=repr(exc);save(report);raise
