"""New numerical risks: scalar chunks, zero beta/norm, precision mode, and 8K adjoints."""
import json
from pathlib import Path
import torch
import validate_gpu as v
from baseline_komp import exact_komp as baseline
from exact_komp import exact_komp


def main():
    original_make=v.make
    for pattern in ('zero_beta','zero_norm','tiny_norm'):
        def make(*args, **kwargs):
            xs=list(original_make(*args, **kwargs))
            if pattern=='zero_beta':
                xs[4][:,::2]=0
            else:
                for i in (0,1,5):
                    xs[i].mul_(0 if pattern=='zero_norm' else 1e-8)
            return tuple(xs)
        v.make=make
        for learned in (False,True):
            v.case(912,1,19,1,4,8,7,torch.float32,pattern,learned,16,scalar=True)
    v.make=original_make
    for learned in (False,True):
        v.case(913,1,65,2,8,16,17,torch.float32,'strong',learned,32,scalar=True)
        # Keep FP64 geometry supported as a strict reference mode after tape changes.
        invoke=v.invoke
        v.invoke=lambda *args,**kwargs:invoke(*args,projection_dtype='float64',**kwargs)
        v.case(914,1,33,2,8,16,17,torch.float32,'near',learned,16,scalar=True)
        v.invoke=invoke
    for dtype in (torch.float32,torch.bfloat16):
        for learned in (False,True):
            data=v.make(915,1,8192,2,32,64,64,dtype,scalar=True)
            xs=tuple(x.detach().requires_grad_(i<8) for i,x in enumerate(data))
            q,k,val,g,beta,k0,s0,scores,init=xs
            kwargs=dict(eviction='learned' if learned else 'fifo',scores=scores,initial_scores=init)
            a,(ak,ass),_=exact_komp(q,k,val,g,beta,(k0,s0),compute_chunk=16,**kwargs)
            b,(bk,bs),_=baseline(q,k,val,g,beta,(k0,s0),compute_chunk=16,**kwargs)
            weights=[torch.randn_like(x)*.01 for x in (a,ak,ass)]
            da=torch.autograd.grad((a,ak,ass),xs[:8],weights,allow_unused=True)
            db=torch.autograd.grad((b,bk,bs),xs[:8],weights,allow_unused=True)
            tol=.01 if dtype==torch.bfloat16 else .005
            for i,(got,want) in enumerate(zip((a,ak,ass,*da),(b,bk,bs,*db))):
                if got is not None and want is not None:
                    v.check(f'8k/{dtype}/learned{learned}/{i}',got,want,tol,tol)
            print('PASS 8K differential',dtype,learned,flush=True)
    Path('evidence/edge.json').write_text(json.dumps({'status':'passed','check_count':len(v.checks),'checks':v.checks},indent=2))


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        Path('evidence/edge-failure.json').write_text(json.dumps({'status':'failed','error':repr(e),'checks':v.checks},indent=2))
        raise
