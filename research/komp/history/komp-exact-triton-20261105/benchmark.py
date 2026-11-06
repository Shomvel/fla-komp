"""Same-device CUDA-event timings; compilation excluded, no quality claims."""
import argparse,json,time
from pathlib import Path
import torch
from exact_komp import exact_komp
from validate_gpu import make,invoke


def timing(fn):
    for _ in range(2):fn()
    torch.cuda.synchronize()
    values=[]
    for _ in range(5):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();fn();b.record();b.synchronize();values.append(a.elapsed_time(b))
    return sorted(values)[len(values)//2]


def main():
    rows=[]
    for learned in [False,True]:
        for t in [128,512]:
            data=make(91,1,t,2,32,64,64,torch.float32)
            data=tuple(x.requires_grad_(i<8) for i,x in enumerate(data))
            for chunk in [16,32]:
                def forward():return invoke(data,learned,learned,chunk)
                def both():
                    y,(_,s),_=forward()
                    torch.autograd.grad(y.square().mean()+s.square().mean(),data[:8],allow_unused=True)
                fw=timing(forward);fb=timing(both)
                torch.cuda.reset_peak_memory_stats()
                before=torch.cuda.memory_allocated();both();torch.cuda.synchronize()
                extra=(torch.cuda.max_memory_allocated()-before)/2**20
                row={'B':1,'T':t,'H':2,'M':32,'K':64,'V':64,'learned':learned,
                     'compute_chunk':chunk,'dtype':'float32','projection_dtype':'float64',
                     'forward_ms':fw,'forward_backward_ms':fb,'extra_peak_MiB':extra}
                rows.append(row);print(json.dumps(row),flush=True)
    Path('evidence/benchmark.json').write_text(json.dumps({'gpu':torch.cuda.get_device_name(),'rows':rows},indent=2))


if __name__=='__main__':main()
