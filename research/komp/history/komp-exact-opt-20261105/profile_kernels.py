"""Diagnostic kernel attribution, after warmup; not used as benchmark timings."""
import json
from pathlib import Path
import torch
from compare import inputs,run


rows=[]
for name in ('fifo','learned'):
    xs=inputs(8192,torch.bfloat16)
    leaves=[xs[i] for i in (0,1,2,3,4,5,6,8)]
    for chunk in (16,32):
        with torch.no_grad():
            weights=[torch.randn_like(x)*.01 for x in run(name,xs,chunk)]
        def train():
            return torch.autograd.grad(run(name,xs,chunk),leaves,weights,allow_unused=True)
        for _ in range(3):train()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(3):train()
            torch.cuda.synchronize()
        kernels=[]
        for event in prof.key_averages():
            if str(event.device_type).endswith('CUDA'):
                kernels.append({'name':event.key,'count':event.count,'total_us':event.self_device_time_total,'us_per_call':event.self_device_time_total/event.count})
        kernels.sort(key=lambda x:x['total_us'],reverse=True)
        rows.append({'operator':name,'chunk':chunk,'iterations':3,'kernels':kernels})
        print(name,chunk,json.dumps(kernels[:8]),flush=True)
Path('evidence/kernel-profile.json').write_text(json.dumps(rows,indent=2))
