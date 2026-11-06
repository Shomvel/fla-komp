"""Same-GPU gated merging: serial vs chunk; scalar/slot decay and block baseline."""
import json
from pathlib import Path
import torch
from validate_bsgd import inputs,run
from komp_slot.bsgd import bsgd_merging
from komp_slot.bsgd_chunk import chunk_bsgd_merging
from komp_slot.block_fast import block_komp_fast
from komp_bench.timing import graph_ms

dest=Path('evidence/bsgd-chunk-benchmark.json')
report={'rows':[],'note':'Same GPU, BF16 q/k/v; FP32 gate/state. Chunk BSGD is mathematically equivalent to gated serial BSGD; block KOMP is a different rule.'}
configs=((2,256,4,32,32,16),(2,2048,4,64,64,32),(64,192,8,64,64,32))
for cfg in configs:
    for slot in (False,True):
        x=inputs(*cfg,torch.bfloat16,slot=slot)
        variants=[('chunk16',16),('chunk32',32),('chunk64',64)]
        if not slot:variants=[('serial',0),*variants,('block64',64)]
        for name,c in variants:
            xx=[a.detach().requires_grad_() for a in x]
            def fn():
                if name=='serial':return run(bsgd_merging,xx)
                if name=='block64':
                    o,(k,v)=block_komp_fast(*xx[:5],(xx[6],xx[7]),chunk_size=64)
                    return o,k,v
                return run(chunk_bsgd_merging,xx,chunk_size=c)
            row=dict(config=cfg,slot_decay=slot,mode=name)
            try:
                with torch.no_grad():
                    row['fwd_ms']=graph_ms(fn,unroll=2,replays=5)
                    weights=[torch.randn_like(a)*.01 for a in fn()]
                active=[a for i,a in enumerate(xx) if not(name=='block64' and i==5)]
                backward=lambda:torch.autograd.grad(fn(),active,weights)
                backward();torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
                backward();torch.cuda.synchronize()
                row['extra_train_gib']=(torch.cuda.max_memory_allocated()-base)/2**30
                row['fwd_bwd_ms']=graph_ms(backward,unroll=2,replays=5)
                row['status']='passed'
            except Exception as exc:
                row['status']='failed';row['error']=repr(exc);raise
            finally:
                report['rows'].append(row);dest.write_text(json.dumps(report,indent=2))
                print(json.dumps(row),flush=True)
report['status']='passed';dest.write_text(json.dumps(report,indent=2))
