"""Small prototype timings, not claims of chunk-parallel throughput."""
import json
from pathlib import Path
import torch
from validate_bsgd import inputs,run
from komp_slot.bsgd import bsgd_merging
from komp_slot.block_fast import block_komp_fast
from komp_bench.timing import graph_ms

dest=Path('evidence/bsgd-benchmark.json')
report={'rows':[],'note':'BF16 q/k/v; FP32 state/gates; CUDA graph; BSGD is token-recursive, block KOMP is chunked and semantically different.'}
for cfg in ((2,256,4,32,32,16),(2,2048,4,64,64,32)):
    x=inputs(*cfg,torch.bfloat16)
    for name,kw in [('gated',dict(mode='gated')),('grid_cyclic',dict(mode='grid')),('grid_search',dict(mode='grid',routing='min_error')),('block_c64',{})]:
        leaves=[a.detach().requires_grad_() for a in x]
        def fn():
            if name=='block_c64':
                o,(k,v)=block_komp_fast(*leaves[:5],(leaves[6],leaves[7]),chunk_size=64)
                return o,k,v
            return run(bsgd_merging,leaves,**kw)
        row={'config':cfg,'mode':name}
        try:
            with torch.no_grad():
                row['fwd_ms']=graph_ms(fn,unroll=1,replays=3)
                weights=[torch.randn_like(a)*.01 for a in fn()]
            active=[a for i,a in enumerate(leaves) if not (name=='block_c64' and i==5)]
            backward=lambda:torch.autograd.grad(fn(),active,weights)
            backward();torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
            backward();torch.cuda.synchronize()
            row['extra_train_gib']=(torch.cuda.max_memory_allocated()-base)/2**30
            row['fwd_bwd_ms']=graph_ms(backward,unroll=1,replays=3)
            row['status']='passed'
        except Exception as e:
            row['status']='failed';row['error']=repr(e);raise
        finally:
            report['rows'].append(row);dest.write_text(json.dumps(report,indent=2))
            print(json.dumps(row),flush=True)
report['status']='passed';dest.write_text(json.dumps(report,indent=2))
