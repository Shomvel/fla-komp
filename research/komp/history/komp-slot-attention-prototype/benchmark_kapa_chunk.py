"""Same-H100 serial vs chunk KAPA, raw-key full forward/backward CUDA graphs."""
import json
from pathlib import Path
import torch
from validate_kapa_geometry import raw_inputs
from komp_slot.kapa import gaussian_protected_kapa
from komp_slot.kapa_chunk import chunk_gaussian_protected_kapa
from komp_bench.timing import graph_ms

report={'rows':[],'scope':'Same raw BF16 q/k/v/centers; FP32 gates/state; p=2. Includes Gaussian, inverse Gram, direction solve, recurrence and all input gradients. CUDA graph replay.'}
dest=Path('evidence/kapa-chunk-benchmark.json')
for cfg in ((2,256,4,32,16,32),(2,2048,4,64,32,64),(64,192,8,64,32,64)):
    for slot in (False,True):
        torch.manual_seed(127)
        raw=raw_inputs(*cfg,torch.bfloat16,slot=slot)
        for c in (0,16,32,64):
            leaves=[x.detach().requires_grad_() for x in raw]
            fn=lambda:chunk_gaussian_protected_kapa(*leaves,chunk_size=c) if c else gaussian_protected_kapa(*leaves)
            row=dict(config=cfg,slot_decay=slot,chunk_size=c,mode='chunk' if c else 'serial')
            try:
                with torch.no_grad():
                    row['fwd_ms']=graph_ms(fn,unroll=1,replays=5)
                    weights=[torch.randn_like(x)*.01 for x in fn()]
                train=lambda:torch.autograd.grad(fn(),leaves,weights)
                train();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
                train();torch.cuda.synchronize();row['extra_train_gib']=(torch.cuda.max_memory_allocated()-base)/2**30
                row['fwd_bwd_ms']=graph_ms(train,unroll=1,replays=5);row['status']='passed'
            except Exception as exc:row['status']='failed';row['error']=repr(exc);raise
            finally:report['rows'].append(row);dest.write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
report['status']='passed';dest.write_text(json.dumps(report,indent=2))
