"""Feature-level protected KAPA timings; excludes Gaussian/Gram adapter."""
import json
from pathlib import Path
import torch
from validate_kapa import inputs
from komp_slot.kapa import protected_kapa
from komp_bench.timing import graph_ms

dest=Path('evidence/kapa-benchmark.json');report={'rows':[],'scope':'Feature core including direction solve and all gradients; excludes Gaussian adapter; serial value recurrence, no chunk optimization.'}
for cfg in ((2,256,4,16,32),(2,2048,4,32,64)):
    x=inputs(*cfg,torch.bfloat16)
    for anchors in (0,2,4):
        row=dict(config=cfg,anchors=anchors)
        try:
            leaves=[a.detach().requires_grad_() for a in x]
            fn=lambda:protected_kapa(*leaves,anchors=anchors)
            with torch.no_grad():
                row['fwd_ms']=graph_ms(fn,unroll=1,replays=3)
                weights=[torch.randn_like(a)*.01 for a in fn()]
            backward=lambda:torch.autograd.grad(fn(),leaves,weights)
            backward();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
            backward();torch.cuda.synchronize();row['extra_train_gib']=(torch.cuda.max_memory_allocated()-base)/2**30
            row['fwd_bwd_ms']=graph_ms(backward,unroll=1,replays=3);row['status']='passed'
        except Exception as exc:row['status']='failed';row['error']=repr(exc);raise
        finally:report['rows'].append(row);dest.write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
report['status']='passed';dest.write_text(json.dumps(report,indent=2))
