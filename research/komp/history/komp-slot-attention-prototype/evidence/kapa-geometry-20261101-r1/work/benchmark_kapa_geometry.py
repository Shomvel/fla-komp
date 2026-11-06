"""Raw-key KAPA timings including Gaussian features, Gram inverse, all gradients."""
import json
from pathlib import Path
import torch
from komp_slot.kapa import gaussian_protected_kapa, protected_kapa
from komp_slot.kapa_reference import gaussian_features as torch_features
from komp_slot.kapa_geometry import gaussian_features
from komp_bench.timing import graph_ms
from validate_kapa_geometry import raw_inputs

report={'rows':[],'scope':'Raw q/k/centers end-to-end including inverse Gram and every input gradient; serial recurrence.'}
dest=Path('evidence/kapa-geometry-benchmark.json')
for cfg in ((2,256,4,32,16,32),(2,2048,4,64,32,64)):
    raw=raw_inputs(*cfg,torch.bfloat16)
    for backend in ('torch_geometry','triton_geometry'):
        leaves=[x.detach().requires_grad_() for x in raw]
        def fn():
            if backend=='triton_geometry':return gaussian_protected_kapa(*leaves)
            q,k,v,g,beta,u,initial=leaves
            qf,h,p=torch_features(q,k,u)
            return protected_kapa(qf,h,v,g,beta,p,initial)
        row=dict(config=cfg,backend=backend,anchors=2)
        try:
            with torch.no_grad():
                row['fwd_ms']=graph_ms(fn,unroll=1,replays=3)
                weights=[torch.randn_like(x)*.01 for x in fn()]
            train=lambda:torch.autograd.grad(fn(),leaves,weights)
            train();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
            train();torch.cuda.synchronize();row['extra_train_gib']=(torch.cuda.max_memory_allocated()-base)/2**30
            row['fwd_bwd_ms']=graph_ms(train,unroll=1,replays=3);row['status']='passed'
        except Exception as exc:row['status']='failed';row['error']=repr(exc);raise
        finally:report['rows'].append(row);dest.write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
report['status']='passed';dest.write_text(json.dumps(report,indent=2))
