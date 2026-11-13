"""Bounded C128 launch-geometry sweep on one GPU, keeping all arithmetic flags fixed."""
import json,runpy,statistics,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
ns=runpy.run_path(str(ROOT/'probe.py'))
op,step,args=ns['op'],ns['step'],ns['args']
base=dict(op.WARPS)
base_key_block=op.KEY_BLOCK
base_mapgrad_bv=op.MAPGRAD_BV
results=[]
# C128 has twice as many token rows as the historical tuning; extra warps can bound register pressure.
configs=[('base',{}),('ut8',{'ut_slot':8}),('compose8',{'compose':8}),
         ('projback8',{'proj_bwd_ix':8}),('back8',{'ut_bwd':8}),
         ('keygrad8',{'keygrad':8}),('keyblock32',{'KEY_BLOCK':32}),
         ('keyblock16',{'KEY_BLOCK':16}),('mapgrad64',{'MAPGRAD_BV':64})]
for name,overrides in configs:
    op.WARPS.update(base)
    op.KEY_BLOCK=overrides.get('KEY_BLOCK',base_key_block)
    op.MAPGRAD_BV=overrides.get('MAPGRAD_BV',base_mapgrad_bv)
    op.WARPS.update({k:v for k,v in overrides.items() if k in base})
    for _ in range(2):step()
    torch.cuda.synchronize();times=[]
    for _ in range(12):
        start=time.perf_counter();value=step();torch.cuda.synchronize();times.append(time.perf_counter()-start)
    row=dict(name=name,overrides=overrides,seconds=times,median_seconds=statistics.median(times),value=value)
    results.append(row)
    (ROOT/'evidence'/f'{args.tag}-sweep.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(row),flush=True)
op.WARPS.update(base)
