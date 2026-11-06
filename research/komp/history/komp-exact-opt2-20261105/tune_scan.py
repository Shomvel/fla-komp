"""Select hierarchy granularity from same-device timings, including small scans."""
import json
from pathlib import Path
import torch
from exact_komp.hierarchical import scan
from baseline_komp.chunk_backward import boundary_scan


def timing(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):fn()
    values=[]
    for _ in range(3):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(20):graph.replay()
        b.record();b.synchronize();values.append(a.elapsed_time(b)/20)
    return sorted(values)[1]


rows=[];torch.manual_seed(42)
for bh,nc in [(2,32),(2,64),(2,512),(16,512)]:
    m=32;dv=64
    f=.96*torch.eye(m,device='cuda')[None,None]+.01*torch.randn(bh,nc,m,m,device='cuda')/m**.5
    u=torch.randn(bh,nc,m,dv,device='cuda');initial=torch.randn(bh,m,dv,device='cuda')
    out=torch.empty_like(u);final=torch.empty_like(initial)
    for group in (0,8,16,32):
        def fn():
            if group:scan(f,u,initial,out,final,reverse=True,group=group)
            else:boundary_scan[(bh,2)](f,u,initial,out,final,m,dv,nc,32,32,num_warps=4)
        fn()
        if group==0:ref=out.clone();rf=final.clone()
        else:
            torch.testing.assert_close(out,ref,atol=1e-4,rtol=1e-4)
            torch.testing.assert_close(final,rf,atol=1e-4,rtol=1e-4)
        row={'BH':bh,'NC':nc,'group':group,'ms':timing(fn)}
        rows.append(row);print(json.dumps(row),flush=True)
        Path('evidence/scan-tuning.json').write_text(json.dumps({'rows':rows},indent=2))
