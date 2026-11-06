"""Same arithmetic, different warp layouts; reject any numerical regression."""
import json
from pathlib import Path
import torch
import triton
from exact_komp import geometry as geo
from validate_gpu import make


def timing(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):fn()
    samples=[]
    for _ in range(3):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(20):graph.replay()
        b.record();b.synchronize();samples.append(a.elapsed_time(b)/20)
    return sorted(samples)[1]


rows=[]
for m,d,t in [(32,64,8192),(8,16,513),(64,128,513)]:
    bh=2;bm=max(16,triton.next_power_of_2(m));bd=max(16,triton.next_power_of_2(d))
    data=make(123,1,t,bh,m,d,64,torch.float32,scalar=True)
    q,k=data[:2];k0=data[5].reshape(bh,m,d)
    q=q.transpose(1,2).contiguous().reshape(bh,t,d);k=k.transpose(1,2).contiguous().reshape(bh,t,d)
    alloc=lambda *shape,dt=torch.float32:torch.empty(shape,device='cuda',dtype=dt)
    qn=alloc(bh,t,d);pool=alloc(bh,t+m,d)
    ids=alloc(bh,t+1,m,dt=torch.int32);drop=alloc(bh,t,dt=torch.int32)
    sf=alloc(bh,m);bf=alloc(bh,m,dt=torch.int64)
    scores0=torch.zeros_like(sf);birth0=torch.arange(-m,0,device='cuda',dtype=torch.int64)[None].expand(bh,m).contiguous()
    geo.normalize[(t+m,bh)](q,k,k0,qn,pool,t,m,d,bd,num_warps=4,enable_fp_fusion=False)
    geo.fifo_ids[(triton.cdiv(t+1,32),bh)](scores0,birth0,ids,drop,sf,bf,t,m,bm,0,32,num_warps=4)
    p,h,r=[alloc(bh,t,m) for _ in range(3)];n=alloc(bh,t)
    dp,dh,dr=[torch.randn_like(p)*.01 for _ in range(3)];dn=torch.randn_like(n)*.01
    dqn=torch.empty_like(qn);dpool=torch.empty_like(pool)
    reference=None
    for warps in (4,1,2,8):
        def fwd():geo.forward[(t,bh)](qn,pool,ids,drop,p,h,r,n,t,m,d,bm,bd,.5,1e-4,num_warps=warps,enable_fp_fusion=False)
        def bwd():
            dpool.zero_()
            geo.backward[(t,bh)](qn,pool,ids,drop,p,h,r,n,dp,dh,dr,dn,dqn,dpool,t,m,d,bm,bd,.5,1e-4,num_warps=warps,enable_fp_fusion=False)
        fwd();bwd();torch.cuda.synchronize()
        got=[p,h,r,n,dqn,dpool]
        if reference is None:reference=[x.clone() for x in got]
        errors=[]
        for a,b in zip(got,reference):
            err=float((a-b).square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-6))
            assert torch.isfinite(a).all() and err<.001,(m,d,warps,err)
            errors.append(err)
        row={'M':m,'D':d,'T':t,'BH':bh,'warps':warps,'forward_ms':timing(fwd),'backward_ms':timing(bwd),'rms_errors':errors}
        rows.append(row);print(json.dumps(row),flush=True)
        Path('evidence/geometry-tuning.json').write_text(json.dumps({'rows':rows},indent=2))
