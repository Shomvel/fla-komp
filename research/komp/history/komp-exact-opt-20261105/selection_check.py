"""Bit-exact physical slot, score and birth trajectory against the original selector."""
import json
from pathlib import Path
import torch
import triton
from exact_komp import geometry as new
from baseline_komp import geometry as old


torch.manual_seed(77)
rows=[]
for t,m in [(1,1),(19,7),(65,32),(129,64),(8192,32)]:
    for pattern in ('random','ties','reject','replace'):
        bh=2;start=37
        score=torch.randn(bh,t,device='cuda')
        initial=torch.randn(bh,m,device='cuda')
        birth=torch.arange(start-m,start,device='cuda',dtype=torch.int64)[None].expand(bh,m).contiguous()
        if pattern=='ties':score.zero_();initial.zero_();birth.zero_()
        if pattern=='reject':score=-torch.arange(1,t+1,device='cuda').float()[None].expand(bh,t).contiguous();initial.zero_()
        if pattern=='replace':score=torch.arange(1,t+1,device='cuda').float()[None].expand(bh,t).contiguous();initial.zero_()
        def buffers():
            return (torch.empty(bh,t+1,m,device='cuda',dtype=torch.int32),
                    torch.empty(bh,t,device='cuda',dtype=torch.int32),
                    torch.empty(bh,m,device='cuda'),torch.empty(bh,m,device='cuda',dtype=torch.int64))
        a,b=buffers(),buffers();bm=max(16,triton.next_power_of_2(m))
        new.select[(bh,)](score,initial,birth,*a,t,m,bm,start,64,num_warps=4)
        old.select[(bh,)](score,initial,birth,*b,t,m,bm,True,start,num_warps=4)
        assert all(torch.equal(x,y) for x,y in zip(a,b)),(t,m,pattern)
        rows.append({'T':t,'M':m,'pattern':pattern,'identical':True})
        print('PASS selection',t,m,pattern,flush=True)
Path('evidence/selection.json').write_text(json.dumps({'status':'passed','cases':rows},indent=2))
