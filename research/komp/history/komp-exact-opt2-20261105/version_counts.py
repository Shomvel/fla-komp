"""Count factorization work for the exact seeded benchmark dictionary trajectory."""
import json
from pathlib import Path
import torch
from compare import inputs
from exact_komp import geometry as geo

xs=inputs(8192,torch.bfloat16)
bh,t,m=2,8192,32
scores=xs[8].transpose(1,2).contiguous().reshape(bh,t)
initial=torch.zeros(bh,m,device='cuda')
birth=torch.arange(-m,0,device='cuda',dtype=torch.int64)[None].expand(bh,m).contiguous()
ids=torch.empty(bh,t+1,m,device='cuda',dtype=torch.int32)
drop=torch.empty(bh,t,device='cuda',dtype=torch.int32)
sf=torch.empty_like(initial);bf=torch.empty_like(birth)
geo.select[(bh,)](scores,initial,birth,ids,drop,sf,bf,t,m,32,0,64,num_warps=4)
d=drop.cpu();sparse=0;dense=0;factors=0
for h in range(bh):
    for start in range(0,t,32):
        block=d[h,start:start+32];n=int((block>=0).sum())
        if n<=4:
            sparse+=1;factors+=1+int((block[1:]>=0).sum())
        else:
            dense+=1;factors+=len(block)
result={'B':1,'H':bh,'T':t,'M':m,'accepted_tokens':int((d>=0).sum()),
        'sparse_windows':sparse,'dense_windows':dense,'factorizations_per_pass':factors,
        'previous_factorizations_per_pass':bh*t,'fraction_saved':1-factors/(bh*t),
        'note':'FP32 forward and backward each rebuild factors. Reuse is bounded to 32-token windows.'}
Path('evidence/version-counts.json').write_text(json.dumps(result,indent=2));print(result)
