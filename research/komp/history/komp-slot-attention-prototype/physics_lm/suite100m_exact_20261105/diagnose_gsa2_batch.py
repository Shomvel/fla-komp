"""Read-only native batch/packing differential; never changes experiment kernels."""
import json,random,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import ddp_train as d
from models import LM
from tasks import Factory

torch.set_num_threads(2);torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
model=LM('gsa2',731,checkpoint_layers=False).cuda().train()
rows=[Factory('mano').row(random.Random(923+i)) for i in range(17)]
ids,mask=d.base.batch(rows)
valid=torch.arange(ids.shape[1])[None] < torch.tensor([len(r['input_ids']) for r in rows])[:,None]

def forward(x,dtype):
    outputs=[]
    hooks=[layer.register_forward_hook(lambda m,i,o:outputs.append(o.detach().float().cpu())) for layer in model.layers]
    with torch.no_grad(),d.amp(dtype):model(x)
    for h in hooks:h.remove()
    return outputs

def compare(a,b,which=valid):
    reports=[]
    for i,(x,y) in enumerate(zip(a,b)):
        delta=(x-y)[which];ref=x[which]
        reports.append({'layer':i,'rms_relative':(delta.square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-8)).item(),'max_abs':delta.abs().max().item()})
    return reports

report={'status':'completed','sources':d.digest(),'shape':list(ids.shape),'lengths':[len(r['input_ids']) for r in rows],'checks':{}}
for dtype in ['fp32','bf16']:
    baseline=forward(ids,dtype)
    rev=torch.arange(len(rows)-1,-1,-1,device='cuda')
    reordered=[x.flip(0) for x in forward(ids[rev],dtype)]
    changed=ids.clone();changed[1:]=1
    intervention=forward(changed,dtype)
    only_first=torch.zeros_like(valid);only_first[0]=valid[0]
    out={'permutation':compare(baseline,reordered),'other_document_intervention':compare(baseline,intervention,only_first)}
    for label,global_padding in [('partition_global_pad',True),('partition_local_pad',False)]:
        combined=[torch.zeros_like(x) for x in baseline]
        for rank in range(8):
            idx=list(range(rank,len(rows),8))
            local_ids,_=d.base.batch([rows[i] for i in idx])
            if global_padding:local_ids=torch.nn.functional.pad(local_ids,(0,ids.shape[1]-local_ids.shape[1]))
            for target,result in zip(combined,forward(local_ids,dtype)):target[idx,:result.shape[1]]=result
        out[label]=compare(baseline,combined)
    for length in [64,128]:
        padded=torch.nn.functional.pad(ids,(0,length-ids.shape[1]))
        out[f'pad_{length}']=compare(baseline,[x[:,:ids.shape[1]] for x in forward(padded,dtype)])
    report['checks'][dtype]=out
    print('LAYOUT',dtype,json.dumps(out),flush=True)
path=ROOT/'evidence/gsa2-layout-diagnostic.json';path.write_text(json.dumps(report,indent=2)+'\n')
print('DIAGNOSTIC_COMPLETE',str(path),flush=True)
