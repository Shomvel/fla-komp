import gc,json,sys
from pathlib import Path
import torch
from models import ROOT,legacy
from data import Stream
kind=sys.argv[1];folder=Path(sys.argv[2]);torch.set_num_threads(2)
state=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);stream=Stream(json.loads((ROOT/'configs/data-manifest.json').read_text()),0);ids=stream.next()[:258].cuda()[None];gold=None;rows=[]
for repeat in range(2):
 model=legacy.LM(kind).cuda().eval();model.load_state_dict(state['model']);opt=legacy.optimizer(model);opt.load_state_dict(state['optimizer'])
 with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(ids[:,:-1],ids[:,1:])/257
 loss.backward();grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()};opt.step();delta={n:p.detach().cpu()-state['model'][n] for n,p in model.named_parameters()}
 if gold is None:gold=(grads,delta,float(loss.detach()))
 else:
  for label,got,want in [('grad',grads,gold[0]),('delta',delta,gold[1])]:
   for name,a in got.items():
    b=want[name];d=a-b;rows.append(dict(name=label+'/'+name,rms=float(d.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-7)),absolute=float(d.abs().max())))
 del model,opt,loss;gc.collect();torch.cuda.empty_cache()
result=dict(status='measured',kind=kind,shape='B1T257',worst=sorted(rows,key=lambda r:r['rms'],reverse=True)[:10],checks=rows)
(ROOT/f'evidence/old-replay-{kind}.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
