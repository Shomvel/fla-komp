"""Exercise the installed project package, not the standalone import."""
import json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent/'integration'))
from komp_slot.exact_komp import exact_komp
from validate_gpu import make,reference,check,checks

data=tuple(x.requires_grad_(i<8) for i,x in enumerate(make(717,1,17,2,8,16,17,torch.float32)))
q,k,v,g,beta,k0,s0,scores,sc0=data
o,(kf,sf),_=exact_komp(q,k,v,g,beta,(k0,s0),eviction='learned',scores=scores,initial_scores=sc0)
gold=tuple(x.detach().double().requires_grad_(i<8) for i,x in enumerate(data))
yr,(kr,sr)=reference(gold,True,True)
check('package/output',o,yr);check('package/keys',kf,kr);check('package/state',sf,sr)
grad=torch.autograd.grad(o.square().mean()+sf.square().mean(),data[:8],allow_unused=True)
assert all(x is None or torch.isfinite(x).all() for x in grad)
Path('evidence/integration.json').write_text(json.dumps({'status':'passed','import':'komp_slot.exact_komp',
    'module_file':sys.modules['komp_slot.exact_komp.operator'].__file__,'checks':checks,'finite_gradients':True},indent=2))
print('Project package import, numerical output and backward passed.')
