"""Independent FP64 proof of coordinate transform and all seven VJPs."""
import json
from pathlib import Path
import torch
from komp_slot.block_reference import block_komp_reference


def transform(q,k,v,g,beta,state,c):
    scalar=g.mean(-1)
    cumul=(g-scalar[...,None]).cumsum(1)
    origin=(torch.minimum(cumul.detach().amin(1),torch.zeros_like(cumul[:,0]))+
            torch.maximum(cumul.detach().amax(1),torch.zeros_like(cumul[:,0])))*.5
    scale=(cumul-origin[:,None]).exp()
    o,(kf,sf)=block_komp_reference(q,k,v/scale,scalar,beta,
        (state[0],state[1]*origin.exp()[:,:,None,:]),c)
    return o*scale,kf,sf*scale[:,-1,:,None,:]


def main():
    torch.manual_seed(42);results=[]
    for c in (1,4,16):
        for same in (False,True):
            b,t,h,d,m=1,19,2,5,3
            xs=[torch.randn(b,t,h,d,dtype=torch.float64) for _ in range(3)]
            g=-torch.rand(b,t,h,d,dtype=torch.float64)*.1
            if same:g.fill_(-.005)
            xs += [g,torch.rand(b,t,h,dtype=torch.float64),torch.randn(b,h,m,d,dtype=torch.float64),torch.randn(b,h,m,d,dtype=torch.float64)]
            xs=[x.requires_grad_() for x in xs]
            got=transform(*xs[:5],(xs[5],xs[6]),c)
            o,(k,s)=block_komp_reference(*xs[:5],(xs[5],xs[6]),c,decay_axis='value');ref=(o,k,s)
            w=[torch.randn_like(x) for x in got]
            ga=torch.autograd.grad(got,xs,w);gr=torch.autograd.grad(ref,xs,w)
            errors=[float((x-y).detach().abs().max()) for x,y in zip((*got,*ga),(*ref,*gr))]
            assert max(errors)<1e-10,errors
            results.append(dict(chunk=c,same=same,max_abs=max(errors)))
    path=Path('evidence/value-decay/coordinates.json');path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(dict(status='passed',checks=60,results=results),indent=2))
    print(path.read_text())
if __name__=='__main__':main()
