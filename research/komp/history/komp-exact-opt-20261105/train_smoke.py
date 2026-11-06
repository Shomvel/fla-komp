"""Optimizer-level CUDA differential run with real GDN-slot parameterization."""
import copy,json,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from exact_komp import exact_komp
from reference import block_komp_token_reference

torch.manual_seed(731)
torch.backends.cuda.matmul.allow_tf32=False


class TinyModel(nn.Module):
    def __init__(self,learned):
        super().__init__();self.learned=learned
        self.q=nn.Linear(16,16);self.k=nn.Linear(16,16);self.v=nn.Linear(16,16)
        self.decay=nn.Linear(16,8,bias=False)
        self.A_log=nn.Parameter(torch.zeros(2,4))
        self.dt_bias=nn.Parameter(torch.full((2,4),math.log(math.expm1(-math.log(.98)))))
        self.beta=nn.Linear(16,2);self.score=nn.Linear(16,2)
        self.keys=nn.Parameter(torch.randn(2,4,8))
        self.out=nn.Linear(16,8)
        nn.init.normal_(self.decay.weight,std=.02)
        nn.init.normal_(self.score.weight,std=.02)

    def forward(self,x,use_reference):
        b,t,_=x.shape
        q,k,v=[p(x).reshape(b,t,2,8) for p in (self.q,self.k,self.v)]
        g=-self.A_log.exp()*F.softplus(self.decay(x).reshape(b,t,2,4)+self.dt_bias)
        beta=self.beta(x).sigmoid();scores=self.score(x)
        keys=self.keys[None].expand(b,-1,-1,-1)
        state=x.new_zeros(b,2,4,8);init=x.new_zeros(b,2,4)
        if use_reference:
            be=beta.double()*(2*scores.double().sigmoid()) if self.learned else beta.double()
            o,_=block_komp_token_reference(q.double(),k.double(),v.double(),g.double(),be,
                (keys.double(),state.double()),chunk_size=1,evict='learned' if self.learned else 'fifo',
                scores=scores.detach().double(),initial_scores=init.double())
        else:
            o,_,_=exact_komp(q,k,v,g,beta,(keys,state),eviction='learned' if self.learned else 'fifo',
                scores=scores,initial_scores=init)
        return self.out(o.float().reshape(b,t,16))


def main():
    records=[]
    for learned in [False,True]:
        model=TinyModel(learned).cuda();gold=copy.deepcopy(model)
        # Timescale parameters stay outside weight decay, matching the experiment.
        def opt(m):
            named=list(m.named_parameters())
            return torch.optim.AdamW([
                {'params':[p for n,p in named if n not in ('A_log','dt_bias')],'weight_decay':.01},
                {'params':[p for n,p in named if n in ('A_log','dt_bias')],'weight_decay':0.}],lr=.003)
        oa,og=opt(model),opt(gold)
        x=torch.randn(2,17,16,device='cuda')
        target=torch.tanh(x[...,:8])
        losses=[]
        for step in range(12):
            row={'step':step,'learned':learned}
            for name,m,o,ref in [('triton',model,oa,False),('reference',gold,og,True)]:
                o.zero_grad(set_to_none=True)
                loss=F.mse_loss(m(x,ref),target);loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
                assert torch.isfinite(loss) and torch.isfinite(grad)
                o.step();row[name]=float(loss)
            assert abs(row['triton']-row['reference'])<.001, row
            diffs=[float((a-b).abs().max()) for a,b in zip(model.parameters(),gold.parameters())]
            row['parameter_max_abs']=max(diffs)
            assert row['parameter_max_abs']<.003,row
            losses.append(row);print(json.dumps(row),flush=True)
        assert losses[-1]['triton']<losses[0]['triton']
        records.extend(losses)
    result={'status':'passed','steps_per_variant':12,'variants':['fifo','learned'],
            'device':torch.cuda.get_device_name(),'records':records}
    Path('evidence/train-smoke.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
