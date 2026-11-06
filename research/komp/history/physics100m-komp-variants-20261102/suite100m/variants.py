"""Three additions to the frozen 100M KOMP backbone; kernels are unchanged."""
import math
import torch
from torch import nn
from torch.nn import functional as F
import base_models as base
from komp_slot.decay import matched_dt_bias

KINDS=('komp_learned','komp_gdn_slot','komp_learned_gdn_slot')

class VariantMixer(base.Mixer):
    def __init__(self, original, kind):
        super().__init__('komp', original.window)
        self.load_state_dict(original.state_dict(), strict=True)
        self.learned=kind in ('komp_learned','komp_learned_gdn_slot')
        self.gdn=kind in ('komp_gdn_slot','komp_learned_gdn_slot')
        if self.learned:
            self.score=nn.Linear(base.D,base.H)
            nn.init.zeros_(self.score.weight);nn.init.zeros_(self.score.bias)
            # Hard selection has no gradient: initial atom scores are fixed zeros.
            self.register_buffer('initial_score',torch.zeros(base.H,base.M))
        if self.gdn:
            self.decay=nn.Linear(base.D,base.H*base.M,bias=False)
            nn.init.zeros_(self.decay.weight)
            self.A_log=nn.Parameter(torch.zeros(base.H,base.M))
            self.dt_bias=nn.Parameter(torch.full((base.H,base.M),matched_dt_bias(.98)))
            self.A_log._no_weight_decay=True;self.dt_bias._no_weight_decay=True

    def log_decay(self,x):
        with torch.autocast('cuda',enabled=False):
            z=F.linear(x.float(),self.decay.weight,self.decay.bias).reshape(*x.shape[:2],base.H,base.M)
            if self.gdn:
                # Same GDN formula as komp_slot.decay.gdn_decay, directly in log
                # space because block_fast accepts log alpha (avoid exp/log underflow).
                return -self.A_log.float().exp()*F.softplus(z+self.dt_bias.float())
            return F.logsigmoid(z)/8.

    def forward(self,x):
        b,t,_=x.shape
        q,k,v=[conv(proj(x))[0].reshape(b,t,base.H,base.K) for conv,proj in zip(self.convs,(self.q,self.k,self.v))]
        g=self.log_decay(x)
        with torch.autocast('cuda',enabled=False):
            beta=F.linear(x.float(),self.beta.weight,self.beta.bias).sigmoid()
            scores=F.linear(x.float(),self.score.weight,self.score.bias) if self.learned else None
            if self.learned:beta=beta*(2*scores.sigmoid())
        keys=self.keys[None].expand(b,base.H,base.M,base.K).contiguous()
        values=torch.zeros(b,base.H,base.M,base.K,device=x.device)
        extra=dict(evict='learned',scores=scores,initial_scores=self.initial_score[None].expand(b,base.H,base.M)) if self.learned else {}
        o,_=base.block_komp_fast(q,k,v,g,beta,(keys,values),chunk_size=128,**extra)
        y=self.norm(o.float()).to(x.dtype).reshape(b,t,base.D)
        return self.out(y*F.silu(self.gate(x)))

class LM(base.LM):
    def __init__(self,kind,seed,window=96,checkpoint_layers=True):
        assert kind in KINDS
        # Keep FFN width, all backbone weights, conv filters and initial dictionary
        # identical to the original KOMP, rather than resizing the FFN for new heads.
        super().__init__('komp',seed,window,checkpoint_layers)
        self.kind=kind
        for layer in self.layers:layer.mix=VariantMixer(layer.mix,kind)

def metadata(model):
    result=base.metadata(model)
    result.update(block_chunk=128,block_slots=base.M,eviction='learned' if model.layers[0].mix.learned else 'fifo',
                  score_write_gate=model.layers[0].mix.learned,decay='gdn_per_slot' if model.layers[0].mix.gdn else 'gsa_per_slot',
                  initial_alpha=.98,initial_beta=.2,shared_backbone='exact original KOMP',
                  decay_formula='-exp(A_log)*softplus(Wx+dt_bias)' if model.layers[0].mix.gdn else 'logsigmoid(Wx+b)/8',
                  timescale_weight_decay=0. if model.layers[0].mix.gdn else None)
    return result

optimizer=base.optimizer
