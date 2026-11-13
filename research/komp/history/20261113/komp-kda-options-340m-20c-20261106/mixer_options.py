"""Three independent KDA-inspired options around the unchanged KOMP operators."""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from legacy_models import Mixer,D,H,K,M,ALPHA


def head_log_decays(spread=4.):
    if not math.isfinite(spread) or spread < 1:
        raise ValueError('timescale_spread must be finite and >= 1')
    factors=torch.logspace(-math.log10(spread),math.log10(spread),H,dtype=torch.float64)
    rates=factors/factors.mean()*(-math.log(ALPHA))
    return -rates


class OptionalMixer(Mixer):
    def __init__(self,config):
        super().__init__(config.kind)
        self.output_sigmoid=config.output_sigmoid
        self.low_rank_gates=config.low_rank_gates
        if config.low_rank_gates:
            r=config.gate_rank
            if not isinstance(r,int) or r<1 or r>min(D,H*M):
                raise ValueError('gate_rank must be an integer in [1,512]')
            self.gate=nn.Sequential(nn.Linear(D,r,bias=False),nn.Linear(r,D,bias=False))
            self.decay=nn.Sequential(nn.Linear(D,r,bias=False),nn.Linear(r,H*M,bias=True))

    def operands(self,x):
        if not self.low_rank_gates:
            return super().operands(x)
        b,t,_=x.shape
        q,k,v=[conv(proj(x))[0].reshape(b,t,H,K) for conv,proj in zip(self.convs,(self.q,self.k,self.v))]
        with torch.autocast('cuda',enabled=False):
            g=F.logsigmoid(self.decay(x.float())).reshape(b,t,H,M)/8
            beta=F.linear(x.float(),self.beta.weight,self.beta.bias).sigmoid()
        keys=self.keys[None].expand(b,H,M,K).contiguous()
        values=torch.zeros(b,H,M,K,device=x.device)
        return q,k,v,g,beta,keys,values

    def forward(self,x):
        if not self.low_rank_gates and not self.output_sigmoid:
            return super().forward(x)
        from legacy_models import exact_komp,block_komp_fast
        b,t,_=x.shape
        q,k,v,g,beta,keys,values=self.operands(x)
        if self.kind=='exact':
            o,_,_=exact_komp(q,k,v,g,beta,(keys,values),eviction='fifo',compute_chunk=16,projection_dtype='float32')
        else:
            o,_=block_komp_fast(q,k,v,g,beta,(keys,values),chunk_size=128)
        y=self.norm(o.float()).to(x.dtype).reshape(b,t,D)
        z=self.gate(x)
        gate=z.sigmoid() if self.output_sigmoid else F.silu(z)
        return self.out(y*gate)


def initialize_mixer_options(mixer,config,seed,layer_idx):
    """Keep shared legacy initialization; opt-in modules use independent named seeds."""
    if config.low_rank_gates:
        for name,module in [('gate',mixer.gate),('decay',mixer.decay)]:
            key=f'komp_options.layer{layer_idx}.{name}'
            torch.manual_seed(seed+int(hashlib.sha256(key.encode()).hexdigest()[:8],16))
            nn.init.normal_(module[0].weight,0,.02)
            if name=='gate':
                # Match full .02 weight's expected effective matrix-entry variance.
                nn.init.normal_(module[1].weight,0,1/math.sqrt(config.gate_rank))
            else:
                # Random down + zero up permits up to learn on the first update.
                nn.init.zeros_(module[1].weight)
                nn.init.constant_(module[1].bias,math.log(ALPHA**8/(1-ALPHA**8)))
    if config.multi_timescale_init:
        g=head_log_decays(config.timescale_spread)
        # Stable logit(exp(8*g)); average g is exactly log(.995) before FP32 cast.
        biases=(8*g-torch.log(-torch.expm1(8*g))).repeat_interleave(M)
        target=mixer.decay[1].bias if config.low_rank_gates else mixer.decay.bias
        with torch.no_grad():target.copy_(biases.to(target))
