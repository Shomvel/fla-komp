"""Native FLA Block KOMP: GDN slot decay, learned eviction, and options 123."""
from pathlib import Path
import hashlib
import math
import sys
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get('KOMP_SNAPSHOT', str(ROOT / 'snapshot')))
from fla.layers.komp import KOMPMixerBase
from fla.models.komp.configuration_komp import KOMPConfig
from fla.models.komp.modeling_komp import KOMPBlock, KOMPModel, TrainingLM
from fla.models.komp import legacy
from fla.models.komp.conversion import map_weights
from fla.ops.komp.block_fast import block_komp_fast
from fla.ops.komp.geometry_cache import geometry_contexts
D, H, K, M, L, FF = 2048, 32, 64, 32, 24, 3072
PARAMETERS = 1003181056
OPTION_DEFAULTS = dict(output_sigmoid=True, low_rank_gates=True,
                       multi_timescale_init=True, gate_rank=64, timescale_spread=4.)


def head_log_decays(spread=4., alpha=.995):
    if not math.isfinite(spread) or spread < 1:
        raise ValueError('timescale_spread must be finite and >= 1')
    factors = torch.logspace(-math.log10(spread), math.log10(spread), H, dtype=torch.float64)
    return factors / factors.mean() * math.log(alpha)


def initialize_mixer_options(mixer, config, seed, layer_idx):
    """Named gate seeds match existing FF3072 combo; GDN rates match its initial g."""
    if config.low_rank_gates:
        for name, module in [('gate', mixer.gate), ('decay', mixer.decay)]:
            key = f'komp_options.layer{layer_idx}.{name}'
            torch.manual_seed(seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
            nn.init.normal_(module[0].weight, 0, .02)
            if name == 'gate':
                nn.init.normal_(module[1].weight, 0, 1 / math.sqrt(config.gate_rank))
            else:
                nn.init.zeros_(module[1].weight)
    else:
        nn.init.zeros_(mixer.decay.weight)
    g = head_log_decays(config.timescale_spread, config.alpha0) if config.multi_timescale_init else torch.full((H,), math.log(config.alpha0), dtype=torch.float64)
    rates = -g[:, None].expand(H, M)
    # Inverse softplus; A_log remains zero, so all head diversity lives in dt_bias.
    with torch.no_grad():
        mixer.A_log.zero_()
        mixer.dt_bias.copy_(torch.log(torch.expm1(rates)).to(mixer.dt_bias))
        mixer.score.weight.zero_()
        mixer.score.bias.zero_()
        mixer.initial_score.zero_()


class LearnedSlotMixer(KOMPMixerBase):
    def __init__(self, config, layer_idx):
        super().__init__('block')
        self.config, self.layer_idx = config, layer_idx
        self.checkpoint_mixer = config.checkpoint_mixer
        self.cache_ut_layers = int(os.environ.get('KOMP_UT_CACHE_LAYERS','0'))
        if not 0 <= self.cache_ut_layers <= L:
            raise ValueError('KOMP_UT_CACHE_LAYERS must be between 0 and 24')
        self.output_sigmoid = config.output_sigmoid
        self.low_rank_gates = config.low_rank_gates
        if config.low_rank_gates:
            r = config.gate_rank
            if isinstance(r, bool) or not isinstance(r, int) or not 1 <= r <= H*M:
                raise ValueError(f'gate_rank must be an integer in [1,{H*M}]')
            self.gate = nn.Sequential(nn.Linear(D, r, bias=False), nn.Linear(r, D, bias=False))
            # dt_bias supplies the sole additive offset, outside the low-rank product.
            self.decay = nn.Sequential(nn.Linear(D, r, bias=False), nn.Linear(r, H*M, bias=False))
        else:
            self.decay = nn.Linear(D, H*M, bias=False)
        self.A_log = nn.Parameter(torch.zeros(H, M))
        self.dt_bias = nn.Parameter(torch.empty(H, M))
        self.A_log._no_weight_decay = self.dt_bias._no_weight_decay = True
        self.score = nn.Linear(D, H)
        self.register_buffer('initial_score', torch.zeros(H, M))
        initialize_mixer_options(self, config, 42, layer_idx)

    def operands(self, x):
        b, t, _ = x.shape
        q, k, v = [conv(proj(x))[0].reshape(b,t,H,K)
                   for conv,proj in zip(self.convs,(self.q,self.k,self.v))]
        with torch.autocast(x.device.type, enabled=False):
            z = self.decay(x.float()).reshape(b,t,H,M)
            g = -self.A_log.float().exp()*F.softplus(z+self.dt_bias.float())
            scores = F.linear(x.float(),self.score.weight.float(),self.score.bias.float())
            beta = F.linear(x.float(),self.beta.weight.float(),self.beta.bias.float()).sigmoid()
            beta = beta*(2*scores.sigmoid())
        keys = self.keys[None].expand(b,H,M,K).contiguous()
        values = torch.zeros(b,H,M,K,device=x.device,dtype=torch.float32)
        initial_scores = self.initial_score[None].expand(b,H,M)
        return q,k,v,g,beta,keys,values,scores,initial_scores

    def _mix(self,x):
        q,k,v,g,beta,keys,values,scores,initial = self.operands(x)
        o,_ = block_komp_fast(q,k,v,g,beta,(keys,values),chunk_size=128,
                             evict='learned',scores=scores,initial_scores=initial)
        y = self.norm(o.float()).to(x.dtype).flatten(-2)
        z = self.gate(x)
        gate = z.sigmoid() if self.output_sigmoid else F.silu(z)
        return self.out(y*gate)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, output_attentions=False, **kwargs):
        if use_cache or past_key_values is not None:
            raise ValueError('Pretraining only; generation cache unsupported')
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError('Dense packed inputs required; mask tail labels')
        if kwargs:
            raise TypeError(f'Unsupported mixer arguments: {sorted(kwargs)}')
        options = {'context_fn':geometry_contexts} if self.layer_idx < self.cache_ut_layers else {}
        y = checkpoint(self._mix,hidden_states,use_reentrant=False,**options) if self.training and self.checkpoint_mixer else self._mix(hidden_states)
        return y,None,None


class LearnedBlock(KOMPBlock):
    mixer_class = LearnedSlotMixer
class LearnedModel(KOMPModel):
    block_class = LearnedBlock
class LearnedTrainingLM(TrainingLM):
    model_class = LearnedModel


def initialize_fresh(model, seed):
    """Preserve same-FF3072 legacy shared backbone; initialize changed gates separately."""
    def named(name,module):
        torch.manual_seed(seed+int(hashlib.sha256(name.encode()).hexdigest()[:8],16))
        std=.02/math.sqrt(2*L) if name.endswith(('.mix.out','.down')) else .02
        nn.init.normal_(module.weight,0,std)
        if getattr(module,'bias',None) is not None: nn.init.zeros_(module.bias)
    named('embed',model.model.embeddings);named('head',model.lm_head)
    nn.init.ones_(model.model.norm.weight)
    shared=hashlib.sha256();count=0
    for i,layer in enumerate(model.model.layers):
        torch.manual_seed(seed+100+i);old=legacy.Block('block')
        for name,module in old.named_modules():
            if isinstance(module,(nn.Linear,nn.Embedding)) and not name.endswith(('.decay','.beta')):
                named(f'layers.{i}.{name}',module)
        for j,conv in enumerate(old.mix.convs):
            torch.manual_seed(seed+6000+i*3+j);nn.init.normal_(conv.weight,0,.02)
        mapped=map_weights({f'layers.{i}.'+n:v for n,v in old.state_dict().items()})
        current=layer.state_dict();prefix=f'model.layers.{i}.'
        with torch.no_grad():
            for name,value in mapped.items():
                local=name[len(prefix):]
                if local.startswith('attn.decay.') or (model.config.low_rank_gates and local.startswith('attn.gate.')):
                    continue
                assert local in current and current[local].shape==value.shape,name
                current[local].copy_(value)
                assert torch.equal(current[local],value)
                if not local.startswith('attn.gate.'):
                    shared.update(name.encode());shared.update(value.contiguous().numpy().tobytes());count+=1
        initialize_mixer_options(layer.attn,model.config,seed,i)
    model.shared_backbone_initial_sha=shared.hexdigest()
    model.shared_backbone_tensor_count=count


def LM(kind,seed=42,options=None):
    if kind!='block':raise ValueError('Only Block authorized')
    opts=dict(OPTION_DEFAULTS)
    if options is not None:
        if set(options)-set(opts):raise ValueError('Unknown options')
        opts.update(options)
    config=KOMPConfig(kind='block',checkpoint_mixer=True,**opts)
    config.decay_formula='-exp(A_log)*softplus(W_up W_down x+dt_bias)'
    config.eviction='learned';config.score_write_gate=True
    config.initial_score='fixed_zero';config.chunk_size=128
    config.initialization='legacy seed42 shared FF3072 backbone; named combo gate seeds; mean logalpha=log(.995)'
    torch.manual_seed(seed);model=LearnedTrainingLM(config)
    initialize_fresh(model,seed)
    expected=2*32000*D+D+L*(5*D*D+D*H*M+2*D*H+2*H+H*M*K+2*H*M+12*D+K+2*D+3*D*FF)
    if opts['low_rank_gates']:expected-=L*(D*D+D*H*M-opts['gate_rank']*(D+D+D+H*M))
    assert sum(p.numel() for p in model.parameters())==expected
    return model


def optimizer(model):
    groups={True:[],False:[]}
    for n,p in model.named_parameters():
        nd=p.ndim<2 or n.endswith(('.keys','.A_log','.dt_bias'))
        groups[nd].append(p)
    return torch.optim.AdamW([dict(params=ps,weight_decay=0. if nd else .1) for nd,ps in groups.items()],
                            lr=3e-4,betas=(.8,.95),eps=1e-15,fused=True)


def metadata(model):
    return dict(kind='block',parameters=sum(p.numel() for p in model.parameters()),
        config=model.config.to_dict(),width=D,layers=L,ff=FF,heads=H,head_dim=K,slots=M,
        options={k:getattr(model.config,k) for k in OPTION_DEFAULTS},
        alpha0=.995,beta0=.2,decay_axis='physical_slot',decay_formula=model.config.decay_formula,
        eviction='learned top-M score; ties prefer newer keys',score_write_gate='beta*2*sigmoid(score)',
        score_projection='full Linear(D,H)',initial_score='fixed zero buffer, not trainable',
        selection_gradient=False,score_gradient='through write strength',
        initialization=model.config.initialization,
        shared_backbone_initial_sha=model.shared_backbone_initial_sha,
        shared_backbone_tensor_count=model.shared_backbone_tensor_count,
        initial_head_log_decays=head_log_decays(model.config.timescale_spread).tolist(),
        checkpoint_policy='mixer_only',head_checkpoint=False,
        projection='float32',state_compute='float32',chunk_tape='bfloat16',
        output_gate='rank64 Linear x Linear, no bias/nonlinearity; sigmoid',
        decay_projection='rank64, no bias/nonlinearity, random down and zero up',
        no_weight_decay=['vectors','keys','A_log','dt_bias'],
        comparison_limit='GDN slot+learned+write gate+three options; not a single-switch ablation; FF differs from original')
