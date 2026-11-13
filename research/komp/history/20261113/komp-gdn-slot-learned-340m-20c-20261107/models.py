"""340M Block KOMP: original GDN slot decay and token-score eviction."""
from pathlib import Path
import hashlib
import math
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'snapshot'))
from fla.layers.komp import KOMPMixerBase
from fla.models.komp.configuration_komp import KOMPConfig
from fla.models.komp.modeling_komp import KOMPBlock, KOMPModel, TrainingLM
from fla.models.komp import legacy
from fla.models.komp.conversion import map_weights
from fla.ops.komp.block_fast import block_komp_fast
D, H, K, M, L, FF = 1024, 16, 64, 32, 24, 1824
PARAMETERS = 340372736

class LearnedSlotMixer(KOMPMixerBase):
    def __init__(self, config, layer_idx):
        super().__init__('block')
        self.config, self.layer_idx = config, layer_idx
        self.checkpoint_mixer = config.checkpoint_mixer
        self.decay = nn.Linear(D, H*M, bias=False)
        nn.init.zeros_(self.decay.weight)
        self.A_log = nn.Parameter(torch.zeros(H, M))
        self.dt_bias = nn.Parameter(torch.full((H, M), math.log(math.expm1(-math.log(config.alpha0)))))
        self.A_log._no_weight_decay = self.dt_bias._no_weight_decay = True
        self.score = nn.Linear(D, H)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        self.register_buffer('initial_score', torch.zeros(H, M))

    def operands(self, x):
        b, t, _ = x.shape
        q, k, v = [conv(proj(x))[0].reshape(b,t,H,K)
                   for conv,proj in zip(self.convs,(self.q,self.k,self.v))]
        with torch.autocast(x.device.type, enabled=False):
            z = F.linear(x.float(),self.decay.weight.float()).reshape(b,t,H,M)
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
        return self.out(y*F.silu(self.gate(x)))

    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, output_attentions=False, **kwargs):
        if use_cache or past_key_values is not None:
            raise ValueError('Pretraining only; generation cache unsupported')
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError('Dense packed inputs required; mask tail labels')
        if kwargs:
            raise TypeError(f'Unsupported mixer arguments: {sorted(kwargs)}')
        y = checkpoint(self._mix,hidden_states,use_reentrant=False) if self.training and self.checkpoint_mixer else self._mix(hidden_states)
        return y,None,None

class LearnedBlock(KOMPBlock):
    mixer_class = LearnedSlotMixer
class LearnedModel(KOMPModel):
    block_class = LearnedBlock
class LearnedTrainingLM(TrainingLM):
    model_class = LearnedModel


def initialize_fresh(model, seed):
    """Preserve every original FF1824 backbone weight and initial key exactly."""
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
                if local.startswith('attn.decay.'):continue
                assert local in current and current[local].shape==value.shape,name
                current[local].copy_(value)
                assert torch.equal(current[local],value)
                shared.update(name.encode());shared.update(value.contiguous().numpy().tobytes());count+=1
    model.shared_backbone_initial_sha=shared.hexdigest()
    model.shared_backbone_tensor_count=count


def LM(kind,seed=42):
    if kind!='block':raise ValueError('Only Block authorized')
    config=KOMPConfig(kind='block',checkpoint_mixer=True)
    config.decay_formula='-exp(A_log)*softplus(Wx+dt_bias)'
    config.eviction='learned';config.score_write_gate=True
    config.initial_score='fixed_zero';config.chunk_size=128
    config.initialization='legacy seed42 shared backbone; alpha=.995'
    torch.manual_seed(seed);model=LearnedTrainingLM(config)
    initialize_fresh(model,seed)
    assert sum(p.numel() for p in model.parameters())==PARAMETERS
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
        alpha0=.995,beta0=.2,decay_axis='physical_slot',decay_formula=model.config.decay_formula,
        eviction='learned top-M score; ties prefer newer keys',score_write_gate='beta*2*sigmoid(score)',
        score_projection='full Linear(D,H)',initial_score='fixed zero buffer, not trainable',
        selection_gradient=False,score_gradient='through write strength',
        initialization=model.config.initialization,
        shared_backbone_initial_sha=model.shared_backbone_initial_sha,
        shared_backbone_tensor_count=model.shared_backbone_tensor_count,
        checkpoint_policy='mixer_only',head_checkpoint=False,
        projection='float32',state_compute='float32',chunk_tape='bfloat16',
        output_gate='full Linear(D,D), SiLU; no KDA three-option changes',
        no_weight_decay=['vectors','keys','A_log','dt_bias'])
