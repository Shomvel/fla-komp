"""Paired ~340M backbones, changing only dictionary compression frequency."""
import hashlib
import importlib.machinery
import importlib.util
import math
from pathlib import Path
import sys
import types
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'snapshot'))
FLA=ROOT/'snapshot/fla-source'
for name,rel in [('fla','fla'),('fla.ops','fla/ops')]:
    package=types.ModuleType(name);package.__path__=[str(FLA/rel)]
    package.__file__=str(FLA/rel/'__init__.py');package.__version__='0.6.0'
    package.__spec__=importlib.machinery.ModuleSpec(name,loader=None,is_package=True)
    sys.modules[name]=package
spec=importlib.util.spec_from_file_location('fla.modules',FLA/'fla/modules/__init__.py',submodule_search_locations=[str(FLA/'fla/modules')])
module=importlib.util.module_from_spec(spec);sys.modules['fla.modules']=module;spec.loader.exec_module(module)
from fla.modules import ShortConvolution
from komp_slot.exact_komp import exact_komp
from komp_slot.block_fast import block_komp_fast
D,H,K,M,L,FF,VOCAB=1024,16,64,32,24,2272,32000
ALPHA,BETA=.995,.2

class Mixer(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        for name in ('q','k','v','out','gate'):setattr(self,name,nn.Linear(D,D,bias=False))
        self.convs=nn.ModuleList([ShortConvolution(D,4,bias=False,activation='silu',backend='triton') for _ in range(3)])
        self.norm=nn.RMSNorm(K,eps=1e-6)
        self.decay=nn.Linear(D,H*M);self.beta=nn.Linear(D,H)
        self.keys=nn.Parameter(torch.stack([torch.linalg.qr(torch.randn(K,M)).Q.T for _ in range(H)]))
        nn.init.zeros_(self.decay.weight);nn.init.constant_(self.decay.bias,math.log(ALPHA**8/(1-ALPHA**8)))
        nn.init.zeros_(self.beta.weight);nn.init.constant_(self.beta.bias,math.log(BETA/(1-BETA)))
    def operands(self,x):
        b,t,_=x.shape
        q,k,v=[conv(proj(x))[0].reshape(b,t,H,K) for conv,proj in zip(self.convs,(self.q,self.k,self.v))]
        with torch.autocast('cuda',enabled=False):
            g=F.logsigmoid(F.linear(x.float(),self.decay.weight,self.decay.bias)).reshape(b,t,H,M)/8
            beta=F.linear(x.float(),self.beta.weight,self.beta.bias).sigmoid()
        keys=self.keys[None].expand(b,H,M,K).contiguous()
        values=torch.zeros(b,H,M,K,device=x.device)
        return q,k,v,g,beta,keys,values
    def forward(self,x):
        b,t,_=x.shape
        q,k,v,g,beta,keys,values=self.operands(x)
        if self.kind=='exact':
            o,_,_=exact_komp(q,k,v,g,beta,(keys,values),eviction='fifo',compute_chunk=16,projection_dtype='float32')
        else:o,_=block_komp_fast(q,k,v,g,beta,(keys,values),chunk_size=128)
        y=self.norm(o.float()).to(x.dtype).reshape(b,t,D)
        return self.out(y*F.silu(self.gate(x)))

class Block(nn.Module):
    def __init__(self,kind):
        super().__init__();self.n1=nn.RMSNorm(D,eps=1e-6);self.n2=nn.RMSNorm(D,eps=1e-6)
        self.mix=Mixer(kind);self.up=nn.Linear(D,2*FF,bias=False);self.down=nn.Linear(FF,D,bias=False)
    def forward(self,x):
        x=x+self.mix(self.n1(x));a,b=self.up(self.n2(x)).chunk(2,-1)
        return x+self.down(F.silu(a)*b)

class LM(nn.Module):
    def __init__(self,kind,seed=42,checkpoint_layers=True):
        super().__init__();assert kind in ('exact','block');self.kind=kind;self.checkpoint_layers=checkpoint_layers
        torch.manual_seed(seed);self.embed=nn.Embedding(VOCAB,D)
        self.layers=nn.ModuleList()
        for i in range(L):
            torch.manual_seed(seed+100+i);self.layers.append(Block(kind))
        self.norm=nn.RMSNorm(D,eps=1e-6);self.head=nn.Linear(D,VOCAB,bias=False)
        for name,module in self.named_modules():
            if isinstance(module,(nn.Linear,nn.Embedding)) and not name.endswith(('.decay','.beta')):
                torch.manual_seed(seed+int(hashlib.sha256(name.encode()).hexdigest()[:8],16))
                # FLA-style depth scaling for residual output projections.
                std=.02/math.sqrt(2*L) if name.endswith(('.mix.out','.down')) else .02
                nn.init.normal_(module.weight,0,std)
                if getattr(module,'bias',None) is not None:nn.init.zeros_(module.bias)
        for i,layer in enumerate(self.layers):
            for j,conv in enumerate(layer.mix.convs):
                torch.manual_seed(seed+6000+i*3+j);nn.init.normal_(conv.weight,0,.02)
    def forward(self,ids,labels=None,valid_tokens=None):
        x=self.embed(ids)
        for layer in self.layers:
            x=checkpoint(layer,x,use_reentrant=False) if self.training and self.checkpoint_layers else layer(x)
        x=self.norm(x)
        if labels is None:return x
        # Bound vocabulary-logit memory, keeping the head in the DDP graph.
        loss=x.new_zeros((),dtype=torch.float32)
        flat=x.flatten(0,1);target=labels.flatten()
        if valid_tokens is not None:
            target=target.clone();target[valid_tokens:]=-100
        for start in range(0,len(flat),256):
            loss=loss+checkpoint(lambda h,y:F.cross_entropy(self.head(h).float(),y,reduction='sum',ignore_index=-100),flat[start:start+256],target[start:start+256],use_reentrant=False)
        return loss

def optimizer(model):
    groups={True:[],False:[]}
    for name,p in model.named_parameters():groups[p.ndim<2 or name.endswith('.keys')].append(p)
    return torch.optim.AdamW([dict(params=ps,weight_decay=0. if flag else .1) for flag,ps in groups.items()],lr=3e-4,betas=(.8,.95),eps=1e-15,fused=True)

def metadata(model):
    return dict(kind=model.kind,parameters=sum(p.numel() for p in model.parameters()),width=D,layers=L,ff=FF,heads=H,head_dim=K,slots=M,alpha0=ALPHA,beta0=BETA,seed=42,vocab=VOCAB,short_conv=4,compression_tokens=1 if model.kind=='exact' else 128,exact_compute_chunk=16 if model.kind=='exact' else None,eviction='fifo',temperature=.5,ridge=1e-4,projection='float32',state_compute='float32',chunk_tape='float32' if model.kind=='exact' else 'bfloat16',residual_init_std=.02/math.sqrt(2*L))
