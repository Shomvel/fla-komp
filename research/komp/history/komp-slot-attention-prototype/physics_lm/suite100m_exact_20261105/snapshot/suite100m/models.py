"""100M-class common backbone; native FLA baselines and conv-equipped KOMP/SWA."""
import hashlib
import importlib.machinery
import importlib.util
import math
import os
from pathlib import Path
import sys
import types
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ROOT=Path(os.environ.get('KOMP_SOURCE',Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(ROOT))
FLA=Path(os.environ['FLA_SOURCE'])
# Avoid unrelated transformer model registrations; execute native module exports.
for name,rel in [('fla','fla'),('fla.ops','fla/ops'),('fla.layers','fla/layers')]:
    package=types.ModuleType(name);package.__path__=[str(FLA/rel)]
    package.__file__=str(FLA/rel/'__init__.py');package.__version__='0.6.0'
    package.__spec__=importlib.machinery.ModuleSpec(name,loader=None,is_package=True)
    sys.modules[name]=package
spec=importlib.util.spec_from_file_location('fla.modules',FLA/'fla/modules/__init__.py',submodule_search_locations=[str(FLA/'fla/modules')])
module=importlib.util.module_from_spec(spec);sys.modules['fla.modules']=module;spec.loader.exec_module(module)
from fla.modules import ShortConvolution
from fla.layers.gated_deltanet import GatedDeltaNet
from fla.layers.gsa import GatedSlotAttention
from fla.layers.gsa2 import GatedSlotAttention2
from komp_slot.block_fast import block_komp_fast
from komp_slot.exact_komp import exact_komp

D,H,K,M,L,FF,VOCAB=512,8,64,32,15,2048,32000
KINDS=('gdn','full','swa','gsa','gsa2','komp','exact')

class Native(nn.Module):
    def __init__(self,kind,index):
        super().__init__()
        if kind=='gdn':self.layer=GatedDeltaNet(hidden_size=D,head_dim=K,num_heads=H,expand_v=1.,use_short_conv=True,conv_size=4,layer_idx=index)
        elif kind=='gsa':self.layer=GatedSlotAttention(hidden_size=D,num_heads=H,expand_k=1.,expand_v=1.,num_slots=M,use_short_conv=True,conv_size=4,layer_idx=index)
        else:self.layer=GatedSlotAttention2(hidden_size=D,head_dim=K,num_heads=H,num_slots=64,expand_v=1.,mode='chunk',layer_idx=index)
    def forward(self,x):return self.layer(x,use_cache=False)[0]

class Mixer(nn.Module):
    def __init__(self,kind,window=96):
        super().__init__();self.kind=kind;self.window=window
        for name in ('q','k','v','out','gate'):setattr(self,name,nn.Linear(D,D,bias=False))
        self.convs=nn.ModuleList([ShortConvolution(D,4,bias=False,activation='silu',backend='triton') for _ in range(3)])
        self.norm=nn.RMSNorm(K)
        if kind in ('komp','exact'):
            self.decay=nn.Linear(D,H*M);self.beta=nn.Linear(D,H)
            self.keys=nn.Parameter(torch.stack([torch.linalg.qr(torch.randn(K,M)).Q.T for _ in range(H)]))
            nn.init.zeros_(self.decay.weight);nn.init.constant_(self.decay.bias,math.log(.98**8/(1-.98**8)))
            nn.init.zeros_(self.beta.weight);nn.init.constant_(self.beta.bias,math.log(.2/.8))
        self.register_buffer('rope_freq',10000.**(-torch.arange(0,K,2).float()/K),persistent=False)
    def rope(self,x):
        angle=torch.arange(x.shape[1],device=x.device).float()[:,None]*self.rope_freq[None]
        cs,sn=angle.cos()[None,:,None],angle.sin()[None,:,None]
        a,b=x.float()[...,::2],x.float()[...,1::2]
        return torch.stack((a*cs-b*sn,a*sn+b*cs),-1).flatten(-2).to(x.dtype)
    def forward(self,x):
        b,t,_=x.shape
        q,k,v=[conv(proj(x))[0].reshape(b,t,H,K) for conv,proj in zip(self.convs,(self.q,self.k,self.v))]
        if self.kind in ('komp','exact'):
            with torch.autocast('cuda',enabled=False):
                g=F.logsigmoid(F.linear(x.float(),self.decay.weight,self.decay.bias)).reshape(b,t,H,M)/8
                beta=F.linear(x.float(),self.beta.weight,self.beta.bias).sigmoid()
            keys=self.keys[None].expand(b,H,M,K).contiguous();values=torch.zeros(b,H,M,K,device=x.device)
            if self.kind=='exact':
                o,_,_=exact_komp(q,k,v,g,beta,(keys,values),eviction='fifo',compute_chunk=16,projection_dtype='float32')
            else:
                o,_=block_komp_fast(q,k,v,g,beta,(keys,values),chunk_size=128)
        else:
            q,k=self.rope(q).transpose(1,2),self.rope(k).transpose(1,2);v=v.transpose(1,2)
            if self.kind=='full':o=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            else:
                pos=torch.arange(t,device=x.device);distance=pos[:,None]-pos[None,:]
                o=F.scaled_dot_product_attention(q,k,v,attn_mask=(distance>=0)&(distance<self.window))
            o=o.transpose(1,2)
        y=self.norm(o.float()).to(x.dtype).reshape(b,t,D)
        return self.out(y*F.silu(self.gate(x)))

class Block(nn.Module):
    def __init__(self,kind,index,window,ff=FF):
        super().__init__();self.n1=nn.RMSNorm(D);self.n2=nn.RMSNorm(D)
        self.mix=Native(kind,index) if kind in ('gdn','gsa','gsa2') else Mixer(kind,window)
        self.up=nn.Linear(D,2*ff,bias=False);self.down=nn.Linear(ff,D,bias=False)
    def forward(self,x):
        x=x+self.mix(self.n1(x));a,b=self.up(self.n2(x)).chunk(2,-1)
        return x+self.down(F.silu(a)*b)

class LM(nn.Module):
    def __init__(self,kind,seed,window=96,checkpoint_layers=True):
        super().__init__();assert kind in KINDS
        self.kind=kind;self.checkpoint_layers=checkpoint_layers
        torch.manual_seed(seed);self.embed=nn.Embedding(VOCAB,D)
        self.layers=nn.ModuleList()
        probe=Block(kind,0,window)
        mixer_params=sum(p.numel() for p in probe.mix.parameters())
        self.ff=max(64,round((100_000_000-2*VOCAB*D-L*(mixer_params+2*D)-D)/(L*3*D*64))*64)
        del probe
        for i in range(L):
            torch.manual_seed(seed+100+i);self.layers.append(Block(kind,i,window,self.ff))
        self.norm=nn.RMSNorm(D);self.head=nn.Linear(D,VOCAB,bias=False)
        # Preserve native FLA's gate/decay initialization. Named seeds fix shared backbone parameters.
        for name,module in self.named_modules():
            shared='.mix.' not in name
            custom=kind in ('komp','exact','full','swa') and name.endswith(('.q','.k','.v','.out','.gate'))
            if isinstance(module,(nn.Linear,nn.Embedding)) and (shared or custom):
                torch.manual_seed(seed+int(hashlib.sha256(name.encode()).hexdigest()[:8],16))
                nn.init.normal_(module.weight,0,.02)
                if getattr(module,'bias',None) is not None:nn.init.zeros_(module.bias)
        # Identical causal conv filters for the custom KOMP/full/SWA paths.
        if kind in ('komp','exact','full','swa'):
            for i,layer in enumerate(self.layers):
                for j,conv in enumerate(layer.mix.convs):
                    torch.manual_seed(seed+6000+i*3+j);nn.init.normal_(conv.weight,0,.02)
    def forward(self,ids):
        x=self.embed(ids)
        for layer in self.layers:
            x=checkpoint(layer,x,use_reentrant=False) if self.training and self.checkpoint_layers else layer(x)
        return self.norm(x)

def optimizer(model,lr,wd):
    groups={}
    for name,p in model.named_parameters():
        no_decay=bool(getattr(p,'_no_weight_decay',False))
        residual=name.endswith(('.mix.out.weight','.mix.layer.o_proj.weight','.down.weight'))
        groups.setdefault((no_decay,residual),[]).append(p)
    return torch.optim.AdamW([dict(params=ps,weight_decay=0. if flag else wd,residual_output=residual) for (flag,residual),ps in groups.items()],lr=lr,betas=(.9,.95),fused=True)

def metadata(model):
    return dict(kind=model.kind,parameters=sum(p.numel() for p in model.parameters()),width=D,layers=L,ff=model.ff,heads=H,head_dim=K,short_conv=True,conv_size=4,block_chunk=128 if model.kind=='komp' else None,block_slots=M if model.kind in ('komp','exact') else None,exact_compute_chunk=16 if model.kind=='exact' else None,compression_tokens=1 if model.kind=='exact' else 128 if model.kind=='komp' else None,eviction='fifo' if model.kind in ('komp','exact') else None,projection_dtype='float32' if model.kind=='exact' else None,gsa_slots=M if model.kind=='gsa' else 64 if model.kind=='gsa2' else None,rope_base=10000 if model.kind in ('full','swa') else None,swa_window=model.layers[0].mix.window if model.kind=='swa' else None)
