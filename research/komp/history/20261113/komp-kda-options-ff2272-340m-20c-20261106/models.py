"""Registered FLA KOMP model; native FLA blocks/loop/loss, compatible checkpoint mapping."""
import copy,importlib.machinery,json,os,sys,types,hashlib,math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import legacy_models as legacy
from legacy_models import ROOT,FLA,D,H,K,M,L,FF,VOCAB,ALPHA,BETA,exact_komp,block_komp_fast
for name,rel in [('fla.layers','fla/layers'),('fla.models','fla/models'),('fla.models.gated_deltanet','fla/models/gated_deltanet')]:
 package=types.ModuleType(name);package.__path__=[str(FLA/rel)];package.__file__=str(FLA/rel/'__init__.py');package.__spec__=importlib.machinery.ModuleSpec(name,loader=None,is_package=True);sys.modules[name]=package
from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetBlock,GatedDeltaNetPreTrainedModel
from fla.models.modeling_utils import FLAModel,FLAForCausalLM
from fla.modules import GatedMLP
from transformers import AutoConfig,AutoModel,AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

class KOMPConfig(GatedDeltaNetConfig):
 model_type='komp'
 def __init__(self,kind='exact',checkpoint_mixer=True,output_sigmoid=False,low_rank_gates=False,multi_timescale_init=False,gate_rank=64,timescale_spread=4.,**kwargs):
  defaults=dict(hidden_size=D,num_hidden_layers=L,num_heads=H,head_dim=K,expand_v=1.,intermediate_size=FF,vocab_size=VOCAB,norm_eps=1e-6,fuse_norm=False,fuse_swiglu=False,fuse_cross_entropy=False,fuse_linear_cross_entropy=True,use_cache=False,tie_word_embeddings=False)
  defaults.update(kwargs);super().__init__(**defaults);self.kind=kind;self.checkpoint_mixer=checkpoint_mixer
  for name,want in [('hidden_size',D),('num_hidden_layers',L),('num_heads',H),('head_dim',K),('intermediate_size',FF),('vocab_size',VOCAB),('fuse_norm',False)]:
   if getattr(self,name)!=want:raise ValueError(f'Frozen KOMP340M config requires {name}={want}')
  if kind not in ('exact','block'):raise ValueError('kind must be exact or block')
  self.num_slots=M;self.alpha0=ALPHA;self.beta0=BETA;self.temperature=.5;self.ridge=1e-4
  self.output_sigmoid=output_sigmoid;self.low_rank_gates=low_rank_gates;self.multi_timescale_init=multi_timescale_init;self.gate_rank=gate_rank;self.timescale_spread=timescale_spread

from mixer_options import OptionalMixer,initialize_mixer_options

class KOMPMixer(OptionalMixer):
 def __init__(self,config,layer_idx):
  super().__init__(config);self.layer_idx=layer_idx;self.checkpoint_mixer=config.checkpoint_mixer
 def forward(self,hidden_states,attention_mask=None,past_key_values=None,use_cache=False,output_attentions=False,**kwargs):
  if use_cache or past_key_values is not None:raise ValueError('KOMP pretraining adapter does not implement generation cache')
  if attention_mask is not None and not bool(attention_mask.all()):raise ValueError('Packed pretraining requires fully valid inputs; mask labels for tail loss')
  y=checkpoint(super().forward,hidden_states,use_reentrant=False) if self.training and self.checkpoint_mixer else super().forward(hidden_states)
  return y,None,None

class RoundingCompatibleMLP(GatedMLP):
 """Native FLA projections, preserving the previous BF16 SiLU-before-multiply rounding."""
 def forward(self,x,**kwargs):return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))

class KOMPBlock(GatedDeltaNetBlock):
 def __init__(self,config,layer_idx):
  # Native block creates the native FLA MLP and norm/residual contract.
  super().__init__(config,layer_idx);self.attn=KOMPMixer(config,layer_idx)
  self.mlp=RoundingCompatibleMLP(config.hidden_size,intermediate_size=config.intermediate_size,fuse_swiglu=False)

class KOMPModel(GatedDeltaNetPreTrainedModel,FLAModel):
 config_class=KOMPConfig
 def __init__(self,config):
  super().__init__(config);self.padding_idx=config.pad_token_id;self.vocab_size=config.vocab_size
  self.embeddings=nn.Embedding(config.vocab_size,config.hidden_size,padding_idx=config.pad_token_id)
  self.layers=nn.ModuleList(KOMPBlock(config,i) for i in range(config.num_hidden_layers));self.norm=nn.RMSNorm(config.hidden_size,eps=config.norm_eps)
  self.use_attnres=False;self.gradient_checkpointing=False
  # All weights come from a validated legacy checkpoint; no post_init reinitialization.

class KOMPForCausalLM(GatedDeltaNetPreTrainedModel,FLAForCausalLM):
 config_class=KOMPConfig
 _tied_weights_keys={}
 def __init__(self,config):
  super().__init__(config);self.model=KOMPModel(config);self.lm_head=nn.Linear(config.hidden_size,config.vocab_size,bias=False);self.vocab_size=config.vocab_size;self.criterion=None;self.kind=config.kind
 def forward(self,input_ids=None,attention_mask=None,inputs_embeds=None,labels=None,shift_labels=None,num_items_in_batch=None,use_cache=False,past_key_values=None,return_dict=True,logits_to_keep=0,**kwargs):
  if kwargs:raise TypeError(f'Unsupported KOMP model arguments: {sorted(kwargs)}')
  if use_cache or past_key_values is not None:raise ValueError('Generation cache is not implemented')
  output=self.model(input_ids=input_ids,attention_mask=attention_mask,inputs_embeds=inputs_embeds,use_cache=False,return_dict=True)
  loss,logits=self._compute_logits_and_loss(output.last_hidden_state,labels=labels,shift_labels=shift_labels,num_items_in_batch=num_items_in_batch,logits_to_keep=logits_to_keep)
  if not return_dict:return (loss,logits) if loss is not None else (logits,)
  return CausalLMOutputWithPast(loss=loss,logits=logits)
 def forward_train(self,input_ids,aligned_labels=None,valid_tokens=None):
  hidden=self.model(input_ids=input_ids,use_cache=False,return_dict=True).last_hidden_state
  if aligned_labels is None:return hidden
  aligned=aligned_labels
  if valid_tokens is not None:
   aligned=aligned_labels.reshape(-1).clone();aligned[valid_tokens:]=-100;aligned=aligned.reshape(aligned_labels.shape)
  if torch.is_autocast_enabled('cuda'):hidden=hidden.to(torch.get_autocast_dtype('cuda'))
  loss,_=self._compute_logits_and_loss(hidden,shift_labels=aligned,num_items_in_batch=None)
  return loss*aligned.ne(-100).sum()

class TrainingLM(KOMPForCausalLM):
 """Thin adapter for the existing trainer's aligned-label sum-loss contract."""
 def forward(self,input_ids,labels=None,valid_tokens=None):return self.forward_train(input_ids,labels,valid_tokens)

AutoConfig.register('komp',KOMPConfig,exist_ok=True)
AutoModel.register(KOMPConfig,KOMPModel,exist_ok=True)
AutoModelForCausalLM.register(KOMPConfig,KOMPForCausalLM,exist_ok=True)

def mappings(old_name):
 if old_name=='embed.weight':return [('model.embeddings.weight',None)]
 if old_name=='head.weight':return [('lm_head.weight',None)]
 if old_name=='norm.weight':return [('model.norm.weight',None)]
 fields=old_name.split('.');assert fields[0]=='layers',old_name
 stem='model.layers.'+fields[1]+'.';tail='.'.join(fields[2:])
 if tail=='up.weight':return [(stem+'mlp.gate_proj.weight',slice(0,FF)),(stem+'mlp.up_proj.weight',slice(FF,None))]
 for old,new in [('n1.','attn_norm.'),('n2.','mlp_norm.'),('mix.','attn.'),('down.','mlp.down_proj.')]:
  if tail.startswith(old):return [(stem+new+tail[len(old):],None)]
 raise KeyError(old_name)

def map_weights(state):
 mapped={}
 for name,value in state.items():
  for target,part in mappings(name):mapped[target]=value if part is None else value[part].clone()
 return mapped

def initialize_fresh(model,seed=42):
 """Reproduce legacy seed42 shared weights layer by layer, then initialize opt-in gates."""
 def initialize_named(name,module):
  torch.manual_seed(seed+int(hashlib.sha256(name.encode()).hexdigest()[:8],16))
  std=.02/math.sqrt(2*L) if name.endswith(('.mix.out','.down')) else .02
  nn.init.normal_(module.weight,0,std)
  if getattr(module,'bias',None) is not None:nn.init.zeros_(module.bias)
 initialize_named('embed',model.model.embeddings);initialize_named('head',model.lm_head)
 nn.init.ones_(model.model.norm.weight)
 shared_hash=hashlib.sha256()
 for i,layer in enumerate(model.model.layers):
  torch.manual_seed(seed+100+i);old=legacy.Block(model.kind)
  for name,module in old.named_modules():
   if isinstance(module,(nn.Linear,nn.Embedding)) and not name.endswith(('.decay','.beta')):
    initialize_named(f'layers.{i}.{name}',module)
  for j,conv in enumerate(old.mix.convs):
   torch.manual_seed(seed+6000+i*3+j);nn.init.normal_(conv.weight,0,.02)
  mapped=map_weights({f'layers.{i}.'+k:v for k,v in old.state_dict().items()})
  current=layer.state_dict();prefix=f'model.layers.{i}.'
  with torch.no_grad():
   for key,value in mapped.items():
    local=key[len(prefix):]
    if local in current:
     assert current[local].shape==value.shape
     current[local].copy_(value)
     if not local.startswith(('attn.gate.','attn.decay.')):
      shared_hash.update(key.encode());shared_hash.update(value.contiguous().numpy().tobytes())
  initialize_mixer_options(layer.attn,model.config,seed,i)
 model.shared_backbone_initial_sha=shared_hash.hexdigest()

def LM(kind,seed=42,options=None):
 if options is None:options=json.loads((ROOT/'configs/options.json').read_text())
 keys=('output_sigmoid','low_rank_gates','multi_timescale_init','gate_rank','timescale_spread')
 config=KOMPConfig(kind=kind,checkpoint_mixer=True,**{k:options[k] for k in keys if k in options})
 torch.manual_seed(seed);model=TrainingLM(config);initialize_fresh(model,seed)
 expected=339966848+24*3*1024*(FF-1824)-(24*(1024*1024+1024*512-config.gate_rank*(1024+1024+1024+512)) if config.low_rank_gates else 0)
 assert sum(p.numel() for p in model.parameters())==expected
 return model

def optimizer(model):
 groups={True:[],False:[]}
 for name,p in model.named_parameters():groups[p.ndim<2 or name.endswith('.keys')].append(p)
 return torch.optim.AdamW([dict(params=ps,weight_decay=0. if flag else .1) for flag,ps in groups.items()],lr=3e-4,betas=(.8,.95),eps=1e-15,fused=True)

def map_optimizer(old_model,old_state,new_model,new_opt):
 old_groups=legacy.optimizer(old_model).param_groups;old_names={id(p):n for n,p in old_model.named_parameters()};names={}
 for parameters,saved in zip(old_groups,old_state['param_groups']):
  for p,index in zip(parameters['params'],saved['params']):names[index]=old_names[id(p)]
 transformed={};source_group={}
 for group in old_state['param_groups']:
  for index in group['params']:
   for target,part in mappings(names[index]):
    source_group[target]=group
    transformed[target]={k:(v[part].clone() if torch.is_tensor(v) and v.ndim>0 and part is not None else v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k,v in old_state['state'].get(index,{}).items()}
 new_names={id(p):n for n,p in new_model.named_parameters()};out=new_opt.state_dict()
 for params,group in zip(new_opt.param_groups,out['param_groups']):
  reference=None
  for p,index in zip(params['params'],group['params']):
   name=new_names[id(p)];out['state'][index]=transformed[name];source=source_group[name]
   if reference is None:reference=source
   else:
    for key in ['lr','betas','eps','weight_decay']:assert source[key]==reference[key]
  for key,value in reference.items():
   if key!='params':group[key]=copy.deepcopy(value)
 return out

def metadata(model):
 return dict(options={k:getattr(model.config,k) for k in ('output_sigmoid','low_rank_gates','multi_timescale_init','gate_rank','timescale_spread')},shared_backbone_initial_sha=getattr(model,'shared_backbone_initial_sha',None),kind=model.kind,parameters=sum(p.numel() for p in model.parameters()),width=D,layers=L,ff=FF,heads=H,head_dim=K,slots=M,alpha0=ALPHA,beta0=BETA,model_definition='registered FLA KOMPModel/KOMPForCausalLM; native GatedDeltaNetBlock.forward/FLAModel.forward',fuse_norm=False,fuse_swiglu=False,fused_linear_cross_entropy=True,checkpoint_policy='mixer_only' if model.config.checkpoint_mixer else 'none',head_checkpoint=False,projection='float32',state_compute='float32',chunk_tape='float32' if model.kind=='exact' else 'bfloat16')
