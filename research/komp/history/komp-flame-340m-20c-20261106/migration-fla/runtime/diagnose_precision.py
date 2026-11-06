"""Isolate rounding differences without changing acceptance thresholds."""
import gc,json,types
from pathlib import Path
import torch
from torch.nn import functional as F
from models import ROOT,LM,legacy,map_weights,map_optimizer,mappings,optimizer
from data import Stream

def metrics(a,b):
 a,b=a.detach().float(),b.detach().float();d=a-b
 return dict(rms=float(d.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-7)),normalized_max=float(d.abs().max()/b.abs().max().clamp_min(1)),absolute=float(d.abs().max()))
def rounded(self,x,**kwargs):return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))
def packed(self,x,**kwargs):
 gate,y=F.linear(x,torch.cat((self.gate_proj.weight,self.up_proj.weight),0)).chunk(2,-1)
 return self.down_proj(F.silu(gate)*y)
def mean_ce(self,ids,labels=None,valid_tokens=None):
 hidden=self.model(input_ids=ids,use_cache=False,return_dict=True).last_hidden_state
 if labels is None:return hidden
 target=labels.reshape(-1).clone()
 if valid_tokens is not None:target[valid_tokens:]=-100
 if self._cast_head:hidden=hidden.to(torch.get_autocast_dtype('cuda'))
 mean,_=self._compute_logits_and_loss(hidden,shift_labels=target.reshape(labels.shape),num_items_in_batch=None)
 return mean*target.ne(-100).sum()
def main():
 torch.set_num_threads(2);all_results={}
 for kind in ['exact','block']:
  stop=json.loads((ROOT.parent/f'stop-{kind}.json').read_text());state=torch.load(Path(stop['latest']['path'])/'model.pt',map_location='cpu',weights_only=False)
  stream=Stream(json.loads((ROOT/'configs/data-manifest.json').read_text()),0);ids=stream.next()[:258].cuda()[None]
  old=legacy.LM(kind).cuda().eval();old.load_state_dict(state['model']);oldopt=legacy.optimizer(old);oldopt.load_state_dict(state['optimizer'])
  with torch.autocast('cuda',dtype=torch.bfloat16):oldloss=old(ids[:,:-1],ids[:,1:])/257
  oldloss.backward();gold={n:p.grad.detach().cpu().clone() for n,p in old.named_parameters()};loss=float(oldloss.detach());oldopt.step();delta={n:(p.detach().cpu()-state['model'][n]) for n,p in old.named_parameters()}
  old.cpu();del oldopt,oldloss;gc.collect();torch.cuda.empty_cache();results=[]
  for mode in ['original','mean_ce','mean_cast_ce','rounded_mlp_mean_cast','packed_mlp_mean_cast']:
   model=LM(kind).cuda().eval();model.load_state_dict(map_weights(state['model']));opt=optimizer(model);opt.load_state_dict(map_optimizer(old,state['optimizer'],model,opt))
   if mode!='original':model._cast_head='cast' in mode;model.forward=types.MethodType(mean_ce,model)
   if mode.startswith('rounded'): 
    for layer in model.model.layers:layer.mlp.forward=types.MethodType(rounded,layer.mlp)
   if mode.startswith('packed'):
    for layer in model.model.layers:layer.mlp.forward=types.MethodType(packed,layer.mlp)
   with torch.autocast('cuda',dtype=torch.bfloat16):current=model(ids[:,:-1],ids[:,1:])/257
   current.backward();named=dict(model.named_parameters());rows=[]
   for name,g in gold.items():
    parts=[named[n].grad.detach().cpu() for n,_ in mappings(name)];a=parts[0] if len(parts)==1 else torch.cat(parts,0);rows.append(dict(name='grad/'+name,**metrics(a,g)))
   opt.step()
   for name,g in delta.items():
    parts=[named[n].detach().cpu() for n,_ in mappings(name)];a=(parts[0] if len(parts)==1 else torch.cat(parts,0))-state['model'][name];rows.append(dict(name='delta/'+name,**metrics(a,g)))
   failed=[r for r in rows if not ((r['rms']<.035 and r['normalized_max']<.04) or r['absolute']<=1e-7)]
   row=dict(mode=mode,loss=float(current.detach()),oldloss=loss,failures=len(failed),worst=sorted(rows,key=lambda x:x['rms'],reverse=True)[:10],checks=rows);results.append(row)
   (ROOT/f'evidence/precision-diagnosis-{kind}.json').write_text(json.dumps(results,indent=2));print(kind,mode,'failures',len(failed),'loss',row['loss'],'worst',row['worst'][:2],flush=True)
   del opt,model,current,named;gc.collect();torch.cuda.empty_cache()
  del old,state,gold,delta;gc.collect();torch.cuda.empty_cache();all_results[kind]=results
if __name__=='__main__':main()
