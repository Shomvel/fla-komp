"""Checkpoint-backed numerical and activation-memory acceptance, independent of formal state."""
import gc,json,time
from pathlib import Path
import torch
from models import ROOT,LM,legacy,map_weights,map_optimizer,mappings,optimizer,metadata
from data import Stream

def compare(a,b,name,rows,tolerance=.035):
 a,b=a.detach().double(),b.detach().double();delta=a-b
 rms=float(delta.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-7));mx=float(delta.abs().max()/b.abs().max().clamp_min(1.))
 absolute=float(delta.abs().max());rows.append(dict(name=name,rms=rms,normalized_max=mx,rms_tolerance=tolerance,max_tolerance=.04,near_zero_absolute_tolerance=1e-7));assert torch.isfinite(a).all() and ((rms<tolerance and mx<.04) or absolute<=1e-7),rows[-1]

def main(kind,folder):
 torch.set_num_threads(2);torch.cuda.set_device(0)
 checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
 old=legacy.LM(kind).cuda();old.load_state_dict(checkpoint['model']);old.eval();old_opt=legacy.optimizer(old);old_opt.load_state_dict(checkpoint['optimizer'])
 new=LM(kind).cuda();new.load_state_dict(map_weights(checkpoint['model']));new.eval()
 assert sum(p.numel() for p in old.parameters())==sum(p.numel() for p in new.parameters())==339966848
 opt=optimizer(new);mapped_opt=map_optimizer(old,checkpoint['optimizer'],new,opt);opt.load_state_dict(mapped_opt)
 # Invert the name/slice mapping to verify all weights and both Adam moments exactly.
 target=new.state_dict()
 for name,value in checkpoint['model'].items():
  pieces=[target[n].cpu() for n,_ in mappings(name)];got=pieces[0] if len(pieces)==1 else torch.cat(pieces,0);assert torch.equal(got,value),name
 old_parameters=dict(old.named_parameters());new_parameters=dict(new.named_parameters())
 for name,p in old_parameters.items():
  original=old_opt.state[p]
  targets=[new_parameters[n] for n,_ in mappings(name)]
  for key,value in original.items():
   pieces=[opt.state[target][key] for target in targets]
   if torch.is_tensor(value):
    if value.ndim==0:
     assert all(torch.equal(x.cpu(),value.cpu()) for x in pieces),(name,key)
    else:
     restored=pieces[0] if len(pieces)==1 else torch.cat(pieces,0)
     assert torch.equal(restored.cpu(),value.cpu()),(name,key)
 manifest=json.loads((ROOT/'configs/data-manifest.json').read_text());stream=Stream(manifest,0);ids=torch.cat((stream.next(),stream.next()[1:]))[:258].cuda()[None];rows=[]
 for model in [old,new]:model.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):
  a=old(ids[:,:-1]);b=new(ids[:,:-1]);compare(b,a,'hidden',rows,.02)
  la=old(ids[:,:-1],ids[:,1:])/257;lb=new(ids[:,:-1],ids[:,1:])/257
 compare(lb,la,'fused_ce_loss',rows,.002)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  for valid in [257,129]:compare(new(ids[:,:-1],ids[:,1:],valid)/valid,old(ids[:,:-1],ids[:,1:],valid)/valid,f'loss_mask_{valid}',rows,.002)
  changed=ids[:,:-1].clone();changed[:,128:]=(changed[:,128:]+37)%32000
  compare(new(changed)[:,:128],b[:,:128],'causal_prefix',rows,1e-5)
 la.backward();lb.backward()
 actual=dict(new.named_parameters())
 for name,p in old.named_parameters():
  parts=[actual[n].grad for n,_ in mappings(name)];g=parts[0] if len(parts)==1 else torch.cat(parts,0)
  compare(g,p.grad,'grad/'+name,rows)
 old_opt.step();opt.step()
 for name,p in old.named_parameters():
  initial=checkpoint['model'][name].cuda();pieces=[actual[n].detach() for n,_ in mappings(name)];new_value=pieces[0] if len(pieces)==1 else torch.cat(pieces,0)
  compare(new_value-initial,p.detach()-initial,'adam_parameter_delta/'+name,rows)
 del old,new,opt,old_opt,mapped_opt,a,b,la,lb,old_parameters,new_parameters,actual,initial,new_value,pieces;gc.collect();torch.cuda.empty_cache()
 # B32/T2048 one native block: count saved activation storage and infer full stack capacity.
 new=LM(kind).cuda();new.load_state_dict(map_weights(checkpoint['model']));del checkpoint;new.train()
 block=new.model.layers[0];x=torch.randn(32,2048,1024,device='cuda',requires_grad=True);memory=[]
 parameters={p.untyped_storage().data_ptr() for p in new.parameters()}
 for enabled in [True,False]:
  block.attn.checkpoint_mixer=enabled;seen={};torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
  def pack(t):
   storage=t.untyped_storage();key=storage.data_ptr()
   if key not in parameters:seen[key]=storage.nbytes()
   return t
  with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t),torch.autocast('cuda',dtype=torch.bfloat16):y=block(x)[0]
  saved=sum(seen.values());forward_peak=torch.cuda.max_memory_allocated()-base
  (y.float().square().mean()).backward();memory.append(dict(mixer_checkpoint=enabled,saved_activation_bytes=saved,forward_increment_bytes=forward_peak,peak_increment_bytes=torch.cuda.max_memory_allocated()-base,conservative_full_stack_gib=(saved*24+12*1024**3)/1024**3))
  new.zero_grad(set_to_none=True);x.grad=None;del y;gc.collect();torch.cuda.empty_cache()
 # Use measured conservative stack estimate. No full-stack no-checkpoint attempt above65GiB.
 no_ckpt_safe=memory[-1]['conservative_full_stack_gib']<60
 policy='none' if no_ckpt_safe else 'mixer_only'
 result=dict(status='passed',kind=kind,parameters=339966848,all_weights_mapped_exactly=True,adam_slots_restored_exactly=True,numerical=rows,activation_memory=memory,selected_checkpoint_policy=policy,full_no_checkpoint_admissible=no_ckpt_safe)
 (ROOT/f'evidence/compat-{kind}.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='numerical'}),flush=True)
if __name__=='__main__':
 import sys
 main(sys.argv[1],Path(sys.argv[2]))
