import gc,json,sys
from pathlib import Path
import torch
from models import ROOT,LM,map_weights
kind=sys.argv[1];folder=Path(sys.argv[2]);torch.set_num_threads(2)
state=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);model=LM(kind).cuda();model.load_state_dict(state['model'] if kind=='exact' else map_weights(state['model']));del state;model.train()
block=model.model.layers[0];x=torch.randn(16,2048,1024,device='cuda',requires_grad=True);parameters={p.untyped_storage().data_ptr() for p in model.parameters()};rows=[]
for enabled in [True,False]:
 block.attn.checkpoint_mixer=enabled;seen={};torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
 def pack(t):
  storage=t.untyped_storage()
  if storage.data_ptr() not in parameters:seen[storage.data_ptr()]=storage.nbytes()
  return t
 with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t),torch.autocast('cuda',dtype=torch.bfloat16):y=block(x)[0]
 saved=sum(seen.values());(y.float().square().mean()).backward();assert all(torch.isfinite(p.grad).all() for p in block.parameters() if p.grad is not None)
 rows.append(dict(mixer_checkpoint=enabled,saved_activation_bytes=saved,peak_increment_bytes=torch.cuda.max_memory_allocated()-base,conservative_full_stack_gib=(saved*24+12*1024**3)/1024**3))
 model.zero_grad(set_to_none=True);x.grad=None;del y;gc.collect();torch.cuda.empty_cache()
policy='none' if rows[-1]['conservative_full_stack_gib']<60 else 'mixer_only'
result=dict(status='passed',selected_checkpoint_policy=policy,activation_memory=rows,no_checkpoint_estimated_safe=policy=='none')
(ROOT/f'evidence/memory-{kind}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
