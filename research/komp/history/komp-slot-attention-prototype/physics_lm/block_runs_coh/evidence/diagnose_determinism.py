"""Verify whether scatter-add reduction order explains the suffix-intervention failure."""
import json,sys
from pathlib import Path
root=Path('/gpfs/users/linlizhi/komp-coh-trials-20261101');sys.path.insert(0,str(root/'snapshot/physics_lm'))
import torch
import block_coherence as study
be=study.be;outdir=root/'diagnosis-determinism';outdir.mkdir(exist_ok=True)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
ids,_=be.base.batch(be.base.load('train')[:4]);capture=[];original=study.route

def wrap(q,k,v,g,beta,state,**kw):
 result=original(q,k,v,g,beta,state,**kw)
 capture.append(([x.detach().clone() for x in (q,k,v,g,beta,*state)],kw))
 return result
be.block_komp_fast=be.block_komp_flex=wrap
results=[]
for det in (False,True):
 torch.use_deterministic_algorithms(det)
 for tag in ('80','95','99','off'):
  kind='coh_fifo_'+tag;model=be.BlockLM(kind,731).cuda().eval();capture.clear()
  with torch.no_grad(),be.base.amp(): ref=model(ids)
  xx,kw=capture[0];result=dict(deterministic=det,kind=kind,same_input_model=[],suffix=[],same_input_operator=[])
  for _ in range(5):
   with torch.no_grad(),be.base.amp(): out=model(ids)
   result['same_input_model'].append(float((ref-out).abs().max()))
  with torch.no_grad(): op0=study.FLEX(*xx[:5],tuple(xx[5:]),coherence=study.VARIANTS[kind]['theta'],**kw)
  for _ in range(5):
   with torch.no_grad(): op1=study.FLEX(*xx[:5],tuple(xx[5:]),coherence=study.VARIANTS[kind]['theta'],**kw)
   result['same_input_operator'].append(dict(output=float((op0[0]-op1[0]).abs().max()),state=float((op0[1][1]-op1[1][1]).abs().max())))
  for cut in (31,63,64,65,127,128,129):
   changed=ids.clone();changed[:,cut:]=(changed[:,cut:]+7)%be.base.VOCAB
   with torch.no_grad(),be.base.amp(): out=model(changed)
   result['suffix'].append(dict(cut=cut,max_abs=float((ref[:,:cut]-out[:,:cut]).abs().max())))
  if det:
   assert max(result['same_input_model'])==0
   assert all(r['output']==r['state']==0 for r in result['same_input_operator'])
   assert all(r['max_abs']==0 for r in result['suffix'])
  results.append(result);print(json.dumps(result),flush=True)
  del model;capture.clear();torch.cuda.empty_cache()
(outdir/'results.json').write_text(json.dumps(dict(status='passed',rows=results),indent=2))
