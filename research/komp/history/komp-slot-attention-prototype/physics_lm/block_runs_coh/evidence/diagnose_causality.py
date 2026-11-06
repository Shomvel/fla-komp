"""Read-only numerical localization of the first coherence preflight failure."""
import json,os,sys
from pathlib import Path
root=Path('/gpfs/users/linlizhi/komp-coh-trials-20261101')
sys.path.insert(0,str(root/'snapshot/physics_lm'))
import torch
import block_coherence as study
be=study.be
outroot=root/'diagnosis';outroot.mkdir(exist_ok=True)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
rows=be.base.load('train')[:4];ids,_=be.base.batch(rows)
CAPTURE=[];original=study.route

def capture(q,k,v,g,beta,state,**kw):
 result=original(q,k,v,g,beta,state,**kw)
 CAPTURE.append(([x.detach().clone() for x in (q,k,v,g,beta,*state)],result[0].detach().clone(),kw))
 return result
be.block_komp_fast=be.block_komp_flex=capture
results=[]
for theta_tag in ('80','95','99','off'):
 model=be.BlockLM('coh_fifo_'+theta_tag,731).cuda().eval()
 for cut in (31,64,127,129):
  changed=ids.clone();changed[:,cut:]=(changed[:,cut:]+7)%be.base.VOCAB
  CAPTURE.clear()
  with torch.no_grad(),be.base.amp():before=model(ids)
  a=list(CAPTURE);CAPTURE.clear()
  with torch.no_grad(),be.base.amp():after=model(changed)
  b=list(CAPTURE)
  record=dict(theta=theta_tag,cut=cut,model_max=float((before[:,:cut]-after[:,:cut]).abs().max()),layers=[])
  for li,((xa,oa,kw),(xb,ob,_)) in enumerate(zip(a,b)):
   item=dict(layer=li,input_prefix=[float((u[:,:cut]-v[:,:cut]).abs().max()) for u,v in zip(xa[:5],xb[:5])],
    output_prefix=float((oa[:,:cut]-ob[:,:cut]).abs().max()))
   if li==0 and item['output_prefix']>0:
    for dtype in (torch.float32,torch.float64):
     ys=[];traces=[]
     for xx in (xa,xb):
      trace_rows=[];code=study.flex_module._flex.__code__
      lines=Path(code.co_filename).read_text().splitlines()
      target=next(i+1 for i,line in enumerate(lines) if line.strip().startswith('coef = torch.cat((gend[..., None]*S, dend'))
      def tracer(frame,event,arg):
       if frame.f_code is not code:return None
       if event=='line' and frame.f_lineno==target:
        local=frame.f_locals
        trace_rows.append({n:local[n].detach().clone() for n in ('W','A','Hn','Rn','Dp','K','S','target','admitted') if n in local})
       return tracer
      cast=[x.to(dtype) for x in xx];sys.settrace(tracer)
      try:
       with torch.no_grad(): yy=study.FLEX(*cast[:5],tuple(cast[5:]),coherence=study.VARIANTS[model.variant]['theta'],**kw)[0]
      finally:sys.settrace(None)
      ys.append(yy);traces.append(trace_rows)
     detail=dict(output_prefix=float((ys[0][:,:cut]-ys[1][:,:cut]).abs().max()),chunks=[])
     for ci,(ta,tb) in enumerate(zip(*traces)):
      n=min(64,cut-ci*64)
      if n<=0:break
      fields={}
      for name in ta:
       u,v=ta[name],tb[name]
       if name in ('W','target','admitted'):u,v=u[...,:n,:] if name=='W' else u[...,:n],v[...,:n,:] if name=='W' else v[...,:n]
       if name in ('A','Hn','Rn','Dp'):u,v=u[...,:n,:n],v[...,:n,:n]
       fields[name]=float((u.double()-v.double()).abs().max())
      detail['chunks'].append(fields)
     item[str(dtype)]=detail
   record['layers'].append(item)
  results.append(record);print(json.dumps(record),flush=True)
 del model;torch.cuda.empty_cache()
(outroot/'causality.json').write_text(json.dumps(results,indent=2))
