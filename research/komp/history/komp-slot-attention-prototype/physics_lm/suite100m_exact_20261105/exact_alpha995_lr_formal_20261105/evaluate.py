"""Held-out evaluation only after validation selection and final training."""
import argparse,json,os,time,sys,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'snapshot'))
sys.path.insert(0,str(ROOT/'snapshot/suite100m'))
from eval_progress import load_progress,save_progress
from tasks import Factory,EOS
from alpha_models import LM
from ddp_train import digest
from train import RUNS,DATA,Corpus,evaluate,atomic_json

@torch.no_grad()
def generate(model,task,probes,records=None,on_batch=None):
 fac=Factory(task);records=list(records or []);model.eval()
 limits={'mano':1,'depo2':7,'brevo2':201,'lano':730,'capo':24}
 for start in range(len(records),len(probes),8):
  block=probes[start:start+8];seqs=[list(p['prompt']) for p in block];guesses=[[] for _ in block];active=[True]*len(block)
  for step in range(limits[task]):
   t=max(map(len,seqs));ids=torch.zeros(len(block),t,dtype=torch.long,device='cuda')
   for i,seq in enumerate(seqs):ids[i,:len(seq)]=torch.tensor(seq,device='cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):
    h=model(ids);last=h[torch.arange(len(block),device='cuda'),torch.tensor([len(s)-1 for s in seqs],device='cuda')];next_token=model.head(last).argmax(-1).tolist()
   for i,tok in enumerate(next_token):
    if not active[i]:continue
    seqs[i].append(tok);guesses[i].append(tok)
    stop=task=='mano' or (task=='depo2' and tok>3) or (task in ('brevo2','lano') and tok==EOS) or (task=='capo' and tok==fac.bios.vocab['.'])
    active[i]=not stop
   if not any(active):break
  for p,g in zip(block,guesses):records.append(dict(probe=p,guess=g,valid=fac.grade(p,g)))
  if on_batch is not None:on_batch(records)
  print('generation',task,len(records),sum(r['valid'] for r in records),flush=True)
 return records

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();dest=RUNS/'results'/a.run
 summary=json.loads((dest/'summary.json').read_text());cfg=summary['config'];assert cfg['phase']=='final'
 assert cfg['sources']==digest() and cfg['model']['alpha0']==.995
 select=RUNS/'selection'/f"{cfg['task']}-{cfg['model']['kind']}.json";chosen=json.loads(select.read_text())
 assert cfg['lr'] in chosen['allowed_lrs'] and cfg['wd']==chosen['wd']
 saved=torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=False);model=LM(cfg['model']['kind'],cfg['seed']).cuda();model.load_state_dict(saved['model']);del saved
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;started=time.monotonic()
 probes=json.loads((DATA/cfg['task']/'test.probes.json').read_text())[:256]
 checkpoint_stat=(dest/'checkpoint.pt').stat()
 identity=dict(run=a.run,summary_sha256=hashlib.sha256((dest/'summary.json').read_bytes()).hexdigest(),
  probes_sha256=hashlib.sha256(json.dumps(probes,sort_keys=True).encode()).hexdigest(),
  evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()+(ROOT/'snapshot/eval_progress.py').read_bytes()).hexdigest(),
  checkpoint_size=checkpoint_stat.st_size,checkpoint_mtime_ns=checkpoint_stat.st_mtime_ns)
 progress=dest/'eval-progress.json';fac=Factory(cfg['task'])
 resume=load_progress(progress,identity,probes,fac.grade)
 previous_seconds=resume['seconds'] if resume else 0
 metrics=resume['test'] if resume else evaluate(model,Corpus(cfg['task'],'test','final'))
 def persist(records):save_progress(progress,identity,metrics,records,previous_seconds+time.monotonic()-started)
 records=resume['records'] if resume else []
 persist(records)
 print('EVAL_RESUME',a.run,len(records),flush=True)
 records=generate(model,cfg['task'],probes,records,persist)
 atomic_json(dest/'generations.json',records)
 result=dict(status='completed',test=metrics,generation_accuracy=sum(r['valid'] for r in records)/len(records),n_generation=len(records),seconds=previous_seconds+time.monotonic()-started)
 if cfg['task']=='mano':
  result['by_operations']={str(n):dict(n=sum(r['probe']['meta']['operations']==n for r in records),valid=sum(r['valid'] and r['probe']['meta']['operations']==n for r in records)) for n in range(1,17)}
 atomic_json(dest/'heldout.json',result);print('HELDOUT',a.run,json.dumps(result),flush=True)
