"""Independent-task training with bounded logit memory and resumable optimizer state."""
import argparse,hashlib,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from tasks import Factory,TASKS
from models import LM,KINDS,metadata,optimizer

DATA=Path(os.environ['SUITE_DATA']);RUNS=Path(os.environ['SUITE_RUNS'])

def sources():
 root=Path(__file__).parent
 files=[*root.glob('*.py'),*Path(os.environ['KOMP_SOURCE']).joinpath('komp_slot').rglob('*.py')]
 return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}

def atomic_json(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,path)

class Corpus:
 def __init__(self,task,split,budget='tune'):
  root=DATA/task
  offsets='tune' if split=='train' and budget=='tune' else split
  self.offsets=np.memmap(root/f'{offsets}.offsets.u64',dtype='<u8',mode='r')
  self.ids=np.memmap(root/f'{split}.tokens.u16',dtype='<u2',mode='r');self.mask=np.memmap(root/f'{split}.mask.u8',dtype='u1',mode='r')
 def __len__(self):return len(self.offsets)-1
 def row(self,index):
  a,b=map(int,self.offsets[index:index+2]);return dict(input_ids=self.ids[a:b],loss_mask=self.mask[a:b])
 def batches(self,seed,bsz=128):
  rng=np.random.default_rng(seed);order=rng.permutation(len(self))
  lengths=np.diff(self.offsets)
  for start in range(0,len(order),4096):
   pool=order[start:start+4096];pool=pool[np.argsort(lengths[pool],kind='stable')]
   groups=[pool[j:j+bsz] for j in range(0,len(pool),bsz)];rng.shuffle(groups)
   yield from groups

def batch(rows):
 t=max(len(r['input_ids']) for r in rows);t=math.ceil(t/16)*16
 ids=torch.zeros(len(rows),t,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids,dtype=torch.bool)
 for i,r in enumerate(rows):
  n=len(r['input_ids']);ids[i,:n]=torch.as_tensor(np.array(r['input_ids']),device='cuda');mask[i,:n]=torch.as_tensor(np.array(r['loss_mask']),device='cuda',dtype=torch.bool)
 return ids,mask

def loss_sum(model,ids,mask):
 h=model(ids)[:,:-1][mask[:,1:]];gold=ids[:,1:][mask[:,1:]]
 def ce(a,b):return F.cross_entropy(F.linear(a,model.head.weight).float(),b,reduction='sum')
 result=h.sum()*0
 for start in range(0,len(gold),256):
  a,b=h[start:start+256],gold[start:start+256]
  result=result+(checkpoint(ce,a,b,use_reentrant=False) if model.training else ce(a,b))
 return result,len(gold)

@torch.no_grad()
def evaluate(model,corpus):
 model.eval();total=0.;count=0
 for offset in range(0,len(corpus),4):
  ids,mask=batch([corpus.row(i) for i in range(offset,min(offset+4,len(corpus)))])
  with torch.autocast('cuda',dtype=torch.bfloat16):loss,n=loss_sum(model,ids,mask)
  total+=float(loss);count+=n
 model.train();return dict(ce=total/count,supervised_positions=count)

def schedule(step,total,peak):
 warm=max(1,int(total*.05));factor=step/warm if step<=warm else .1+.9*(1+math.cos(math.pi*(step-warm)/max(1,total-warm)))/2
 t=(step-1)/max(1,total-1);blend=.5-.5*math.cos(math.pi*min(1,max(0,(t-.25)/.5)))
 return peak*factor,math.exp((1-blend)*math.log(.1)+blend*math.log(.3))

def preflight(kind):
 model=LM(kind,731).cuda();report=dict(model=metadata(model),checks=[],sources=sources())
 assert 99_000_000<report['model']['parameters']<101_000_000,report['model']
 # Whole backbone causality; direct mixer SWA window excludes convolution's three-token receptive extension.
 ids=torch.randint(1,100,(2,129),device='cuda');model.eval()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  before=model(ids);changed=ids.clone();changed[:,65:]+=17;after=model(changed)
  torch.testing.assert_close(before[:,:65],after[:,:65],atol=1e-5,rtol=1e-5)
 report['checks'].append('full-model future intervention passed')
 model.train();opt=optimizer(model,1e-3,.01)
 rows=[Factory('mano').row(random.Random(923+i)) for i in range(4)];ids,mask=batch(rows);losses=[]
 for step in range(8):
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):ls,n=loss_sum(model,ids,mask);loss=ls/n
  loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
  assert torch.isfinite(gn) and torch.isfinite(loss)
  opt.step();losses.append(float(loss.detach()))
 assert losses[-1]<losses[0],losses
 # Verify checkpoint and optimizer restore, then a finite real-data backward.
 import io
 model.eval()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  before=model(ids)
 probe=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),probe)
 probe.seek(0);saved=torch.load(probe,map_location='cpu',weights_only=True)
 model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  restored=model(ids)
 torch.testing.assert_close(restored,before,atol=1e-5,rtol=1e-5)
 assert all(int(state['step'])==8 for state in opt.state.values())
 del saved,probe,before,restored
 model.train();opt.zero_grad(set_to_none=True)
 corpus=Corpus('depo2','train','tune')
 realrows=[corpus.row(i) for i in range(4)];ids,mask=batch(realrows)
 with torch.autocast('cuda',dtype=torch.bfloat16):
  ls,n=loss_sum(model,ids,mask);real_loss=ls/n
 real_loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
 assert torch.isfinite(real_loss) and torch.isfinite(gn)
 opt.step()
 report['checks']+=['model and AdamW checkpoint restore passed','real Depo2 batch forward/backward/update passed']
 report.update(status='passed',losses=losses,real_batch_shape=list(ids.shape),real_batch_loss=float(real_loss.detach()),peak_bytes=torch.cuda.max_memory_allocated())
 dest=RUNS/'preflight';dest.mkdir(parents=True,exist_ok=True);atomic_json(dest/f'{kind}.json',report)
 print('PREFLIGHT',json.dumps(report['model']),losses,flush=True)

def train(args):
 ticket=json.loads((RUNS/'preflight'/f'{args.kind}.json').read_text());assert ticket['status']=='passed' and ticket['sources']==sources()
 name=f'{args.task}-{args.kind}-{args.phase}-lr{args.lr:g}-wd{args.wd:g}-seed{args.seed}'
 dest=RUNS/'results'/name;dest.mkdir(parents=True,exist_ok=True)
 assert not (dest/'summary.json').exists(),'completed run must not be rerun'
 model=LM(args.kind,args.seed).cuda();opt=optimizer(model,args.lr,args.wd)
 corpus=Corpus(args.task,'train',args.phase);val=Corpus(args.task,'val',args.phase)
 groups=list(corpus.batches(12345));total=len(groups);done=0;consumed=0;supervised=0;elapsed=0.
 config=dict(name=name,task=args.task,phase=args.phase,seed=args.seed,lr=args.lr,wd=args.wd,batch=128,microbatch_token_cap=8192,steps=total,model=metadata(model),sources=sources(),selection='final validation CE only')
 ckpt=dest/'checkpoint.pt'
 if ckpt.exists():
  old=torch.load(ckpt,map_location='cpu',weights_only=False);assert old['config']==config
  model.load_state_dict(old['model']);opt.load_state_dict(old['optimizer']);done=old['step'];consumed=old['consumed'];supervised=old['supervised'];elapsed=old['elapsed'];del old
 atomic_json(dest/'config.json',config);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
 with (dest/'metrics.jsonl').open('a',buffering=1) as log:
  for j in range(done,total):
   indices=groups[j];rows=[corpus.row(int(i)) for i in indices];nvalid=sum(int(np.sum(r['loss_mask'][1:])) for r in rows)
   rate,mult=schedule(j+1,total,args.lr)
   for group in opt.param_groups:group['lr']=rate*(mult if group['residual_output'] else 1.)
   opt.zero_grad(set_to_none=True);ls_total=0.
   longest=max(len(r['input_ids']) for r in rows);micro=min(128,2**int(math.log2(max(1,8192/math.ceil(longest/16)/16))))
   for offset in range(0,len(rows),micro):
    ids,mask=batch(rows[offset:offset+micro])
    with torch.autocast('cuda',dtype=torch.bfloat16):ls,n=loss_sum(model,ids,mask);loss=ls/nvalid
    loss.backward();ls_total+=float(ls.detach())
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
   assert torch.isfinite(norm) and math.isfinite(ls_total),(name,j,ls_total,float(norm))
   opt.step();consumed+=sum(len(r['input_ids'])-1 for r in rows);supervised+=nvalid
   seconds=elapsed+time.monotonic()-start
   if j==done or (j+1)%25==0 or j+1==total:
    record=dict(step=j+1,total=total,loss=ls_total/nvalid,grad_norm=float(norm),lr=rate,tokens=consumed,seconds=seconds);log.write(json.dumps(record)+'\n');print(name,json.dumps(record),flush=True)
   if (j+1)%250==0 or j+1==total:
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),config=config,step=j+1,consumed=consumed,supervised=supervised,elapsed=seconds),dest/'checkpoint.tmp.pt');os.replace(dest/'checkpoint.tmp.pt',ckpt)
 expected=int(np.diff(corpus.offsets).sum())-len(corpus);assert consumed==expected,(consumed,expected)
 final_val=evaluate(model,val)
 summary=dict(status='completed',config=config,val=final_val,tokens=consumed,supervised_positions=supervised,train_seconds=elapsed+time.monotonic()-start,peak_bytes=torch.cuda.max_memory_allocated())
 atomic_json(dest/'summary.json',summary);print('COMPLETE',name,final_val,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--kind',required=True,choices=KINDS);p.add_argument('--task',choices=TASKS);p.add_argument('--phase',choices=['tune','final'],default='tune');p.add_argument('--seed',type=int,default=731);p.add_argument('--lr',type=float,default=.0005);p.add_argument('--wd',type=float,default=.01);a=p.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 if a.preflight:preflight(a.kind)
 else:train(a)
