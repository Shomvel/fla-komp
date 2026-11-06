"""Four projection modes at one fixed checkpoint; no parameter updates."""
import argparse,datetime,hashlib,importlib,json,math,os,statistics,sys,time,traceback
from pathlib import Path
import torch
import torch.distributed as dist
ROOT=Path(__file__).resolve().parent;T=ROOT/'exact_tuning_20261105'
if not T.exists():T=ROOT.parent/'exact_tuning_20261105'
P=ROOT/'precision_bench_20261105';sys.path[:0]=[str(T),str(P)]
import tune as t
MODES=('fp64','fp32','bf16_projection','bf16_projection_mixed')
NAMES=('embed.weight','head.weight','layers.0.mix.keys','layers.14.mix.keys','layers.0.mix.k.weight','layers.0.mix.decay.weight','layers.0.mix.decay.bias','layers.0.mix.beta.weight','layers.0.mix.out.weight')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def all_finite(value):
 flag=torch.tensor(int(bool(value)),device='cuda');dist.all_reduce(flag,op=dist.ReduceOp.MIN);return bool(flag.item())
def select(candidate,mode):
 op=t.ORIGINAL_EXACT if mode in ('fp64','fp32') else importlib.import_module(mode).exact_komp
 def call(*args,**kwargs):
  kwargs.update(temperature=candidate['temperature'],ridge=candidate['ridge'],projection_dtype='float64' if mode=='fp64' else 'float32')
  return op(*args,**kwargs)
 t.models.exact_komp=call

def timing(model,ids,mask,backward):
 model.train(backward)
 def run():
  if backward:model.zero_grad(set_to_none=True)
  with torch.set_grad_enabled(backward),t.d.amp('bf16'):
   loss,n=t.d.base.loss_sum(model,ids,mask)
   if backward:(loss/n).backward()
  return loss
 for _ in range(5):run()
 torch.cuda.synchronize();batches=[]
 for _ in range(3):
  dist.barrier();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
  start.record()
  for __ in range(10):out=run()
  end.record();end.synchronize()
  ms=torch.tensor(start.elapsed_time(end)/10,device='cuda');dist.all_reduce(ms,op=dist.ReduceOp.MAX);batches.append(ms.item())
 finite=all_finite(torch.isfinite(out).all() and (not backward or all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())))
 return {'status':'measured' if finite else 'rejected','median_ms':statistics.median(batches) if finite else None,'rank_max_ms_each_round':batches if finite else None,'warmup':5,'rounds':3,'iterations_per_round':10,'includes':'LM backbone plus masked head CE; gradient zero+backward for fwd_bwd; no optimizer or DDP gradient communication'}

def main(a):
 rank=dist.get_rank();assert dist.get_world_size()==8
 package=json.loads((P/'variants.json').read_text())
 for file,wanted in package['variants'].items():assert sha(P/file)==wanted,file
 path=T/'results'/a.task/a.candidate/'checkpoint.pt'
 if rank==0:assert sha(path)==a.checkpoint_sha256
 dist.barrier();saved=torch.load(path,map_location='cpu',weights_only=False);config=saved['config'];assert config['sources']==t.sources()
 candidate=config['candidate'];model=t.configure(candidate);model.load_state_dict(saved['model']);del saved
 initial=torch.stack([p.detach().double().sum() for p in model.parameters()])
 corpus=t.d.base.Corpus(a.task,'val','tune');rows=[corpus.row(i) for i in range(rank,16,8)];ids,mask=t.d.base.batch(rows)
 count=mask[:,1:].sum().double();dist.all_reduce(count)
 shapes=[None]*8;dist.all_gather_object(shapes,list(ids.shape))
 reference={};ref_hidden=None
 report={'status':'running','task':a.task,'candidate':candidate,'selection_scope':'fixed previously selected alpha.995 checkpoint, not necessarily final winner','checkpoint':str(path),'checkpoint_sha256':a.checkpoint_sha256,'sources':t.sources(),'script_sha256':sha(Path(__file__)),'variants':package,'local_shapes_by_rank':shapes,'world_size':8,'training_steps':0,'training_tokens':0,'modes':{}}
 for mode in MODES:
  select(candidate,mode);model.train();model.zero_grad(set_to_none=True)
  with torch.no_grad(),t.d.amp('bf16'):hidden=model(ids)[:,:-1][mask[:,1:]].float()
  if not all_finite(torch.isfinite(hidden).all()):
   report['modes'][mode]={'status':'rejected','reason':'nonfinite supervised hidden on fixed real batch','timing':None};continue
  torch.cuda.synchronize();start=time.perf_counter();val=t.evaluate(model,corpus);torch.cuda.synchronize()
  elapsed=torch.tensor(time.perf_counter()-start,device='cuda');dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
  if not all_finite(math.isfinite(val['ce'])):
   report['modes'][mode]={'status':'rejected','reason':'nonfinite full validation CE','validation_ce':None,'timing':None};continue
  model.train();model.zero_grad(set_to_none=True)
  with t.d.amp('bf16'):loss,n=t.d.base.loss_sum(model,ids,mask)
  (loss/count).backward()
  finite=all_finite(torch.isfinite(loss).all() and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
  if not finite:
   report['modes'][mode]={'status':'rejected','reason':'nonfinite fixed batch gradient','validation':val,'timing':None};continue
  total=loss.detach().double();dist.all_reduce(total);gradients={}
  for name,p in model.named_parameters():
   if name not in NAMES:continue
   grad=p.grad.detach().clone();dist.all_reduce(grad)
   if mode=='fp64':reference[name]=grad
   else:
    ref=reference[name];delta=grad-ref;gradients[name]={'finite':bool(torch.isfinite(grad).all()),'rms_relative':float(delta.square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-8)),'normalized_max':float(delta.abs().max()/ref.abs().max().clamp_min(1e-8))}
  hidden_error=None
  if mode=='fp64':ref_hidden=hidden
  else:
   delta=hidden-ref_hidden;sums=torch.stack([delta.double().square().sum(),ref_hidden.double().square().sum()]);dist.all_reduce(sums)
   maxima=torch.stack([delta.abs().max(),ref_hidden.abs().max()]);dist.all_reduce(maxima,op=dist.ReduceOp.MAX)
   hidden_error={'rms_relative':float((sums[0]/sums[1].clamp_min(1e-16)).sqrt()),'normalized_max':float(maxima[0]/maxima[1].clamp_min(1e-8))}
  result={'status':'finite_evaluated','validation':val,'validation_seconds_single_pass_including_compile':elapsed.item(),'fixed_batch_loss':float(total/count),'supervised_hidden_vs_fp64':hidden_error,'gradients_vs_fp64':gradients,'timing':{}}
  result['timing']['forward']=timing(model,ids,mask,False);result['timing']['forward_backward']=timing(model,ids,mask,True)
  report['modes'][mode]=result
  if rank==0:
   print('MODEL_PROJECTION',mode,json.dumps(result),flush=True)
   t.d.base.atomic_json(ROOT/'evidence'/f'bf16-model-{a.task}.json',report)
 after=torch.stack([p.detach().double().sum() for p in model.parameters()]);assert torch.equal(initial,after)
 baseline=json.loads((T/'results'/a.task/a.candidate/'summary.json').read_text())['val']['ce'];assert abs(report['modes']['fp64']['validation']['ce']-baseline)<.005
 report.update(status='completed',weights_unchanged=True,scope='Projection-only exploratory kernels. Fixed checkpoint inference and gradients do not establish training equivalence. Timing is fixed local batch without DDP/optimizer, not an8K single-operator benchmark.')
 if rank==0:t.d.base.atomic_json(ROOT/'evidence'/f'bf16-model-{a.task}.json',report);print('BF16_MODEL_COMPLETE',json.dumps(report),flush=True)
 dist.barrier()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--candidate',required=True);p.add_argument('--checkpoint-sha256',required=True);a=p.parse_args()
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
 dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=15),device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
 try:main(a)
 except BaseException:traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
 else:dist.destroy_process_group()
