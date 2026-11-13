"""New-source KDA operator, real-text DDP, loss alignment and full recovery acceptance."""
import argparse,contextlib,gc,hashlib,importlib.metadata,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from models import ROOT,LM,optimizer,metadata
from data import Stream
from train import fingerprint,atomic_json,save,resume,rng_state,restore_rng

def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def model_hash(model):
 h=hashlib.sha256()
 for n,p in model.state_dict().items():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def optimizer_hash(opt):
 h=hashlib.sha256()
 for index,state in sorted(opt.state_dict()['state'].items()):
  for key,value in sorted(state.items()):
   h.update(f'{index}:{key}'.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes() if torch.is_tensor(value) else str(value).encode())
 h.update(json.dumps(opt.state_dict()['param_groups'],sort_keys=True).encode());return h.hexdigest()
def diff(a,b):
 a,b=a.detach().float(),b.detach().float();d=a-b
 return dict(relative_rms=float(d.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-7)),normalized_max=float(d.abs().max()/b.abs().max().clamp_min(1.)),absolute=float(d.abs().max()))
def accepted(r):return (r['relative_rms']<.035 and r['normalized_max']<.04) or r['absolute']<=1e-7


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['exact','block'],required=True);args=ap.parse_args();kind=args.kind
 rank=int(os.environ['RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2);dist.init_process_group('nccl');assert dist.get_world_size()==8
 random.seed(42);np.random.seed(42);torch.manual_seed(42)
 source=fingerprint();manifest=json.loads((ROOT/'configs/data-manifest.json').read_text());data_sha=hashlib.sha256((ROOT/'configs/data-manifest.json').read_bytes()).hexdigest();batch_sha=hashlib.sha256((ROOT/'configs/batching.json').read_bytes()).hexdigest()
 stream=Stream(manifest,rank)
 if rank==0:
  from google.protobuf.internal.api_implementation import Type
  gold=json.loads((ROOT/'evidence/tokenizer-parity-samples.json').read_text())+json.loads((ROOT/'evidence/tokenizer-edge-gold.json').read_text())
  for sample in gold:assert stream.tokenizer.encode(sample['text'],add_special_tokens=True).ids==sample['ids']
  audit=json.loads((ROOT/'evidence/data-audit.json').read_text())
  assert hashlib.sha256((ROOT/'evidence/data-audit.json').read_bytes()).hexdigest()==manifest['data_audit_sha256']
  for name,sha in audit['tokenizer_files'].items():assert hashlib.sha256((Path(manifest['tokenizer_path'])/name).read_bytes()).hexdigest()==sha
  assert hashlib.sha256(Path(manifest['excluded_text_sha256_path']).read_bytes()).hexdigest()==audit['exclusion_sha256']
  for split in ['validation','test']:assert hashlib.sha256(Path(manifest[split+'_tokens_path']).read_bytes()).hexdigest()==audit['token_files'][split]['sha256']
  atomic_json(ROOT/f'evidence/environment-{kind}.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),protobuf=Type(),versions={n:importlib.metadata.version(n) for n in ['triton','transformers','tokenizers']},tokenizer_gold_passed=len(gold)))
  from option_checks import run_checks
  run_checks(kind,stream);print('OPTIONS_OPERATOR_AND_TOKENIZER_PASS',flush=True)
 dist.barrier()
 model=LM(kind).cuda();opt=optimizer(model);initial=model_hash(model)
 torch.cuda.empty_cache()
 hashes=[None]*8;dist.all_gather_object(hashes,initial);assert len(set(hashes))==1
 if rank==0:
  atomic_json(ROOT/f'evidence/model-{kind}.json',metadata(model));model.config.save_pretrained(ROOT/'configs'/f'model-{kind}')
 print(f'RANK{rank}_MODEL_READY',flush=True)
 # Causality and exact alignment of fused CE to already-shifted real-text labels.
 state=stream.state_dict();ids=stream.next().cuda()[None];stream.load_state_dict(state)
 model.eval()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  x=ids[:,:-1];alter=x.clone();alter[:,1024:]=(alter[:,1024:]+37)%32000
  hidden=model(x);changed=model(alter)
  causal=float((hidden[:,:1024]-changed[:,:1024]).abs().max());assert causal<1e-5
  short=x[:,:129];target=ids[:,1:130];h=model(short);logits=model.lm_head(h).float()
  for valid in [129,77]:
   labels=target.clone();labels.reshape(-1)[valid:]=-100
   direct=F.cross_entropy(logits.reshape(-1,32000),labels.reshape(-1),ignore_index=-100,reduction='sum')/valid
   native=model(short,target,valid)/valid;assert abs(float(native-direct))<.005,(native,direct)
 del hidden,changed,h,logits;model.train();gc.collect();torch.cuda.empty_cache()
 wrapped=DDP(model,device_ids=[rank],broadcast_buffers=False,gradient_as_bucket_view=True)
 out=ROOT/f'preflight-checkpoints-{kind}';out.mkdir(exist_ok=True);cases=[]
 for case,budget in [('full',65536),('final_tail',63616)]:
  torch.cuda.reset_peak_memory_stats();start=time.perf_counter();opt.zero_grad(set_to_none=True);loss_sum=torch.zeros((),device='cuda',dtype=torch.float64)
  for micro in range(2):
   count=min(32768,budget-micro*32768);batch=torch.stack([stream.next() for _ in range(16)]).cuda()
   sync=wrapped.no_sync() if micro==0 else contextlib.nullcontext()
   with sync,torch.autocast('cuda',dtype=torch.bfloat16):raw=wrapped(batch[:,:-1],batch[:,1:],count);loss_sum+=raw.detach().double();(raw/budget).backward()
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
  old=model_hash(model);opt.step();assert old!=model_hash(model)
  assert all(torch.isfinite(p).all() for p in model.parameters())
  hashes=[None]*8;dist.all_gather_object(hashes,model_hash(model));assert len(set(hashes))==1
  dist.all_reduce(loss_sum);peak=torch.tensor(torch.cuda.max_memory_allocated(),device='cuda');dist.all_reduce(peak,op=dist.ReduceOp.MAX);assert int(peak)<72*2**30
  torch.cuda.synchronize();cases.append(dict(case=case,loss=float(loss_sum/(8*budget)),global_valid_tokens=8*budget,grad_norm=float(norm),seconds=time.perf_counter()-start,peak_bytes=int(peak),replicas_identical=True))
  if rank==0:print(json.dumps(cases[-1]),flush=True)
 # Save the trained trial including Adam, then deliberately perturb all states and reload.
 save(out,model,opt,stream,2,8*(65536+63616),rank,source,data_sha)
 mh,oh=model_hash(model),optimizer_hash(opt);saved_rng=rng_state();saved_stream=stream.state_dict()
 expected=stream.next();stream.load_state_dict(saved_stream)
 probes=(torch.rand(8),torch.rand(8,device='cuda'),random.random(),np.random.rand(8));restore_rng(saved_rng)
 with torch.no_grad():next(model.parameters()).add_(1)
 first=next(iter(opt.state.values()));first['exp_avg'].zero_();stream.next();torch.rand(11,device='cuda');random.random();np.random.rand(7)
 resume(out,model,opt,stream,rank,source,data_sha)
 assert model_hash(model)==mh and optimizer_hash(opt)==oh
 assert torch.equal(stream.next(),expected);stream.load_state_dict(saved_stream)
 assert torch.equal(torch.rand(8),probes[0]) and torch.equal(torch.rand(8,device='cuda'),probes[1]) and random.random()==probes[2] and np.array_equal(np.random.rand(8),probes[3])
 restore_rng(saved_rng)
 if rank==0:
  atomic_json(ROOT/f'evidence/preflight-{kind}.json',dict(status='passed',kind=kind,source_sha=source,data_sha=data_sha,batching_sha=batch_sha,parameters=metadata(model)['parameters'],initial_model_sha=initial,shared_backbone_initial_sha=model.shared_backbone_initial_sha,cases=cases,causal_absolute=causal,aligned_full_tail_ce=True,all_gradients_parameters_finite=True,full_model_adam_rng_loader_restore=True,replicas_identical=True,trial_tokens_counted=0,formal_start='Fresh seed42 model and original data cursor, not trial checkpoint'))
  print(f'KOMP_OPTIONS_{kind}_PREFLIGHT_PASS',flush=True)
 stream.close();dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
