"""Frozen training adapters, correct lm_head, cache-free greedy evaluation."""
import argparse,hashlib,json,math,os,re,sys,time,traceback,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=Path('/gpfs/users/linlizhi')
MODELS=[
 ('kda','kda-flame-340m-20c-20261106','kda',False),
 ('block123','komp-kda-options-ff2272-340m-20c-20261106','block',False),
 ('exact123','komp-kda-options-ff2272-340m-20c-20261106','exact',False),
 ('learned','komp-gdn-slot-learned-340m-20c-20261107','block',False),
 ('learned123','komp-gdn-slot-learned-123-340m-20c-20261107','block',False),
 ('block','komp-flame-340m-20c-20261106/migration-fla-mb16/runtime','block',False),
 ('value','komp-value-decay-block-340m-20c-20261106','block',False),
 ('exact_partial','komp-flame-340m-20c-20261106/migration-fla-mb16/runtime','exact',True),
]
def atomic(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def score(row,text):
 if row['metric']=='string_match_all':
  text=re.sub(r'[\x00-\x1f]','\n',text.strip()).strip()
  return sum(x.lower() in text.lower() for x in row['labels'])/len(row['labels'])
 phone=row['task'].startswith('phonebook')
 return max(int(bool(re.search(re.escape(x.strip() if phone else x),text.strip() if phone else text,re.I))) for x in row['labels'])

def context_limit(row, model_max_length):
 if row['task'].startswith('niah'):return 65536
 if row['task'].startswith('phonebook'):return 32768
 return model_max_length

def context_ids(row, model_max_length):
 # Reserve the task's full budget even when acceptance requests fewer new tokens.
 budget=row['generation'].get('max_gen_toks',128)
 limit=context_limit(row,model_max_length)-budget
 assert limit>0
 return row['input_ids'][-limit:]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--smoke-only',action='store_true');ap.add_argument('--shard',type=int,choices=range(3),required=True);args=ap.parse_args()
 name,folder,kind,partial=MODELS[args.index];runroot=BASE/folder;run=runroot/'runs'/kind
 from sharding import lock_worker,batch_path,check_batch,write_atomic,digest,publish_batch
 worker_lock=lock_worker(name,args.shard)
 # Gate numerical work on an independent SSH-node claim/publication probe.
 probe=worker_lock/'publish-probe.json'
 publish_batch(probe,dict(model=name,shard=args.shard,batch_key='probe',plan_sha='probe',checkpoint_sha='probe',results=[]))
 while not (ROOT/'sharded/cross-node-verified-r4b.json').exists():time.sleep(2)
 (ROOT/'sharded/evidence-v2').mkdir(parents=True,exist_ok=True)
 plan=json.loads((ROOT/'sharded/plan.json').read_text());plan_sha=digest(ROOT/'sharded/plan.json')
 assert args.shard in range(plan['shards'])

 os.environ['CUDA_VISIBLE_DEVICES']=str(args.index)
 os.environ['TRITON_CACHE_DIR']=str(ROOT/'sharded/triton-cache'/f'{name}-s{args.shard}')
 import torch
 from tokenizers import Tokenizer
 torch.set_num_threads(2);torch.manual_seed(42);torch.cuda.set_device(0)
 sys.path.insert(0,str(runroot))
 import models
 completion=run/'complete.json'
 if not partial and not completion.exists():
  atomic(ROOT/'sharded/evidence-v2'/f'{name}-s{args.shard}-waiting.json',dict(status='waiting_for_final_checkpoint',run=str(run)))
  while not completion.exists():time.sleep(30)
 final=json.loads(completion.read_text()) if completion.exists() else None
 latest=json.loads((run/'latest.json').read_text());ck=Path(latest['path'])/'model.pt'
 if not partial:assert final['status']=='completed' and latest['tokens']==final['supervised_tokens']==6800000000
 source_sha=sha(runroot/'source-manifest.json')
 if final:
  assert final['source_sha']==source_sha
  assert final['data_sha']==sha(runroot/'configs/data-manifest.json')
 manifest=json.loads((runroot/'source-manifest.json').read_text())
 for f,h in manifest['files'].items():assert sha(runroot/f)==h,f
 request_manifest=json.loads((ROOT/'evidence/requests-manifest.json').read_text())
 tokenizer=Tokenizer.from_file(str(Path(request_manifest['tokenizer'])/'tokenizer.json'))
 assert sha(Path(request_manifest['tokenizer'])/'tokenizer.json')==request_manifest['tokenizer_sha']
 eos=request_manifest['eos_token_id']
 model=models.TrainingKDA() if kind=='kda' else models.LM(kind)
 state=torch.load(ck,map_location='cpu',weights_only=False,mmap=True)
 assert state['source_sha']==source_sha and state['data_sha']==sha(runroot/'configs/data-manifest.json') and state['step']==latest['step'] and state['tokens']==latest['tokens']
 model.load_state_dict(state['model'],strict=True);del state
 model=model.cuda().eval()
 model_max_length=next((getattr(model.config,k) for k in ['n_positions','max_position_embeddings','n_ctx'] if getattr(model.config,k,None)),None)
 assert model_max_length==2048,(name,model_max_length)
 atomic(ROOT/'sharded/evidence-v2'/f'{name}-s{args.shard}-runtime-config.json',model.config.to_dict())
 protocol=json.loads((ROOT/'evidence/generation-contract-audit.json').read_text())
 assert protocol['status']=='passed'

 receipt=dict(model=name,kind=kind,checkpoint=str(ck),checkpoint_sha=sha(ck),step=latest['step'],tokens=latest['tokens'],partial=partial,source_sha=source_sha,parameters=sum(p.numel() for p in model.parameters()),tokenizer_sha=request_manifest['tokenizer_sha'],request_manifest_sha=sha(ROOT/'evidence/requests-manifest.json'),torch=torch.__version__,gpu=torch.cuda.get_device_name(),inference='FP32 weights with BF16 autocast; cache-free complete-prefix recomputation; exact-length inputs with no padding; output at true last position; no chat template, no BOS as original HFLM',eval_script_sha=sha(Path(__file__)),model_max_length=model_max_length,context_caps=dict(niah=65536,phonebook=32768,based=model_max_length),protocol_audit_sha=sha(ROOT/'evidence/generation-contract-audit.json'))
 expected=plan['models'][name]
 for key in ['checkpoint_sha','source_sha','tokens','step','request_manifest_sha','protocol_audit_sha']:
  assert receipt[key]==expected[key],(key,receipt[key],expected[key])
 atomic(ROOT/'sharded/evidence-v2'/f'{name}-s{args.shard}-load.json',receipt)
 print('LOADED',name,flush=True)
 @torch.inference_mode()
 def logits(tokens,positions):
  with torch.autocast('cuda',dtype=torch.bfloat16):
   hidden=model(tokens)
   assert isinstance(hidden,torch.Tensor) and hidden.shape[:2]==tokens.shape
   selected=hidden[torch.arange(tokens.shape[0],device='cuda'),positions]
   result=model.lm_head(selected).float()
  assert result.shape==(tokens.shape[0],32000) and bool(torch.isfinite(result).all())
  return result
 @torch.inference_mode()
 def generate(rows,cap=None):
  limits=[min(r['generation'].get('max_gen_toks',128),cap or 100000) for r in rows]
  # Historical HF generation left truncates only beyond the model context cap.
  contexts=[context_ids(r,model_max_length) for r in rows]
  assert all(r['generation']==rows[0]['generation'] for r in rows)
  b=len(rows);assert len({len(x) for x in contexts})==1; width=max(len(x)+n for x,n in zip(contexts,limits))
  ids=torch.full((b,width),eos,device='cuda',dtype=torch.long)
  lengths=torch.tensor([len(x) for x in contexts],device='cuda')
  for i,x in enumerate(contexts):ids[i,:len(x)]=torch.tensor(x,device='cuda')
  done=[False]*b;generated=[[] for _ in rows];texts=['']*b
  start=time.perf_counter()
  for t in range(max(limits)):
   active=[i for i in range(b) if not done[i]]
   current=max(int(lengths[i]) for i in active)
   pred=logits(ids[:,:current].contiguous(),lengths-1).argmax(-1).tolist()
   for i,token in enumerate(pred):
    if done[i]:continue
    generated[i].append(token);ids[i,lengths[i]]=token;lengths[i]+=1
    text=tokenizer.decode(generated[i],skip_special_tokens=True)
    stops=rows[i]['generation'].get('until',[])
    if isinstance(stops,str):stops=[stops]
    stops=list(stops)+[tokenizer.decode([eos],skip_special_tokens=False)]
    cuts=[text.index(s) for s in stops if s and s in text]
    if cuts:text=text[:min(cuts)]
    texts[i]=text
    done[i]=bool(cuts) or token==eos or t+1>=limits[i]
   if all(done):break
  torch.cuda.synchronize()
  return [dict(id=r['id'],case=r['case'],index=r['index'],prediction=text,generated_ids=gen,score=score(r,text),prompt_tokens=len(c),truncated_prompt_tokens=len(r['input_ids'])-len(c),generated_tokens=len(gen)) for r,c,text,gen in zip(rows,contexts,texts,generated)],time.perf_counter()-start
 # Read only a tiny real sample per representative task family for acceptance.
 samples=[]
 for case in ['phonebook_32','based_swde','based_fda','niah_single_1_1024','niah_single_1_8192']:
  p=ROOT/'requests'/f'{case}.jsonl'
  with p.open() as f:samples.append(json.loads(next(f)))
 parity=[]
 for row in samples:
  x=torch.tensor(context_ids(row,model_max_length),device='cuda')[None];n=x.shape[1]
  a=logits(x,torch.tensor([n-1],device='cuda'));b=logits(x.clone(),torch.tensor([n-1],device='cuda'))
  diff=a-b
  parity.append(dict(case=row['case'],tokens=n,absolute=float(diff.abs().max()),top1_equal=int(a.argmax())==int(b.argmax()),true_length=True))
  assert bool(torch.equal(a,b)),parity[-1]
 smoke=[]
 for row in samples[:3]:
  result,seconds=generate([row],cap=4);smoke.append(dict(result=result,seconds=seconds))
 atomic(ROOT/'sharded/evidence-v2'/f'{name}-s{args.shard}-acceptance.json',dict(status='passed',receipt=receipt,exact_length_repeatability=parity,smoke=smoke,scope='real prompts, strict checkpoint loading, finite last-token logits, greedy full-prefix decoding and exact-length repeatability; no cache'))
 print('ACCEPTANCE_PASS',name,flush=True)
 if args.smoke_only:return
 # Fixed original batches are the indivisible shard unit.
 for case in plan['cases']:
  p=ROOT/'requests'/f"{case['case']}.jsonl";assert sha(p)==case['sha256']
  rows={r['id']:r for r in map(json.loads,p.read_text().splitlines())}
  for batch in case['batches']:
   if batch['shard']!=args.shard:continue
   out=batch_path(name,batch)
   if out.exists():
    old=json.loads(out.read_text());check_batch(old,batch,plan_sha)
    assert old['checkpoint_sha']==receipt['checkpoint_sha']
    continue
   # Batch order, shape and generation arguments match original unsharded r3.
   selected=[rows[i] for i in batch['ids']]
   results,seconds=generate(selected)
   prior=ROOT/'sharded/partial-reference'/name/f"{batch['key']}.json"
   if prior.exists():
    old={x['id']:x for x in json.loads(prior.read_text())}
    for result in results:
     if result['id'] in old:assert result==old[result['id']],('partial_replay_mismatch',result['id'])
   value=dict(model=name,shard=args.shard,batch_key=batch['key'],plan_sha=plan_sha,checkpoint_sha=receipt['checkpoint_sha'],eval_script_sha=receipt['eval_script_sha'],results=results,seconds=seconds,inherited=False)
   check_batch(value,batch,plan_sha);publish_batch(out,value)
   progress=dict(model=name,shard=args.shard,case=case['case'],batch_key=batch['key'],samples=len(results),latest_batch_seconds=seconds,time=time.time(),plan_sha=plan_sha)
   atomic(ROOT/'sharded/evidence-v2'/f'{name}-s{args.shard}-progress.json',progress)
   print(json.dumps(progress),flush=True)
 atomic(ROOT/'sharded/evidence-v2'/f'{name}-s{args.shard}-complete.json',dict(status='completed',model=name,shard=args.shard,receipt=receipt,plan_sha=plan_sha))
if __name__=='__main__':
 try:main()
 except Exception as e:
  name=MODELS[int(sys.argv[sys.argv.index('--index')+1])][0]
  shard=int(sys.argv[sys.argv.index('--shard')+1])
  atomic(ROOT/'sharded/evidence-v2'/f'{name}-s{shard}-failure.json',dict(status='failed',error=str(e),traceback=traceback.format_exc(),time=time.time()))
  raise
