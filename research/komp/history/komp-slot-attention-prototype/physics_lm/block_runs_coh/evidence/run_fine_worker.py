"""Near-one adaptive extension; separate immutable snapshot and result namespace."""
import hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path('/gpfs/users/linlizhi/komp-coh-trials-20261101');SNAP=ROOT/'snapshot-fine';RESULT=ROOT/'results-fine'
MODE=os.environ['FINE_MODE'];INDEX=int(os.environ['FINE_INDEX']);WORKER=ROOT/'fine-workers'/f'{MODE}-{INDEX}'
EXPECTED=json.loads((ROOT/'fine-snapshot-sha256.json').read_text())

def dump(p,obj):
 t=p.with_suffix(f'.tmp-{MODE}-{INDEX}');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def verify():
 for n,h in EXPECTED.items():assert hashlib.sha256((SNAP/n).read_bytes()).hexdigest()==h,n
def wait(p):
 while not p.exists():
  if (ROOT/'fine-failure.json').exists():raise RuntimeError('Fine worker failed')
  time.sleep(5)
 if (ROOT/'fine-failure.json').exists():raise RuntimeError('Fine worker failed')
def run(args,name):
 verify()
 if (ROOT/'fine-failure.json').exists():raise RuntimeError('Fine worker failed; no new runs')
 dump(WORKER/'active.json',dict(name=name,command=args));start=time.time()
 with (WORKER/f'{name}.log').open('w') as output:
  p=subprocess.run([sys.executable,*args],cwd=SNAP/'physics_lm',stdout=output,stderr=subprocess.STDOUT)
 row=dict(name=name,command=args,exit_code=p.returncode,elapsed_seconds=time.time()-start)
 with (WORKER/'progress.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
 assert p.returncode==0,row
 print(json.dumps(row),flush=True)
def trial(kind,seed):
 name=f'{kind}-seed{seed}';run(['block_coherence.py','--kind',kind,'--seed',str(seed)],name)
 folder=RESULT/'results'/name;r=json.loads((folder/'summary.json').read_text())
 assert r['status']=='completed' and r['unique_tokens']==10001108 and r['reload_validation_ce_error']<1e-4
 if seed==731:run(['block_coherence.py','--diagnose','--kind',kind,'--seed',str(seed)],name+'-diagnose')
 (folder/'model.pt').unlink()
def main():
 import torch,triton,google.protobuf
 from google.protobuf.internal.api_implementation import Type
 WORKER.mkdir(parents=True,exist_ok=True);verify()
 dump(WORKER/'environment.json',dict(torch=str(torch.__version__),triton=triton.__version__,gpu=torch.cuda.get_device_name(),python=sys.version,protobuf=google.protobuf.__version__,protobuf_backend=Type()))
 if MODE=='pilot':
  if INDEX==0:
   run(['block_coherence.py','--preflight'],'preflight-fine6')
   dump(ROOT/'fine-preflight-ready.json',dict(status='passed'))
  wait(ROOT/'fine-preflight-ready.json')
  tag=('995','999')[INDEX];trial('coh_fifo_'+tag,731)
  dump(WORKER/'completion.json',dict(status='completed',runs=[['coh_fifo_'+tag,731]]))
 else:
  decision=json.loads((ROOT/'fine-screening.json').read_text());tags=decision['expand_tags']
  tasks=[('coh_fifo_'+tag,seed) for seed in range(732,738) for tag in tags]
  for kind,seed in tasks[INDEX::12]:trial(kind,seed)
  dump(WORKER/'e1-done.json',dict(status='completed'))
  if INDEX==0:
   for i in range(12):wait(ROOT/'fine-workers'/f'expand-{i}'/'e1-done.json')
   ranking=[]
   for tag in ['99',*tags]:
    directory=ROOT/'results' if tag=='99' else RESULT
    rr=[json.loads((directory/'results'/f'coh_fifo_{tag}-seed{s}'/'summary.json').read_text()) for s in range(731,738)]
    ranking.append(dict(tag=tag,mean_content_ce=statistics.mean(x['confirm']['mean_content_ce'] for x in rr)))
   winner=min(ranking,key=lambda x:x['mean_content_ce'])['tag']
   dump(ROOT/'fine-selection.json',dict(status='locked',selected_tag=winner,ranking=ranking,selection='seven-seed confirm mean_content_ce among .99 and pilot-surviving finer thresholds',adaptive=True))
  wait(ROOT/'fine-selection.json');winner=json.loads((ROOT/'fine-selection.json').read_text())['selected_tag']
  combo=[] if winner=='99' else [(f'coh_{family}_{winner}',s) for s in range(731,738) for family in ('gate','learned')]
  for kind,seed in combo[INDEX::12]:trial(kind,seed)
  verify();dump(WORKER/'completion.json',dict(status='completed',fifo_runs=tasks[INDEX::12],combination_runs=combo[INDEX::12]))
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  try:
   with (ROOT/'fine-failure.json').open('x') as out:json.dump(dict(mode=MODE,index=INDEX,error=repr(exc)),out)
  except FileExistsError:pass
  raise
