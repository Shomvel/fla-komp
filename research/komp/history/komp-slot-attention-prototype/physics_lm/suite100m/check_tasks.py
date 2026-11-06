import json,random,time
from tasks import Factory,TASKS
out={}
for task in TASKS:
 start=time.monotonic();f=Factory(task);rng=random.Random(71);lengths=[]
 for i in range(50):
  row=f.row(rng);lengths.append(len(row['input_ids']))
  if task=='capo':continue
  p=f.probe(row)
  answer=row['input_ids'][33:] if task=='lano' else p['answer']
  assert f.grade(p,answer),(task,p)
  wrong=[12345] if task!='lano' else [12345,9998]
  assert not f.grade(p,wrong)
 if task=='capo':
  for p in f.bios.probes('test',50):assert f.grade(p,p['answer']) and not f.grade(p,[1])
 out[task]=dict(n=50,min_length=min(lengths),max_length=max(lengths),mean_length=sum(lengths)/len(lengths),seconds=time.monotonic()-start)
 if task=='capo':out[task]['vocab']=len(f.bios.vocab)
 print(task,out[task],flush=True)
print(json.dumps(out,indent=2))
