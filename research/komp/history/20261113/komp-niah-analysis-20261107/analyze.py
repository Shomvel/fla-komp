from pathlib import Path
import json,gzip,re,collections,statistics
from tokenizers import Tokenizer
root=Path(__file__).resolve().parent
d=json.loads(gzip.decompress((root/'raw.json.gz').read_bytes()));tok=Tokenizer.from_file(str(root/'tokenizer.json'))
req={};bad=[]
for x in d['requests']:
 prompt=x['prompt'];label=x['labels'][0];enc=tok.encode(prompt,add_special_tokens=False)
 assert enc.ids==x['input_ids'],x['id']
 assert len(x['labels'])==1 and prompt.count(label)==1,(x['id'],prompt.count(label))
 start=prompt.index(label);end=start+len(label)
 inds=[i for i,(a,b) in enumerate(enc.offsets) if b>start and a<end]
 assert inds
 needle_start=prompt.rfind('One of the special magic',0,start)
 needle_end=prompt.find('.',end)+1
 q=prompt.rfind('What is the special magic')
 req[x['id']]=dict(id=x['id'],case=x['case'],label=label,n=len(enc.ids),answer_tokens=len(inds),answer_start=inds[0],answer_end=inds[-1]+1,distance=len(enc.ids)-inds[-1]-1,depth=inds[0]/len(enc.ids),needle=prompt[needle_start:needle_end],suffix=prompt[q:],generation=x['generation'])
models=sorted(set(x['model'] for x in d['results']));stats={};examples=[]
def compact(x):return {k:v for k,v in x.items() if k not in ['generated_ids']}
for m in models:
 for case in sorted(set(x['case'] for x in d['requests'])):
  rows=[x for x in d['results'] if x['model']==m and x['case']==case]
  if len(rows)!=500:continue
  assert len({x['id'] for x in rows})==500
  merged=[]
  for x in rows:
   q=req[x['id']];assert x['prompt_tokens']==q['n'];assert x['truncated_prompt_tokens']==0
   score=int(q['label'].lower() in re.sub(r'[\x00-\x1f]','\n',x['prediction'].strip()).strip().lower())
   assert score==x['score']
   s=x['prediction'];norm=re.sub(r'[^a-z0-9]','',s.lower());lab=re.sub(r'[^a-z0-9]','',q['label'].lower())
   prefix=0
   for a,b in zip(s.lstrip(' \n\t:="\''),q['label']):
    if a==b:prefix+=1
    else:break
   merged.append(dict(**q,prediction=s,score=score,gen_tokens=x['generated_tokens'],format_rescue=int(lab in norm and not score),prefix_chars=prefix,ends_eos=x['generated_ids'][-1]==2,empty=not s.strip(),first_number=(re.findall(r'\d+',s)or[''])[0]))
  def agg(v):return dict(n=len(v),correct=sum(x['score'] for x in v),accuracy=sum(x['score'] for x in v)/len(v) if v else None)
  bins={}
  for lo,hi in [(0,128),(128,256),(256,512),(512,1024),(1024,2048),(2048,4096),(4096,8192)]:
   bins[f'{lo}-{hi}']=agg([x for x in merged if lo<=x['distance']<hi])
  quarters={f'{a}-{a+25}%':agg([x for x in merged if a/100<=x['depth']<(a+25)/100]) for a in [0,25,50,75]}
  wrong=[x for x in merged if not x['score']]
  stats[m,case]=dict(**agg(merged),prompt_range=[min(x['n'] for x in merged),max(x['n'] for x in merged)],answer_tokens_range=[min(x['answer_tokens'] for x in merged),max(x['answer_tokens'] for x in merged)],distance_range=[min(x['distance'] for x in merged),max(x['distance'] for x in merged)],distance_bins=bins,depth_quartiles=quarters,empty=sum(x['empty'] for x in merged),format_rescues=sum(x['format_rescue'] for x in merged),cap128=sum(x['gen_tokens']==128 for x in merged),eos=sum(x['ends_eos'] for x in merged),wrong_has_number=sum(bool(re.search(r'\d',x['prediction'])) for x in wrong),wrong_prefix3=sum(x['prefix_chars']>=3 for x in wrong),wrong_first_numbers=collections.Counter(x['first_number'] for x in wrong).most_common(8),wrong_gen_median=statistics.median(x['gen_tokens'] for x in wrong) if wrong else None)
  if case.endswith('_2048'):
   for condition,rr in [('near_wrong',sorted(wrong,key=lambda x:x['distance'])[:3]),('far_wrong',sorted(wrong,key=lambda x:-x['distance'])[:2]),('partial_copy',sorted([x for x in wrong if x['prefix_chars']>=3],key=lambda x:-x['prefix_chars'])[:3]),('correct',[x for x in merged if x['score']][:2])]:
    for x in rr:examples.append(dict(model=m,condition=condition,**x))
result=dict(captured_utc=d['captured_utc'],stats=[dict(model=m,case=c,**v) for (m,c),v in stats.items()],examples=examples,requests=list(req.values()),invariants='All 6000 requests retokenize exactly; each answer present exactly once; all analyzed complete cases have 500 unique IDs, zero prompt truncation, and independently reproduced scores.')
(root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for (m,c),s in stats.items():
 if c.endswith(('_1024','_2048')):
  print(m,c,s['correct'],'range',s['prompt_range'],'answer_tokens',s['answer_tokens_range'],'depth',[(k,v['correct'],v['n']) for k,v in s['depth_quartiles'].items()],'rescue',s['format_rescues'],'empty',s['empty'],'cap128',s['cap128'],'prefix3wrong',s['wrong_prefix3'])
