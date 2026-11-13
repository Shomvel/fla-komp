"""Materialize the existing BP flame task fork's exact zero-shot requests once."""
from pathlib import Path
import os,sys,json,hashlib,random,copy,re
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
os.environ['HF_HUB_CACHE']='/gpfs/users/linlizhi/.cache/huggingface/hub'
os.environ['HF_DATASETS_CACHE']=os.environ['HF_HUB_CACHE']
os.environ['HF_EVALUATE_CACHE']='/gpfs/users/linlizhi/.cache/huggingface/evaluate'
os.environ['HF_DATASETS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1'
import datasets
original_load_dataset=datasets.load_dataset
def local_cache_load(path,*args,**kwargs):
 if str(path).startswith('/gpfs/'):
  kwargs['cache_dir']=str(ROOT/'dataset-cache')
 return original_load_dataset(path,*args,**kwargs)
datasets.load_dataset=local_cache_load
from lm_eval.tasks import TaskManager,get_task_dict
from transformers import AutoTokenizer
manifest=json.loads(Path('/gpfs/users/linlizhi/komp-gdn-slot-learned-123-340m-20c-20261107/configs/data-manifest.json').read_text())
tokpath=manifest['tokenizer_path'];tok=AutoTokenizer.from_pretrained(tokpath)
from tokenizers import Tokenizer
raw=Tokenizer.from_file(str(Path(tokpath)/'tokenizer.json'))
for text in [' hello world','Here is a phonebook.\n A: 123','What is the magic number?']:
 assert tok.encode(text,add_special_tokens=False)==raw.encode(text,add_special_tokens=False).ids
names=['niah_single_1','niah_single_2','niah_single_3']+[f'phonebook_{n}' for n in [32,64,100,128,150,200,256]]+['based_swde','based_fda','based_nq_1024','based_drop']
out=json.loads((ROOT/'evidence/requests-manifest.json').read_text())['cases'] if (ROOT/'evidence/requests-manifest.json').exists() else []
for name in names:
 if any(x['task']==name for x in out):continue
 random.seed(0);np.random.seed(1234)
 manager=TaskManager(metadata={'tokenizer':tokpath,'max_seq_lengths':[1024,2048,4096,8192]})
 task=get_task_dict([name],manager)[name]
 docs=task.test_docs() if task.has_test_docs() else task.validation_docs()
 files={};counts={};lengths={}
 for i,d in enumerate(docs):
  doc=copy.deepcopy(d)
  prefix=task.doc_to_prefix(doc)
  prompt=task.fewshot_context(doc,num_fewshot=0,gen_prefix=prefix)
  reqs=task.construct_requests(doc,prompt,apply_chat_template=False,chat_template=None)
  if not isinstance(reqs,(list,tuple)):reqs=[reqs]
  assert len(reqs)==1 and reqs[0].request_type=='generate_until',name
  ctx,kwargs=reqs[0].args
  labels=doc.get('outputs') if name.startswith('niah') else [doc['label']] if name.startswith('phonebook') else doc.get('answers',[doc.get('value')])
  assert labels and all(isinstance(x,str) for x in labels),(name,i)
  case=name+(f"_{doc['max_length']}" if name.startswith('niah') else '')
  if case not in files:
   files[case]=(ROOT/'requests'/f'{case}.jsonl').open('w');counts[case]=0;lengths[case]=[]
  ids=raw.encode(ctx,add_special_tokens=False).ids
  row=dict(id=f'{case}:{i}',task=name,case=case,index=i,prompt=ctx,input_ids=ids,labels=labels,generation=kwargs,metric='string_match_all' if name.startswith('niah') else 'contains',doc=doc)
  # Verify local scoring semantics against the original task on hit and miss.
  for prediction in ['MATCH_SENTINEL_29',labels[0],labels[0].upper()]:
   score=task.process_results(copy.deepcopy(doc),[prediction])
   expected=(sum(x.lower() in re.sub(r'[\x00-\x1f]','\n',prediction.strip()).strip().lower() for x in labels)/len(labels) if name.startswith('niah') else max(int(bool(re.search(re.escape(x.strip() if name.startswith('phonebook') else x),prediction.strip() if name.startswith('phonebook') else prediction,re.I))) for x in labels))
   assert score.get(str(doc.get('max_length')) if name.startswith('niah') else 'contains')==expected,(name,score,expected)
  files[case].write(json.dumps(row,ensure_ascii=False)+'\n');counts[case]+=1;lengths[case].append(len(ids))
 for case,f in files.items():
  f.close();p=ROOT/'requests'/f'{case}.jsonl'
  out.append(dict(case=case,task=name,count=counts[case],sha256=hashlib.sha256(p.read_bytes()).hexdigest(),min_prompt_tokens=min(lengths[case]),max_prompt_tokens=max(lengths[case])))
 (ROOT/'evidence/requests-manifest.json').write_text(json.dumps(dict(tokenizer=tokpath,tokenizer_sha=hashlib.sha256((Path(tokpath)/'tokenizer.json').read_bytes()).hexdigest(),add_special_tokens=False,eos_token_id=tok.eos_token_id,bos_token_id=tok.bos_token_id,seeds=dict(python=0,numpy=1234),cases=out),indent=2)+'\n')
 print(name,counts,flush=True)
