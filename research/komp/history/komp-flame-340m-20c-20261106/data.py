"""Deterministic document stream with byte-exact resume and heldout exclusion."""
import hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer

class Stream:
 def __init__(self,manifest,rank,world=8):
  self.manifest=manifest
  files=list(manifest['train_files']);random.Random(42).shuffle(files)
  self.files=files[rank::world];assert self.files
  self.tokenizer=Tokenizer.from_file(str(Path(manifest['tokenizer_path'])/'tokenizer.json'))
  excluded=json.loads(Path(manifest['excluded_text_sha256_path']).read_text())
  self.excluded=set(excluded if isinstance(excluded,list) else excluded['sha256'])
  self.state=dict(file_index=0,offset=0,tokens=[],documents=0,excluded_documents=0,epoch=0,sequences=0)
  self.handle=None
 def close(self):
  if self.handle is not None:self.handle.close();self.handle=None
 def state_dict(self):return dict(self.state,tokens=list(self.state['tokens']))
 def load_state_dict(self,state):self.close();self.state=dict(state,tokens=list(state['tokens']))
 def document(self):
  while True:
   if self.handle is None:
    self.handle=open(self.files[self.state['file_index']],'rb');self.handle.seek(self.state['offset'])
   line=self.handle.readline();self.state['offset']=self.handle.tell()
   if not line:
    self.close();self.state['file_index']+=1;self.state['offset']=0
    if self.state['file_index']==len(self.files):self.state['file_index']=0;self.state['epoch']+=1
    continue
   text=json.loads(line)['text'];self.state['documents']+=1
   fingerprint=hashlib.sha256(text.encode('utf-8')).hexdigest()
   if fingerprint in self.excluded:self.state['excluded_documents']+=1;continue
   return text
 def next(self,seq_len=2048):
  while len(self.state['tokens'])<seq_len+1:
   self.state['tokens'].extend(self.tokenizer.encode(self.document(),add_special_tokens=True).ids)
  ids=torch.tensor(self.state['tokens'][:seq_len+1],dtype=torch.long)
  # Shared boundary: every next-token target is counted once, including sequence edges.
  self.state['tokens']=self.state['tokens'][seq_len:];self.state['sequences']+=1
  return ids

def evaluate(model,path,rank,world=8):
 ids=np.memmap(path,dtype=np.uint16,mode='r');blocks=(len(ids)-1)//2048
 assert blocks and (len(ids)-1)%2048==0
 result=torch.zeros(2,device='cuda',dtype=torch.float64);model.eval()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  for i in range(rank,blocks,world):
   row=torch.tensor(np.array(ids[i*2048:i*2048+2049],dtype=np.int64),device='cuda')[None]
   result[0]+=model(row[:,:-1],row[:,1:]);result[1]+=2048
 torch.distributed.all_reduce(result);model.train()
 return dict(cross_entropy=float(result[0]/result[1]),supervised_tokens=int(result[1]),perplexity=float((result[0]/result[1]).exp()))
