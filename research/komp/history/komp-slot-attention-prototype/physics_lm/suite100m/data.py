"""Deterministic ragged corpora, with a shared 10M-token prefix for LR/WD search.

Budgets count non-padding next-token positions (document length minus BOS),
including prompt positions, not only positions carrying a supervised loss.
Training documents may repeat; the budget is not a count of unique tokens.
"""
import argparse
import os
from array import array
import hashlib
import json
from pathlib import Path
import random
import struct
import sys
from tasks import Factory,TASKS,digest

class Writer:
    def __init__(self,root,name):
        self.root=root;self.name=name
        self.tokens=(root/f'{name}.tokens.u16').open('wb')
        self.masks=(root/f'{name}.mask.u8').open('wb')
        self.offsets=array('Q',[0]);self.positions=0;self.supervised=0
        self.fingerprint=hashlib.sha256()
    def add(self,row):
        ids=row['input_ids'];mask=row['loss_mask']
        assert len(ids)==len(mask) and not mask[0]
        packed=struct.pack('<'+'H'*len(ids),*ids)
        self.tokens.write(packed);self.masks.write(bytes(mask));self.fingerprint.update(packed)
        self.offsets.append(self.offsets[-1]+len(ids));self.positions+=len(ids)-1;self.supervised+=sum(mask[1:])
    def close(self):
        self.tokens.close();self.masks.close()
        if sys.byteorder!='little':self.offsets.byteswap()
        with (self.root/f'{self.name}.offsets.u64').open('wb') as f:self.offsets.tofile(f)
        return dict(documents=len(self.offsets)-1,tokens_with_bos=int(self.offsets[-1]),next_token_positions=self.positions,supervised_positions=self.supervised,token_sha256=self.fingerprint.hexdigest())

def build(task,dest,budget=100_000_000,tune_budget=10_000_000,eval_size=1024):
    root=dest/task;root.mkdir(parents=True,exist_ok=False)
    fac=Factory(task);heldout=set();manifest=dict(task=task,budget=budget,tune_budget=tune_budget,seed=731001,definition='non-padding next-token positions; full final document may overshoot',splits={})
    for split,seed in [('val',731002),('test',731003)]:
        rng=random.Random(seed);writer=Writer(root,split);rows=[]
        while len(rows)<eval_size:
            row=fac.row(rng);fp=digest(row['input_ids'])
            if fp in heldout:continue
            heldout.add(fp);rows.append(row);writer.add(row)
        manifest['splits'][split]=writer.close()
        probes=fac.bios.probes(split,eval_size) if task=='capo' else [fac.probe(row) for row in rows]
        (root/f'{split}.probes.json').write_text(json.dumps(probes,separators=(',',':')))
    writer=Writer(root,'train');rng=random.Random(731001);covered=0;tune=None
    while writer.positions<budget:
        index=covered if task=='capo' and covered<len(fac.bios.people) else None
        row=fac.row(rng,index);fp=digest(row['input_ids'])
        if fp in heldout:continue
        writer.add(row);covered+=1
        if tune is None and writer.positions>=tune_budget:
            tune=dict(documents=covered,next_token_positions=writer.positions,supervised_positions=writer.supervised,token_sha256=writer.fingerprint.hexdigest())
            writer.tokens.flush();writer.masks.flush()
            with (root/'tune.offsets.u64').open('wb') as stream:writer.offsets.tofile(stream)
            (root/'tune-ready.json.tmp').write_text(json.dumps(dict(task=task,tune_prefix=tune,splits=manifest['splits'])))
            os.replace(root/'tune-ready.json.tmp',root/'tune-ready.json')
        if covered%100000==0:print(task,covered,writer.positions,flush=True)
    manifest['splits']['train']=writer.close();manifest['tune_prefix']=tune
    if task=='capo':
        assert covered>=len(fac.bios.people),'Budget must cover all fixed people'
        (root/'capo-vocab.json').write_text(json.dumps(fac.bios.vocab,sort_keys=True))
        (root/'capo-people.json').write_text(json.dumps(fac.bios.people))
        manifest['capo']=dict(people=len(fac.bios.people),tokenization='word and punctuation, adapted from official bioS templates',all_people_in_train=True)
    src=Path(__file__).parent
    manifest['source_sha256']={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),src/'tasks.py',src/'physics_port.py',*sorted((src/'source').rglob('*'))] if p.is_file()}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(task,'complete',manifest['splits']['train'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=TASKS);p.add_argument('--dest',type=Path,required=True);p.add_argument('--budget',type=int,default=100_000_000);p.add_argument('--tune-budget',type=int,default=10_000_000);p.add_argument('--eval-size',type=int,default=1024);args=p.parse_args()
    build(args.task,args.dest,args.budget,args.tune_budget,args.eval_size)
