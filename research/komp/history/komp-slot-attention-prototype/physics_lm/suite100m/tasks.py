"""Task-specific generation and prompt-only grading; source-pinned PhysicsLM4 adaptations."""
import ast
import hashlib
import json
import math
from pathlib import Path
import random
import re
import struct
from functools import lru_cache
import physics_port as port

ROOT=Path(__file__).parent
SOURCE=ROOT/'source'
BOS,EOS=9999,9998
TASKS=('mano','lano','depo2','brevo2','capo')


def digest(ids):return hashlib.sha256(struct.pack('<'+'H'*len(ids),*ids)).hexdigest()

def words(ids,mini):
    result=[];current=[]
    for t in ids:
        current.append(t)
        if t>mini:result.append(tuple(current));current=[]
    if current:raise ValueError('unterminated entity')
    return result


class Grammar:
    def __init__(self):
        self.config=json.loads((SOURCE/'Lano-cfg/configs/cfg3f.json').read_text())
        self.levels=self.config['all']
        self.bounds={}
        for level in reversed(range(len(self.levels))):
            for j,node in enumerate(self.levels[level]):
                if node['children'] is None:self.bounds[level,j]=(1,1)
                else:
                    bounds=[(sum(self.bounds[level+1,k][0] for k in rule),sum(self.bounds[level+1,k][1] for k in rule)) for rule in node['children']]
                    self.bounds[level,j]=(min(x[0] for x in bounds),max(x[1] for x in bounds))
    def generate(self,rng,level=0,index=0):
        node=self.levels[level][index]
        if node['children'] is None:return [node['id']]
        return [t for k in rng.choice(node['children']) for t in self.generate(rng,level+1,k)]
    def valid(self,seq):
        if any(x not in (1,2,3) for x in seq):return False
        n=len(seq)
        @lru_cache(None)
        def ends(level,index,start):
            lo,hi=self.bounds[level,index]
            if start+lo>n:return frozenset()
            node=self.levels[level][index]
            if node['children'] is None:return frozenset([start+1]) if seq[start]==node['id'] else frozenset()
            out=set()
            for rule in node['children']:
                positions={start}
                for child in rule:
                    positions={end for begin in positions for end in ends(level+1,child,begin)}
                    if not positions:break
                out.update(positions)
            return frozenset(out)
        return n in ends(0,0,0)


def lexical(text):return re.findall(r"[A-Za-z]+|\d+|[^\w\s]",text)


class Bios:
    def __init__(self):
        # Extract the six original template arrays as literals; no execution of upstream example code.
        tree=ast.parse((SOURCE/'Capo-bioS-bioR/Capo-bioS-bioR.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_text_simple3')
        self.templates={n.targets[0].id:ast.literal_eval(n.value) for n in fn.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id.startswith('sentence_structures')}
        self.pools={p.stem:[s.strip() for s in p.read_text().splitlines() if s.strip()] for p in (SOURCE/'Capo-bioS-bioR/fields').glob('*.txt')}
        rng=random.Random(87123);self.people=[];seen=set()
        for i in range(20000):
            while True:
                name=tuple(rng.choice(self.pools[k]) for k in ('first_name','middle_name','last_name'))
                if name not in seen:seen.add(name);break
            person=dict(name=' '.join(name),birthday=f"{rng.choice(['January','February','March','April','May','June','July','August','September','October','November','December'])} {rng.randint(1,28)}, {rng.randint(1900,2000)}",
                birthcity=rng.choice(self.pools['city']),university=rng.choice(self.pools['university']),field=rng.choice(self.pools['field']),company1city=rng.choice(self.pools['city']),company1name=rng.choice(self.pools['company']))
            self.people.append(person)
        allwords=set()
        for strings in self.templates.values():
            for text in strings:allwords.update(lexical(text))
        for p in self.people:
            for text in p.values():allwords.update(lexical(text))
        for pre in self.prompts():allwords.update(lexical(pre))
        assert len(allwords)<21998,len(allwords)
        self.vocab={w:i+10001 for i,w in enumerate(sorted(allwords))}
        self.inverse={i:w for w,i in self.vocab.items()}
    @staticmethod
    def prompts():return ['{name} was born on','{name} was born in','{name} studied at','{name} studied','{name} worked in','{name} worked at']
    def encode(self,text):return [self.vocab[w] for w in lexical(text)]
    def row(self,rng,index=None):
        person=self.people[rng.randrange(len(self.people)) if index is None else index]
        sentences=[rng.choice(self.templates[f'sentence_structures{i}']).format(**person) for i in range(1,7)]
        rng.shuffle(sentences)
        ids=[BOS]+self.encode(' '.join(sentences))+[EOS]
        return dict(input_ids=ids,loss_mask=[False]+[True]*(len(ids)-1),meta={})
    def probes(self,split,n):
        rng=random.Random(88001 if split=='val' else 88002)
        # Different person subsets across validation/test; every queried person appears in training.
        indices=list(range(10000)) if split=='val' else list(range(10000,20000))
        indices=rng.sample(indices,n)
        fields=['birthday','birthcity','university','field','company1city','company1name']
        rows=[]
        for j,index in enumerate(indices):
            field=j%6;p=self.people[index]
            rows.append(dict(prompt=[BOS]+self.encode(self.prompts()[field].format(**p)),answer=self.encode(p[fields[field]])+self.encode('.'),meta=dict(person=index,field=fields[field])))
        return rows


class Factory:
    def __init__(self,task):
        self.task=task
        if task=='lano':self.grammar=Grammar()
        if task=='capo':self.bios=Bios()
    def row(self,rng,index=None):
        if self.task=='mano':
            length=rng.randint(1,16)
            def tree(n):
                if not n:
                    v=rng.randrange(23);return [5000+v],v
                left=rng.randrange(n);a,av=tree(left);b,bv=tree(n-1-left);op=rng.randrange(3)
                return [op+1]+a+b,([av+bv,av-bv,av*bv][op]%23)
            expression,answer=tree(length)
            ids=[BOS,10*length]+expression+[10*length+4,5000+answer,EOS]
            row=dict(input_ids=ids,loss_mask=[False]*(len(ids)-2)+[True,True],meta=dict(operations=length))
        elif self.task=='lano':
            ids=[BOS]+self.grammar.generate(rng)+[EOS]
            row=dict(input_ids=ids,loss_mask=[False]+[True]*(len(ids)-1),meta={})
        elif self.task=='depo2':
            row=port.gen_depo_one(rng,port.DepoSpec('depo2',125,16,mini_vocab=3,min_tlen=5,max_tlen=7))
        elif self.task=='brevo2':row=port.gen_brevo_one(rng,port.BrevoSpec('brevo2',50,True))
        else:row=self.bios.row(rng,index)
        row['task']=self.task
        assert len(row['input_ids'])<=2048 and max(row['input_ids'])<32000
        return row
    def probe(self,row):
        ids=row['input_ids'];task=self.task
        if task=='mano':return dict(prompt=ids[:-2],answer=[ids[-2]],meta=row['meta'])
        if task=='lano':return dict(prompt=ids[:33],answer=None,meta={})
        if task=='depo2':
            marker=ids.index(9500);end=marker+1
            while ids[end]<=3:end+=1
            return dict(prompt=ids[:marker+1],answer=ids[marker+1:end+1],meta=row['meta'])
        if task=='brevo2':
            marker=ids.index(9997);end=marker+1
            while ids[end]<=4:end+=1
            # Original answer delimiter is also the EOS id.
            assert ids[end+1]==EOS
            return dict(prompt=ids[:end+2],answer=ids[end+2:],meta=row['meta'])
        raise ValueError(task)
    def grade(self,probe,guess):
        ids=probe['prompt'];task=self.task
        if task=='lano':return bool(guess and guess[-1]==EOS and self.grammar.valid(ids[1:]+guess[:-1]))
        if task=='mano':
            expression=iter(ids[2:-1])
            def evaluate():
                token=next(expression)
                if token>=5000:return token-5000
                a,b=evaluate(),evaluate()
                return {1:a+b,2:a-b,3:a*b}[token]%23
            answer=5000+evaluate()
            assert list(expression)==[] and probe['answer']==[answer]
            return guess==[answer]
        if task=='capo':return guess==probe['answer']
        if task=='depo2':
            boundary=next(i for i,t in enumerate(ids) if 9000<t<9500)
            ws=words(ids[1:boundary],3);successor=dict(zip(ws[::2],ws[1::2]))
            query=tuple(ids[boundary+1:-1])
            for _ in range(ids[boundary]-9000):query=successor[query]
            assert list(query)==probe['answer']
            return guess==list(query)
        marker=ids.index(9997);ws=words(ids[1:marker],4);edges=list(zip(ws[::2],ws[1::2]));query=tuple(ids[marker+1:-1]);needed={query}
        while True:
            new=needed|{a for a,b in edges if b in needed}
            if new==needed:break
            needed=new
        if not guess or guess[-1]!=EOS:return False
        try:ans=words(guess[:-1],4)
        except ValueError:return False
        if len(ans)!=len(needed) or set(ans)!=needed:return False
        positions={x:i for i,x in enumerate(ans)}
        return all(positions[a]<positions[b] for a,b in edges if b in needed)
