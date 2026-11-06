"""Matched small-LM Depo/Brevo experiment using the frozen 10M-token corpus."""
import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import benchmark_fla as ops
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent
DATA=Path(os.environ.get('PHYSICS_DATA','/gpfs/users/linlizhi/softmax-10m-20261023/data'))
D,H,K,M,L,FF,VOCAB=128,2,64,32,4,256,32000


class Mixer(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        self.q=nn.Linear(D,D,bias=False);self.k=nn.Linear(D,D,bias=False);self.v=nn.Linear(D,D,bias=False)
        self.out=nn.Linear(D,D,bias=False);self.gate=nn.Linear(D,D,bias=False)
        self.norm=nn.RMSNorm(K)
        self.decay=nn.Linear(D,H*M if kind=='gsa' else H)
        self.beta=nn.Linear(D,H) if kind!='gsa' else None
        if kind=='komp':
            key=torch.randn(H,M,K)
            key=torch.stack([torch.linalg.qr(x.T).Q.T for x in key])
            self.keys=nn.Parameter(key)
        nn.init.zeros_(self.decay.weight);nn.init.constant_(self.decay.bias,math.log(.98/.02))
        if self.beta is not None:
            nn.init.zeros_(self.beta.weight);nn.init.constant_(self.beta.bias,math.log(.2/.8))

    def forward(self,x):
        b,t,_=x.shape
        q,k,v=[f(x).reshape(b,t,H,K) for f in (self.q,self.k,self.v)]
        if self.kind=='komp':
            alpha=torch.sigmoid(self.decay(x).float());beta=torch.sigmoid(self.beta(x).float())
            slots=torch.arange(t,device=x.device)[None,:,None].expand(b,t,H).remainder(M).contiguous()
            keys=self.keys[None].expand(b,H,M,K).contiguous()
            initial=torch.zeros(b,H,M,K,device=x.device)
            h,p,r,_=ops.triton_geometry(q,k,slots,keys)
            o,_=ops.triton_chunk(h,p,r,v,alpha,beta,slots,initial)
        elif self.kind=='gdn':
            g=F.logsigmoid(self.decay(x).float());beta=torch.sigmoid(self.beta(x).float())
            o,_=ops.chunk_gated_delta_rule(q,k,v,g,beta,use_qk_l2norm_in_kernel=True)
        else:
            q,k=[F.normalize(z.float(),dim=-1).to(z.dtype) for z in (q,k)]
            g=F.logsigmoid(self.decay(x).float()).reshape(b,t,H,M)
            s=(1-g.exp()).to(v.dtype)
            o,_=ops.chunk_gsa(q,k,v,s,g)
        y=self.norm(o.float()).to(x.dtype).reshape(b,t,D)
        return self.out(y*F.silu(self.gate(x)))


class Block(nn.Module):
    def __init__(self,kind):
        super().__init__();self.n1=nn.RMSNorm(D);self.n2=nn.RMSNorm(D)
        self.mix=Mixer(kind);self.up=nn.Linear(D,2*FF,bias=False);self.down=nn.Linear(FF,D,bias=False)
    def forward(self,x):
        x=x+self.mix(self.n1(x))
        a,b=self.up(self.n2(x)).chunk(2,-1)
        return x+self.down(F.silu(a)*b)


class LM(nn.Module):
    def __init__(self,kind,seed):
        super().__init__()
        # Shared modules use named seeds, so extra architecture parameters do not shift their initialization.
        torch.manual_seed(seed)
        self.embed=nn.Embedding(VOCAB,D)
        self.layers=nn.ModuleList()
        for i in range(L):
            torch.manual_seed(seed+100+i)
            self.layers.append(Block(kind))
        torch.manual_seed(seed+900)
        self.norm=nn.RMSNorm(D);self.head=nn.Linear(D,VOCAB,bias=False)
        for name,module in self.named_modules():
            if isinstance(module,(nn.Linear,nn.Embedding)) and not name.endswith(('decay','beta')):
                salt=int(hashlib.sha256(name.encode()).hexdigest()[:8],16)
                torch.manual_seed(seed+salt)
                nn.init.normal_(module.weight,0,.02)
                if isinstance(module,nn.Linear) and module.bias is not None:nn.init.zeros_(module.bias)
        if kind=='gsa':
            # Identical slot gates with zero states preserve permutation symmetry.
            # Each slot needs an independent projection, as in FLA's f_proj.
            for i,layer in enumerate(self.layers):
                torch.manual_seed(seed+2000+i)
                nn.init.normal_(layer.mix.decay.weight,0,.02)
    def forward(self,ids):
        x=self.embed(ids)
        for layer in self.layers:x=layer(x)
        return self.norm(x)


def load(split):
    path=DATA/(split+'.jsonl')
    manifest=json.loads((ROOT/'data-manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['splits'][split]['sha256']
    return [json.loads(line) for line in path.open()]


def batch(rows):
    # Right padding to a fixed 192-token length; complete documents, no packing/truncation.
    ids=torch.zeros(len(rows),192,dtype=torch.long)
    mask=torch.zeros_like(ids,dtype=torch.bool)
    for i,row in enumerate(rows):
        n=len(row['input_ids']);assert n<=192
        ids[i,:n]=torch.tensor(row['input_ids']);mask[i,:n]=torch.tensor(row['loss_mask'])
    return ids.cuda(),mask.cuda()


def loss_items(model,rows):
    ids,mask=batch(rows);keep=mask[:,1:]
    h=model(ids)[:,:-1]
    logits=model.head(h[keep]).float();gold=ids[:,1:][keep]
    return F.cross_entropy(logits,gold,reduction='none'),logits.argmax(-1),gold


def amp():return torch.autocast('cuda',dtype=torch.bfloat16)


@torch.no_grad()
def check_slot_diversity(model,kind):
    if kind!='gsa':return
    x=model.embed(torch.arange(16,device='cuda')[None])
    for layer in model.layers:
        gates=layer.mix.decay(layer.n1(x)).reshape(1,16,H,M)
        assert bool((gates.var(dim=-1)>1e-6).all()),'GSA slots have collapsed at initialization'


@torch.no_grad()
def evaluate(model,rows):
    model.eval();result={}
    with amp():
        for task in ('depo','brevo'):
            selected=[r for r in rows if r['task']==task];total=torch.zeros(6,device='cuda')
            for i in range(0,len(selected),32):
                loss,pred,gold=loss_items(model,selected[i:i+32]);content=gold<9000
                total+=torch.stack([loss.sum(),loss.new_tensor(loss.numel()),loss[content].sum(),
                                    content.sum(),((pred==gold)&content).sum(),(pred==gold).sum()])
            a,b,c,d,e,f=total.tolist()
            result[task]={'answer_ce':a/b,'content_ce':c/d,'content_accuracy':e/d,'answer_accuracy':f/b,'content_tokens':int(d)}
    result['mean_content_ce']=sum(result[t]['content_ce'] for t in ('depo','brevo'))/2
    model.train();return result


@torch.no_grad()
def generate(model,rows,limit=128):
    model.eval();records=[]
    with amp():
        for task in ('depo','brevo'):
            selected=[r for r in rows if r['task']==task][:limit]
            for start in range(0,len(selected),32):
                block=selected[start:start+32];prefixes=[];golds=[]
                for row in block:
                    ids=row['input_ids']
                    if task=='depo':
                        boundary=ids.index(9500);end=boundary+1
                        while ids[end]<=50:end+=1
                        gold=ids[boundary+1:end+1]
                    else:
                        boundary=ids.index(9997)+2;gold=ids[boundary+1:]
                    prefixes.append(ids[:boundary+1]);golds.append(gold)
                seq=torch.zeros(len(block),192,dtype=torch.long,device='cuda')
                lengths=torch.tensor([len(p) for p in prefixes],device='cuda')
                for i,p in enumerate(prefixes):seq[i,:len(p)]=torch.tensor(p,device='cuda')
                guesses=[[] for _ in block];active=[True]*len(block)
                for _ in range(3 if task=='depo' else 25):
                    h=model(seq);tok=model.head(h[torch.arange(len(block),device='cuda'),lengths-1]).argmax(-1)
                    seq[torch.arange(len(block),device='cuda'),lengths]=tok;lengths+=1
                    for i,item in enumerate(tok.tolist()):
                        if active[i]:
                            guesses[i].append(item)
                            if (task=='depo' and item>50) or (task=='brevo' and item==9998):active[i]=False
                    if not any(active):break
                for row,gold,guess in zip(block,golds,guesses):
                    valid=guess==gold;meta=dict(row['meta'])
                    if task=='brevo':
                        ids=row['input_ids'];end=ids.index(9997);edges=list(zip(ids[1:end:2],ids[2:end:2]))
                        ans=guess[:-1] if guess and guess[-1]==9998 else guess
                        nodes=set(gold[:-1]);pos={n:i for i,n in enumerate(ans)}
                        valid=bool(guess and guess[-1]==9998 and set(ans)==nodes and len(ans)==len(nodes)
                                   and all(pos[a]<pos[b] for a,b in edges if b in nodes))
                    else:meta['query_k']=next(t-9000 for t in row['input_ids'] if 9000<t<9500)
                    records.append({'task':task,'exact':guess==gold,'valid':valid,'gold':gold,'guess':guess,'meta':meta})
    model.train()
    return {t:{'n':sum(r['task']==t for r in records),
               'exact':sum(r['exact'] for r in records if r['task']==t)/limit,
               'valid':sum(r['valid'] for r in records if r['task']==t)/limit} for t in ('depo','brevo')},records


def schedule(step,total):
    warm=max(8,int(total*.05))
    factor=step/warm if step<=warm else .1+.9*(1+math.cos(math.pi*(step-warm)/(total-warm)))/2
    t=(step-1)/max(total-1,1)
    blend=.5-.5*math.cos(math.pi*min(1,max(0,(t-.25)/.5)))
    return 5e-4*factor,math.exp((1-blend)*math.log(.1)+blend*math.log(.3))


def preflight():
    from validate_data import check
    rows=load('train');val=load('val');test=load('test');confirm=load('confirm')
    seen=set()
    for split in (rows,val,test,confirm):
        for row in split:
            check(row);key=tuple(row['input_ids']);assert key not in seen;seen.add(key)
    assert sum(len(r['input_ids']) for r in rows)==10001108
    report={'data_checked':len(seen),'models':{}}
    sample=rows[:4]
    for kind in ('gdn','gsa','komp'):
        model=LM(kind,731).cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
        check_slot_diversity(model,kind)
        ids,_=batch(sample)
        with torch.no_grad(),amp():
            before=model(ids);changed=ids.clone();changed[:,40:]=(changed[:,40:]+7)%VOCAB
            after=model(changed)
            torch.testing.assert_close(before[:,:40],after[:,:40],atol=1e-5,rtol=1e-5)
        losses=[]
        for step in range(16):
            opt.zero_grad(set_to_none=True)
            with amp():loss=loss_items(model,sample)[0].mean()
            loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert bool(torch.isfinite(norm));opt.step();losses.append(float(loss.detach()))
        assert losses[-1]<losses[0],(kind,losses)
        report['models'][kind]={'params':sum(p.numel() for p in model.parameters()),'losses':losses,'causality':'passed'}
        print('PREFLIGHT',kind,report['models'][kind],flush=True)
        del model,opt;torch.cuda.empty_cache()
    (ROOT/'preflight.json').write_text(json.dumps(report,indent=2))


def train(kind,seed):
    assert (ROOT/'preflight.json').exists()
    dest=ROOT/'results'/f'{kind}-seed{seed}';dest.mkdir(parents=True,exist_ok=False)
    model=LM(kind,seed).cuda();rows=load('train');val=load('val')
    check_slot_diversity(model,kind)
    order=list(range(len(rows)));random.Random(12345).shuffle(order)
    groups={False:[],True:[]}
    for name,p in model.named_parameters():groups[name.endswith(('mix.out.weight','down.weight'))].append(p)
    opt=torch.optim.AdamW([{'params':groups[False]},{'params':groups[True]}],lr=5e-4,betas=(.9,.95),weight_decay=.1,fused=True)
    config={'architecture':kind,'seed':seed,'width':D,'layers':L,'heads':H,'key_value_dim':K,'slots':M,'ffn':FF,
        'vocab':VOCAB,'params':sum(p.numel() for p in model.parameters()),'batch':64,'epoch':1,'steps':math.ceil(len(rows)/64),
        'data':json.loads((ROOT/'data-manifest.json').read_text()),'lr':'5e-4 warmup5% cosine to10%; residual output multiplier .1->.3',
        'schedule_note':'Same prespecified recipe for all arms; not claimed optimal for this smaller backbone',
        'routing':'cyclic t mod M; fixed, no KOMP deletion-cost search','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    config['gsa_slot_initialization']='independent normal(0,.02) gate projection per slot; shared decay bias' if kind=='gsa' else None
    import triton,google.protobuf,platform
    from google.protobuf.internal.api_implementation import Type
    config['environment']={'torch':str(torch.__version__),'triton':str(triton.__version__),'cuda':torch.version.cuda,
        'python':platform.python_version(),'gpu':torch.cuda.get_device_name(),'protobuf':google.protobuf.__version__,'protobuf_backend':Type()}
    config['common_init_sha256']={n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
        for n,p in model.named_parameters() if n in ('embed.weight','head.weight','layers.0.mix.q.weight','layers.0.up.weight')}
    (dest/'manifest.json').write_text(json.dumps(config,indent=2))
    # Exercise the exact safe-loading contract before spending the training budget.
    probe=io.BytesIO()
    torch.save({'model':model.state_dict(),'config':config},probe)
    probe.seek(0)
    restored=torch.load(probe,map_location='cpu',weights_only=True)
    assert restored['config']==config
    model.load_state_dict(restored['model'],strict=True)
    del restored,probe
    log=(dest/'metrics.jsonl').open('w',buffering=1);started=time.monotonic();total=config['steps'];unique=0
    log.write(json.dumps({'step':0,'val':evaluate(model,val[:128])})+'\n')
    torch.cuda.reset_peak_memory_stats()
    for step,offset in enumerate(range(0,len(order),64),1):
        batchrows=[rows[j] for j in order[offset:offset+64]];unique+=sum(len(r['input_ids']) for r in batchrows)
        lr,mult=schedule(step,total);opt.param_groups[0]['lr']=lr;opt.param_groups[1]['lr']=lr*mult
        opt.zero_grad(set_to_none=True)
        with amp():loss=loss_items(model,batchrows)[0].mean()
        loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not bool(torch.isfinite(norm)) or not bool(torch.isfinite(loss)):raise RuntimeError(f'Nonfinite {kind} step {step}')
        opt.step()
        if step%50==0 or step==total:
            record={'step':step,'loss':float(loss.detach()),'grad_norm':float(norm),'lr':lr,'output_mult':mult,
                    'unique_tokens':unique,'seconds':time.monotonic()-started}
            if step%500==0 or step==total:record['val']=evaluate(model,val)
            log.write(json.dumps(record)+'\n');print(kind,seed,json.dumps(record),flush=True)
    assert unique==10001108
    valmetrics=evaluate(model,val);testmetrics=evaluate(model,load('test'));confirm=load('confirm')
    confirmmetrics=evaluate(model,confirm);gen,records=generate(model,confirm)
    summary={'config':config,'val':valmetrics,'test':testmetrics,'confirm':confirmmetrics,'generation':gen,
             'elapsed_seconds':time.monotonic()-started,'unique_tokens':unique,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
             'status':'completed'}
    torch.save({'model':model.state_dict(),'config':config},dest/'model.pt')
    saved=torch.load(dest/'model.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(saved['model'],strict=True)
    reloaded=evaluate(model,val)
    error=abs(reloaded['mean_content_ce']-valmetrics['mean_content_ce'])
    assert error<1e-4,error
    summary['reload_validation_ce_error']=error
    (dest/'generations.json').write_text(json.dumps(records,indent=2))
    (dest/'summary.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',kind,seed,json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--kind',choices=['gsa','gdn','komp']);p.add_argument('--seed',type=int,default=731)
    args=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if args.preflight:preflight()
    else:train(args.kind,args.seed)
