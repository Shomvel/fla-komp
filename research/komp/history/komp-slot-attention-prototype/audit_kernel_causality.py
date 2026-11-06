"""Causal interventions on frozen training kernels and trained checkpoints."""
import hashlib,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F
import triton

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'physics_lm'))
import decay_experiment as exp
import experiment as base
from komp_slot.triton_chunk import triton_chunk,prepare
from komp_slot.geometry import triton_geometry
from komp_slot.reference import recurrent_reference

CHECKPOINTS=Path('/gpfs/users/linlizhi/komp-decay-20261031-train-r1/physics_lm/decay_runs/results')
REPORT={'checks':[],'models':[]}

def exact(tag,a,b):
    err=float((a-b).abs().max())
    REPORT['checks'].append({'name':tag,'max_abs':err})
    assert torch.equal(a,b),(tag,err)

def kernel_audit():
    torch.manual_seed(9041)
    b,t,h,m,d=1,193,2,32,64
    rand=lambda *s:torch.randn(*s,device='cuda')
    q,k,v=rand(b,t,h,d),rand(b,t,h,d),rand(b,t,h,d)
    keys=rand(b,h,m,d);initial=rand(b,h,m,d)*.1
    slots=torch.randint(m,(b,t,h),device='cuda')
    hist,p,r,_=triton_geometry(q,k,slots,keys)
    a=torch.sigmoid(rand(b,t,h,m));beta=torch.sigmoid(rand(b,t,h))
    xs=[hist,p,r,v,a,beta]
    for c in (16,32):
        maps=prepare(hist,p,r,a,beta,slots,c)
        upper=maps[1].triu(1)
        exact(f'c{c}/OV_upper_triangle',upper,torch.zeros_like(upper))
        with torch.no_grad():out,_=triton_chunk(*xs,slots,initial,c)
        for cut in (1,7,15,16,17,31,32,33,63,64,65,127,128,129,191,192):
            # Change every future tensor and route together, keeping the past exact.
            ys=[x.clone() for x in xs];ss=slots.clone()
            for i,y in enumerate(ys):
                y[:,cut:]=torch.rand_like(y[:,cut:]) if i>=4 else torch.randn_like(y[:,cut:])*3
            ss[:,cut:]=torch.randint(m,ss[:,cut:].shape,device='cuda')
            with torch.no_grad():changed,_=triton_chunk(*ys,ss,initial,c)
            exact(f'c{c}/suffix_intervention/{cut}',out[:,:cut],changed[:,:cut])
            zs=[x.detach().clone().requires_grad_() for x in xs]
            result,_=triton_chunk(*zs,slots,initial,c)
            grads=torch.autograd.grad((result[:,cut-1]*torch.randn_like(result[:,cut-1])).sum(),zs)
            for i,g in enumerate(grads):exact(f'c{c}/future_gradient/{cut}/{i}',g[:,cut:],torch.zeros_like(g[:,cut:]))
        print('PASS chunk causal interventions',c,flush=True)
    for cut in (1,15,16,17,31,32,33,127,128,129,192):
        qq,kk,ss=q.clone(),k.clone(),slots.clone()
        qq[:,cut:]=rand(*qq[:,cut:].shape);kk[:,cut:]=rand(*kk[:,cut:].shape)
        ss[:,cut:]=torch.randint(m,ss[:,cut:].shape,device='cuda')
        changed=triton_geometry(qq,kk,ss,keys)
        for i,(x,y) in enumerate(zip((hist,p,r),changed[:3])):exact(f'geometry/future/{cut}/{i}',x[:,:cut],y[:,:cut])
        qg,kg=q.clone().requires_grad_(),k.clone().requires_grad_()
        geo=triton_geometry(qg,kg,slots,keys)
        grads=torch.autograd.grad(sum(z[:,cut-1].square().sum() for z in geo[:3]),(qg,kg))
        for i,g in enumerate(grads):exact(f'geometry/future_gradient/{cut}/{i}',g[:,cut:],torch.zeros_like(g[:,cut:]))
    print('PASS geometry causal interventions',flush=True)

@torch.no_grad()
def stream(model,ids,states=None,offset=0):
    b,t=ids.shape;x=model.embed(ids);new=[]
    for index,layer in enumerate(model.layers):
        mix=layer.mix;z=layer.n1(x)
        q,k,v=[f(z).reshape(b,t,base.H,base.K) for f in (mix.q,mix.k,mix.v)]
        alpha=mix.alpha(z)
        with torch.autocast('cuda',enabled=False):beta=torch.sigmoid(F.linear(z.float(),mix.beta.weight,mix.beta.bias))
        slots=(torch.arange(offset,offset+t,device='cuda')[None,:,None].expand(b,t,base.H)%base.M).contiguous()
        if states is None:
            keys=mix.keys[None].expand(b,base.H,base.M,base.K).contiguous()
            value=torch.zeros(b,base.H,base.M,base.K,device='cuda')
        else:keys,value=states[index]
        hist,p,r,keys=triton_geometry(q,k,slots,keys)
        # Decode uses the independent sequential Torch recurrence, no chunk maps.
        with torch.autocast('cuda',enabled=False):
            o,value=recurrent_reference(hist,p,r,v.float(),alpha,beta,slots,value)
        new.append((keys,value))
        y=mix.norm(o.float()).to(z.dtype).reshape(b,t,base.D)
        x=x+mix.out(y*F.silu(mix.gate(z)))
        u,w=layer.up(layer.n2(x)).chunk(2,-1);x=x+layer.down(F.silu(u)*w)
    return model.head(model.norm(x)[:,-1]).float(),new

def valid_brevo(ids,guess):
    end=ids.index(9997);query=ids[end+1]
    edges=list(zip(ids[1:end:2],ids[2:end:2]))
    nodes={query}
    while True:
        more=nodes|{a for a,b in edges if b in nodes}
        if more==nodes:break
        nodes=more
    if not guess or guess[-1]!=9998:return False
    answer=guess[:-1];pos={a:i for i,a in enumerate(answer)}
    return len(answer)==len(nodes) and set(answer)==nodes and all(pos[a]<pos[b] for a,b in edges if b in nodes)

def model_audit():
    rows=[r for r in base.load('confirm') if r['task']=='brevo'][:128]
    for seed in (731,732):
        name=f'gsa_slot-seed{seed}'
        model=exp.DecayLM('gsa_slot',seed).cuda().eval()
        saved=torch.load(CHECKPOINTS/name/'model.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(saved['model'],strict=True)
        ids,_=base.batch(rows[:4]);perturb=[]
        with torch.no_grad(),base.amp():
            out=model(ids)
            for cut in (1,15,16,17,31,32,33,63,64,65,127,128,129,191):
                altered=ids.clone();altered[:,cut:]=torch.randint(base.VOCAB,altered[:,cut:].shape,device='cuda')
                changed=model(altered)
                exact(name+f'/trained_model_future/{cut}',out[:,:cut],changed[:,:cut])
            # Direct answer-position test: existing teacher-forced suffix vs garbage.
            for row in rows[:16]:
                full,_=base.batch([row]);n=row['input_ids'].index(9997)+3
                altered=full.clone();altered[:,n:]=torch.randint(base.VOCAB,altered[:,n:].shape,device='cuda')
                exact(name+f'/gold_suffix/{len(perturb)}',model(full)[:,:n],model(altered)[:,:n]);perturb.append(n)
        original=json.loads((CHECKPOINTS/name/'generations.json').read_text())
        original=[r for r in original if r['task']=='brevo']
        decoded=[]
        # True prefix-only, one-token recurrent decode. No future padding/gold
        # tokens ever enter this call; correctness is derived from graph edges.
        with base.amp():
            for i,row in enumerate(rows):
                data=row['input_ids'];n=data.index(9997)+3
                prefix=torch.tensor([data[:n]],device='cuda')
                logits,state=stream(model,prefix);guess=[]
                for step in range(25):
                    token=int(logits.argmax(-1));guess.append(token)
                    if token==9998:break
                    logits,state=stream(model,torch.tensor([[token]],device='cuda'),state,n+step)
                decoded.append({'guess':guess,'valid':valid_brevo(data,guess),
                                'original_valid':valid_brevo(data,original[i]['guess']),
                                'same_guess':guess==original[i]['guess'],'n':row['meta']['n']})
                if i%32==31:print('DECODE',name,i+1,flush=True)
        report={'name':name,'n':len(decoded),'recurrent_valid':sum(r['valid'] for r in decoded),
                'original_valid':sum(r['original_valid'] for r in decoded),'same_guess':sum(r['same_guess'] for r in decoded),
                'records':decoded}
        REPORT['models'].append(report);save()
        print('MODEL RESULT',name,{k:v for k,v in report.items() if k!='records'},flush=True)
        del model;torch.cuda.empty_cache()

def save():
    (ROOT/'evidence').mkdir(exist_ok=True)
    (ROOT/'evidence/kernel-causality.json').write_text(json.dumps(REPORT,indent=2)+'\n')

if __name__=='__main__':
    torch.set_num_threads(4);torch.manual_seed(9041);torch.backends.cuda.matmul.allow_tf32=False
    REPORT['environment']={'torch':str(torch.__version__),'triton':triton.__version__,'gpu':torch.cuda.get_device_name()}
    REPORT['sources']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*sorted((ROOT/'komp_slot').glob('*.py'))]}
    try:kernel_audit();model_audit();REPORT['status']='passed'
    except Exception as e:REPORT['status']='failed';REPORT['error']=repr(e);raise
    finally:save()
