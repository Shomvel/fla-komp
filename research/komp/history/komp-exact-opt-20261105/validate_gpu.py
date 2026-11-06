"""Real CUDA differential acceptance; stop on first unexpected error."""
import argparse,json,time,traceback
from pathlib import Path
import torch
from exact_komp import exact_komp
from reference import block_komp_token_reference

ROOT=Path(__file__).resolve().parent
checks=[]


def check(name,got,want,rms_tol=.005,max_tol=.005):
    a,b=got.detach().double(),want.detach().double()
    delta=a-b
    rms=float(delta.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-6))
    maxabs=float(delta.abs().max())
    nmax=maxabs/max(1.,float(b.abs().max()))
    row=dict(name=name,rms_relative=rms,normalized_max=nmax,max_abs=maxabs,
             rms_tolerance=rms_tol,max_tolerance=max_tol)
    checks.append(row)
    if not torch.isfinite(a).all() or rms>rms_tol or nmax>max_tol:
        raise AssertionError(row)


def make(seed,b,t,h,m,d,dv,dtype,pattern='random',scalar=False):
    rng=torch.Generator().manual_seed(seed)
    def rand(*shape):return torch.randn(*shape,generator=rng).to('cuda',dtype)
    q,k,v=rand(b,t,h,d),rand(b,t,h,d),rand(b,t,h,dv)
    k0=rand(b,h,m,d);s0=.1*rand(b,h,m,dv).float()
    if pattern=='near':
        center=rand(b,1,h,d)
        k=(center+.002*k).to(dtype)
        k0=(center.transpose(1,2)+.002*k0).to(dtype)
    g=(-.01-.1*torch.rand(b,t,h,m,generator=rng)).cuda()
    if scalar:g=g[...,0].contiguous()
    beta=torch.sigmoid(rand(b,t,h).float())
    scores=rand(b,t,h).float()
    init_score=rand(b,h,m).float()
    if pattern=='ties':scores.zero_();init_score.zero_()
    if pattern=='reject':scores=-torch.arange(1,t+1,device='cuda').float()[None,:,None].expand(b,t,h).contiguous();init_score.zero_()
    if pattern=='replace':scores=torch.arange(1,t+1,device='cuda').float()[None,:,None].expand(b,t,h).contiguous();init_score.zero_()
    if pattern=='strong':g*=30
    return q,k,v,g,beta,k0,s0,scores,init_score


def reference(data,learned,gate):
    q,k,v,g,beta,k0,s0,scores,init=data
    beta=beta*(2*scores.sigmoid()) if gate else beta
    return block_komp_token_reference(q,k,v,g,beta,(k0,s0),chunk_size=1,
        evict='learned' if learned else 'fifo',scores=scores.detach(),initial_scores=init.detach())


def invoke(data,learned,gate,chunk,**extra):
    q,k,v,g,beta,k0,s0,scores,init=data
    return exact_komp(q,k,v,g,beta,(k0,s0),eviction='learned' if learned else 'fifo',scores=scores,
                     initial_scores=init,score_write_gate=gate,compute_chunk=chunk,**extra)


def case(seed,b,t,h,m,d,dv,dtype,pattern,learned,chunk,grad=True,scalar=False,gate=None):
    gate=learned if gate is None else gate
    data=make(seed,b,t,h,m,d,dv,dtype,pattern,scalar)
    data=tuple(x.clone().requires_grad_(i<8) for i,x in enumerate(data))
    gold=tuple(x.detach().double().requires_grad_(i<8) for i,x in enumerate(data))
    name=f's{seed}/B{b}T{t}H{h}M{m}K{d}V{dv}/{dtype}/{pattern}/learned{learned}/C{chunk}'
    print('START',name,flush=True)
    actual,(ak,ass),_=invoke(data,learned,gate,chunk)
    expected,(ek,es)=reference(gold,learned,gate)
    low=dtype==torch.bfloat16
    rt=.025 if low else .005;mt=.03 if low else .005
    check(name+'/output',actual,expected,rt,mt)
    check(name+'/keys',ak,ek,rt,mt)
    check(name+'/state',ass,es,rt,mt)
    if grad:
        rng=torch.Generator().manual_seed(177)
        weights=[torch.randn(x.shape,generator=rng).cuda() for x in (actual,ak,ass)]
        loss=sum((x.float()*w).sum() for x,w in zip((actual,ak,ass),weights))
        target=sum((x*w.double()).sum() for x,w in zip((expected,ek,es),weights))
        da=torch.autograd.grad(loss,data[:8],allow_unused=True)
        de=torch.autograd.grad(target,gold[:8],allow_unused=True)
        for label,a,e in zip(['q','k','v','g','beta','initial_keys','initial_coefficients','scores'],da,de):
            if e is None:
                if a is not None:check(name+'/grad/'+label,a,torch.zeros_like(a),0.,0.)
            else:check(name+'/grad/'+label,a,e,rt,mt)
    print('PASS',name,flush=True)


def extras():
    for learned in (False,True):
        data=make(600,2,35,2,8,16,17,torch.float32,'ties')
        full,(fk,fs),fm=invoke(data,learned,learned,16)
        alt,(ak,as_),_=invoke(data,learned,learned,32)
        check(f'extra/{learned}/chunk/output',alt,full)
        check(f'extra/{learned}/chunk/state',as_,fs)
        cut=11
        first=tuple(x[:,:cut] if i in [0,1,2,3,4,7] else x for i,x in enumerate(data))
        y1,(k1,s1),m1=invoke(first,learned,learned,16)
        rest=list(x[:,cut:] if i in [0,1,2,3,4,7] else x for i,x in enumerate(data))
        rest[5]=k1;rest[6]=s1;rest[8]=m1['scores']
        y2,(k2,s2),m2=invoke(tuple(rest),learned,learned,32,initial_births=m1['births'],start_position=cut)
        check(f'extra/{learned}/stream/output',torch.cat((y1,y2),1),full)
        check(f'extra/{learned}/stream/state',s2,fs)
        check(f'extra/{learned}/stream/keys',k2,fk)
        changed=[x.clone() for x in data]
        for i in (0,1,2,3,4,7):changed[i][:,cut:]+=.15
        future,_,_=invoke(tuple(changed),learned,learned,16)
        check(f'extra/{learned}/causality',future[:,:cut],full[:,:cut],0.,0.)


def suite(mode):
    if mode=='preflight':
        for learned in [False,True]:case(10,1,9,2,4,8,5,torch.float32,'random',learned,16)
    else:
        for learned in [False,True]:
            for pattern in ['random','ties','near','strong','reject','replace']:
                case(11,1,19,2,8,16,17,torch.float32,pattern,learned,16)
            for dtype in [torch.float32,torch.bfloat16]:
                case(12,2,33,2,32,64,64,dtype,'random',learned,32)
                case(13,1,65,1,32,64,64,dtype,'near',learned,16)
            case(14,1,1,1,7,9,11,torch.float32,'random',learned,16,scalar=True)
            case(15,1,31,1,64,128,128,torch.float32,'random',learned,32)
            case(16,1,129,2,8,32,32,torch.float32,'random',learned,16,grad=False,scalar=True)
        case(17,1,19,2,8,16,17,torch.float32,'random',True,16,gate=False)
        extras()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['preflight','full'],default='full');args=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    started=time.perf_counter()
    try:
        suite(args.mode)
        torch.cuda.synchronize()
    except Exception as e:
        (ROOT/'evidence'/f'{args.mode}-failure.json').write_text(json.dumps({'error':str(e),'traceback':traceback.format_exc(),'checks':checks},indent=2))
        raise
    result={'status':'passed','mode':args.mode,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'elapsed_seconds':time.perf_counter()-started,'checks':checks,'check_count':len(checks)}
    (ROOT/'evidence'/f'{args.mode}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
