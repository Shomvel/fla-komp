"""Differential checks against frozen learned eviction with algorithmic C=1."""
import json
import time
import torch
from learned_c1_komp import parallel_learned
from validate import ROOT, SOURCE, module, inputs, compare, checks


def reference(x,scores,initial_scores,gate=True):
    beta=x[4]*(2*scores.sigmoid()) if gate else x[4]
    return module.block_komp_token_reference(*x[:4],beta,(x[5],x[6]),chunk_size=1,
        evict='learned',scores=scores.detach(),initial_scores=initial_scores)


def call(x,scores,initial_scores,**kw):
    return parallel_learned(*x[:5],(x[5],x[6]),scores,initial_scores=initial_scores,**kw)


def scenario(pattern,dtype):
    length,m=(35,32) if pattern=='m32' else (19,4)
    x=inputs(131,length,m,8,5,dtype,near_duplicate=pattern=='near')
    rng=torch.Generator().manual_seed(83)
    scores=torch.randn(1,length,2,generator=rng,dtype=dtype)
    initial=torch.randn(1,2,m,generator=rng,dtype=dtype)
    if pattern=='replace':
        initial.zero_();scores=torch.arange(1,length+1,dtype=dtype)[None,:,None].expand_as(scores)
    elif pattern=='reject':
        initial.zero_();scores=-torch.arange(1,length+1,dtype=dtype)[None,:,None].expand_as(scores)
    elif pattern=='ties':
        initial.zero_();scores.zero_()
    elif pattern=='mixed_ties':
        scores=scores.round();initial=initial.round()
    return x,scores,initial


def forward_cases():
    counts={}
    for dtype in [torch.float64,torch.float32]:
        for pattern in ['replace','reject','ties','mixed_ties','random','near','m32']:
            x,scores,initial=scenario(pattern,dtype)
            # A high precision oracle is necessary: the old FP32 reference also
            # has amplified rounding error for near-duplicate Gram matrices.
            oracle=tuple(t.double() for t in x)
            expected,(ek,es)=reference(oracle,scores.double(),initial.double())
            tol=2e-9 if dtype==torch.float64 else 3e-3
            for c in [1,4,16,32]:
                actual,(ak,as_),meta=call(x,scores,initial,compute_chunk=c)
                prefix=f'{dtype}/projection64/{pattern}/C{c}'
                compare(prefix+'/output',actual,expected,tol)
                compare(prefix+'/keys',ak,ek,tol)
                compare(prefix+'/state',as_,es,tol)
                if c==16 and dtype==torch.float64:
                    dropped=meta['dropped_slots']
                    counts[pattern]={'old_atom_deleted':int((dropped>=0).sum()),'new_atom_deleted':int((dropped<0).sum())}
    return counts


def gradients():
    for pattern in ['random','reject','near']:
        source,scores0,initial=scenario(pattern,torch.float64)
        rng=torch.Generator().manual_seed(331)
        wo=torch.randn(1,19,2,5,generator=rng,dtype=torch.float64)
        ws=torch.randn(1,2,4,5,generator=rng,dtype=torch.float64)
        for c in [4,32]:
            grads=[]
            for mode in ['reference','parallel']:
                x=tuple(t.clone().requires_grad_() for t in source)
                scores=scores0.clone().requires_grad_()
                if mode=='reference':y,(_,s)=reference(x,scores,initial)
                else:y,(_,s),_=call(x,scores,initial,compute_chunk=c)
                grads.append(torch.autograd.grad((y*wo).sum()+(s*ws).sum(),(*x,scores)))
            for name,a,b in zip(['q','k','v','g','beta','initial_keys','initial_coefficients','scores'],grads[1],grads[0]):
                compare(f'gradient/{pattern}/C{c}/{name}',a,b,2e-8)


def streaming_and_causality():
    for pattern in ['mixed_ties','random']:
        x,scores,initial=scenario(pattern,torch.float64)
        full,(fk,fs),fm=call(x,scores,initial,compute_chunk=16)
        cut=7
        part=tuple(t[:,:cut] for t in x[:5])+x[5:]
        y1,(k1,s1),m1=call(part,scores[:,:cut],initial,compute_chunk=4)
        part=tuple(t[:,cut:] for t in x[:5])+(k1,s1)
        y2,(k2,s2),m2=call(part,scores[:,cut:],m1['scores'],initial_births=m1['births'],start_position=cut,compute_chunk=9)
        compare(pattern+'/stream/output',torch.cat((y1,y2),1),full,2e-9)
        compare(pattern+'/stream/keys',k2,fk,2e-9)
        compare(pattern+'/stream/state',s2,fs,2e-9)
        compare(pattern+'/stream/scores',m2['scores'],fm['scores'],0.)
        compare(pattern+'/stream/births',m2['births'],fm['births'],0.)
        changed=[]
        for t in x[:5]:
            t=t.clone();t[:,cut:]+=.13;changed.append(t)
        scores2=scores.clone();scores2[:,cut:]+=10
        future,_,_=call(tuple(changed)+x[5:],scores2,initial,compute_chunk=16)
        compare(pattern+'/causality',future[:,:cut],full[:,:cut],0.)
    # Multiple sequences must keep separate dictionaries and independent slot schedules.
    x,scores,initial=scenario('mixed_ties',torch.float64)
    xb=tuple(torch.cat((t,t*.9),0) for t in x)
    sb=torch.cat((scores,-scores),0);ib=torch.cat((initial,-initial),0)
    batched,(bk,bs),_=call(xb,sb,ib,compute_chunk=4)
    for bi in range(2):
        xi=tuple(t[bi:bi+1] for t in xb)
        y,(k,s)=reference(xi,sb[bi:bi+1],ib[bi:bi+1])
        compare(f'batch{bi}/output',batched[bi:bi+1],y,2e-9)
        compare(f'batch{bi}/state',bs[bi:bi+1],s,2e-9)
    # Pure hard selection without the optional write gate remains the same forward rule.
    y,(k,s),_=call(x,scores,initial,compute_chunk=4,score_write_gate=False)
    yr,(kr,sr)=reference(x,scores,initial,gate=False)
    compare('no_score_gate/output',y,yr,2e-9)
    compare('no_score_gate/state',s,sr,2e-9)


if __name__=='__main__':
    started=time.perf_counter()
    try:
        branches=forward_cases()
        gradients()
        streaming_and_causality()
    except Exception as exc:
        (ROOT/'learned-validation-failure.json').write_text(json.dumps({'error':str(exc),'checks':checks},indent=2))
        raise
    result={'status':'passed','device':'CPU','torch':torch.__version__,'reference':str(SOURCE),
            'reference_dtype':'float64','projection_dtype':'float64',
            'coefficient_dtypes':['float64','float32'],
            'check_count':len(checks),'checks':checks,'branch_counts':branches,
            'elapsed_seconds':time.perf_counter()-started}
    (ROOT/'learned-validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
