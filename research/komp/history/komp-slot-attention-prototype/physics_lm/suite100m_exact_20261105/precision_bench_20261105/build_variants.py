"""Isolated BF16 projection experiments; never edit the frozen operator."""
from pathlib import Path
import hashlib
import json
import shutil
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'snapshot/komp_slot/exact_komp'
BF16_SWEEP='''@tr.jit
def direct_sweep(gram,rhs,M:tl.constexpr,BM:tl.constexpr):
    # BF16-rounded elementary operations; reductions accumulate in FP32.
    # Casting the regularized Gram here intentionally exposes whether ridge
    # survives BF16 representation. No hidden FP32 factorization fallback.
    r=tl.arange(0,BM)
    a=gram.to(tl.bfloat16)
    sol=rhs.to(tl.bfloat16)
    for j in tl.static_range(M):
        col=tl.sum(tl.where(r[None,:]==j,a,0.).to(tl.float32),1).to(tl.bfloat16)
        pivot=tl.sum(tl.where(r==j,col,0.).to(tl.float32),0).to(tl.bfloat16)
        c=(col.to(tl.float32)/pivot.to(tl.float32)).to(tl.bfloat16)
        bj=(tl.sum(tl.where(r==j,sol,0.).to(tl.float32),0)/pivot.to(tl.float32)).to(tl.bfloat16)
        prod=(col.to(tl.float32)*bj.to(tl.float32)).to(tl.bfloat16)
        sol=tl.where(r==j,bj,(sol.to(tl.float32)-prod.to(tl.float32)).to(tl.bfloat16))
        outer=(col[:,None].to(tl.float32)*c[None,:].to(tl.float32)).to(tl.bfloat16)
        a=(a.to(tl.float32)-outer.to(tl.float32)).to(tl.bfloat16)
        a=tl.where(r[:,None]==j,c[None,:],tl.where(r[None,:]==j,c[:,None],a))
        a=tl.where((r[:,None]==j)&(r[None,:]==j),(-1./pivot.to(tl.float32)).to(tl.bfloat16),a)
    return sol.to(tl.float32)


'''
for name in ('bf16_projection','bf16_projection_mixed'):
    dest=ROOT/name
    if dest.exists():
        raise FileExistsError(f'Preserve existing experiment: {dest}')
    shutil.copytree(SOURCE,dest,ignore=shutil.ignore_patterns('__pycache__'))
    p=dest/'operator.py';s=p.read_text()
    s=s.replace('p,h,r=[alloc((bh,t,m),dtype) for _ in range(3)];n=alloc((bh,t),dtype)',
                'p,h,r=[alloc((bh,t,m),torch.bfloat16) for _ in range(3)];n=alloc((bh,t),torch.bfloat16)')
    s=s.replace('reuse=learned and not fp64 and m<=32','reuse=False  # FIFO-only isolated precision experiment')
    s=s.replace("if projection_dtype not in ('float32','float64'):","if eviction != 'fifo' or projection_dtype != 'float32':\n        raise ValueError('Isolated BF16 projection variant accepts FIFO and its FP32 normalization path only')\n    if projection_dtype not in ('float32','float64'):")
    p.write_text(s)
    p=dest/'geometry.py';s=p.read_text()
    old='    return old,kept,dropped,gram,cross,hist,read,qnew'
    new='''    # Gram/RHS/kernel tapes are represented in BF16; normalized vectors stay FP32.
    gram=gram.to(tl.bfloat16).to(tl.float32)
    cross=cross.to(tl.bfloat16).to(tl.float32)
    hist=hist.to(tl.bfloat16).to(tl.float32)
    read=read.to(tl.bfloat16).to(tl.float32)
    qnew=qnew.to(tl.bfloat16).to(tl.float32)
    return old,kept,dropped,gram,cross,hist,read,qnew'''
    assert old in s;s=s.replace(old,new)
    if name=='bf16_projection':
        start=s.index('@tr.jit\ndef direct_sweep(');end=s.index('@tr.jit\ndef factor_ldlt(',start)
        s=s[:start]+BF16_SWEEP+s[end:]
    p.write_text(s)
manifest={'source':{str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCE.glob('*.py'))},
          'variants':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('bf16_projection*/*.py'))},
          'semantics':{'bf16_projection':'BF16 Gram/RHS/P/H/R/N; BF16-rounded elimination, including regularized diagonal. FP32 normalization, reductions, state, geometry adjoint. Approximate backward through rounded geometry.',
                       'bf16_projection_mixed':'BF16 Gram/RHS/P/H/R/N; ridge addition and solve FP32. FP32 normalization/state/geometry adjoint. Approximate backward through rounded geometry; not a pure BF16 solve.'}}
(ROOT/'variants.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'built':list(manifest['semantics'])}))
