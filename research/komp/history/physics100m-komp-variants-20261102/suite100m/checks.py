"""GPU acceptance for the actual C128 learned/decay paths before training."""
import hashlib
import torch
from torch.nn import functional as F
import base_models as base
from variants import LM,VariantMixer
from komp_slot.block_flex import block_komp_flex
from komp_slot.decay import gdn_decay

def check_variant(kind):
    torch.manual_seed(81731)
    original=base.Mixer('komp').cuda()
    mix=VariantMixer(original,kind).cuda()
    x=torch.randn(1,257,base.D,device='cuda')
    torch.testing.assert_close(mix.log_decay(x).exp(),torch.full((1,257,base.H,base.M),.98,device='cuda'),rtol=0,atol=1e-7)
    evidence={'initial_alpha':.98}
    print('CHECK',kind,'initial decay passed',flush=True)
    if mix.gdn:
        z=(torch.randn(1,7,base.H,base.M,device='cuda')*.2).requires_grad_()
        a=mix.A_log.detach().clone().requires_grad_();bias=mix.dt_bias.detach().clone().requires_grad_()
        alpha=gdn_decay(z,a,bias); ref=torch.exp(-a.exp()*F.softplus(z+bias))
        weight=torch.randn_like(alpha)
        aa=torch.autograd.grad((alpha*weight).sum(),(z,a,bias),retain_graph=True)
        bb=torch.autograd.grad((ref*weight).sum(),(z,a,bias))
        torch.testing.assert_close(alpha,ref,rtol=2e-6,atol=1e-7)
        for left,right in zip(aa,bb):torch.testing.assert_close(left,right,rtol=2e-5,atol=1e-6)
        with torch.no_grad():mix.decay.weight.normal_(0,.02)
        assert (mix.log_decay(x)-mix.log_decay(-x)).abs().max().item()>1e-4
        evidence['gdn_formula_and_gradients']='matches existing Triton helper; input dependence verified'
    if mix.learned:
        with torch.no_grad():mix.score.weight.normal_(0,.02)
    # Cross two C128 boundaries; nonzero random scores exercise true learned selection.
    for cut in (65,127,128,129,255):
        changed=x.clone();changed[:,cut:]+=torch.randn_like(changed[:,cut:])*.7
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            before=mix(x);after=mix(changed)
        torch.testing.assert_close(before[:,:cut],after[:,:cut],rtol=1e-5,atol=1e-5)
    evidence['causal_cuts']=[65,127,128,129,255]
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=(mix(x)-torch.randn_like(x)).square().mean()
    loss.backward()
    names=['decay.weight']+(['score.weight'] if mix.learned else [])+(['A_log','dt_bias'] if mix.gdn else [])
    evidence['gradient_norms']={}
    for name in names:
        grad=dict(mix.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0,name
        evidence['gradient_norms'][name]=grad.norm().item()
    print('CHECK',kind,'causality and gate gradients passed',flush=True)
    del mix,original,x
    # Operator reference covers output, state and all differentiable inputs at C128.
    torch.manual_seed(81732)
    shape=(1,257,2,64)
    tensors=[torch.randn(shape,device='cuda')*.2 for _ in range(3)]
    tensors += [torch.full((1,257,2,32),-.02,device='cuda'),torch.full((1,257,2),.2,device='cuda'),
                F.normalize(torch.randn(1,2,32,64,device='cuda'),dim=-1),torch.randn(1,2,32,64,device='cuda')*.01]
    fast=[v.detach().requires_grad_() for v in tensors];ref=[v.detach().clone().requires_grad_() for v in tensors]
    learned=kind!='komp_gdn_slot'
    extra=dict(evict='learned',scores=torch.randn(1,257,2,device='cuda'),initial_scores=torch.randn(1,2,32,device='cuda')) if learned else {}
    fo,fs=base.block_komp_fast(*fast[:5],tuple(fast[5:]),chunk_size=128,**extra)
    ro,rs=block_komp_flex(*ref[:5],tuple(ref[5:]),chunk_size=128,**extra)
    if not learned:
        # Fast FIFO returns chronological order; flex always returns physical slots.
        # Rotate both reference tensors inside autograd so the same loss cotangent
        # is applied to the same atom, including the partial final chunk.
        rs=tuple(torch.roll(state,shifts=-(shape[1] % 32),dims=2) for state in rs)
        expected=F.normalize(fast[1][:,-32:].transpose(1,2),dim=-1,eps=1e-6)
        torch.testing.assert_close(fs[0],expected,rtol=2e-6,atol=1e-6)
        torch.testing.assert_close(rs[0],expected,rtol=2e-6,atol=1e-6)
        evidence['state_order']='flex physical slots rotated to fast FIFO chronological order before output/state/gradient checks'
    evidence['max_output_error']=(fo-ro).abs().max().item()
    for left,right in zip((fo,*fs),(ro,*rs)):torch.testing.assert_close(left,right,rtol=3e-3,atol=3e-4)
    weights=[torch.randn_like(v) for v in (fo,*fs)]
    fg=torch.autograd.grad(sum((v*w).sum() for v,w in zip((fo,*fs),weights)),fast)
    rg=torch.autograd.grad(sum((v*w).sum() for v,w in zip((ro,*rs),weights)),ref)
    evidence['max_gradient_errors']=[(a-b).abs().max().item() for a,b in zip(fg,rg)]
    for a,b in zip(fg,rg):torch.testing.assert_close(a,b,rtol=1e-2,atol=1e-3)
    print('CHECK',kind,'operator forward/state/backward passed',flush=True)
    del fast,ref,fo,fs,ro,rs,fg,rg,tensors
    # Exact initial shared parameter fingerprint against the original backbone.
    reference=base.LM('komp',731);model=LM(kind,731)
    digest=hashlib.sha256();count=0
    params=dict(model.named_parameters())
    for name,value in reference.named_parameters():
        if '.decay.' in name:continue
        assert torch.equal(value,params[name]),name
        digest.update(name.encode());digest.update(value.detach().numpy().tobytes());count+=value.numel()
    evidence.update(shared_sha256=digest.hexdigest(),shared_parameters=count)
    del reference,model
    torch.cuda.empty_cache()
    return evidence
