"""Prespecified KOMP decay ablation on the same unique-token corpus."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import experiment as base
import torch
from torch import nn
import torch.nn.functional as F
from komp_slot.decay import gdn_decay, gsa_decay, matched_dt_bias

ROOT=Path(__file__).resolve().parent/'decay_runs'
VARIANTS=('sigmoid','gdn_scalar','gsa_scalar','gsa_slot')


class DecayMixer(base.Mixer):
    def __init__(self,original,variant,seed,layer):
        super().__init__('komp')
        self.load_state_dict(original.state_dict())
        self.variant=variant
        if variant=='gdn_scalar':
            self.decay=nn.Linear(base.D,base.H,bias=False)
            nn.init.zeros_(self.decay.weight)
            self.A_log=nn.Parameter(torch.zeros(base.H))
            self.dt_bias=nn.Parameter(torch.full((base.H,),matched_dt_bias(.98)))
        elif variant in ('gsa_scalar','gsa_slot'):
            self.decay=nn.Linear(base.D,base.H*(base.M if variant=='gsa_slot' else 1))
            # Identical scalar/slot initial function; slot keys and routes already
            # break permutation symmetry, so zero weights here are not GSA collapse.
            nn.init.zeros_(self.decay.weight)
            a=.98**8
            nn.init.constant_(self.decay.bias,math.log(a/(1-a)))
        self.last_alpha=None

    def alpha(self,x):
        # Keep the matched initial biases out of BF16 autocast rounding.
        with torch.autocast('cuda',enabled=False):
            logits=F.linear(x.float(),self.decay.weight,self.decay.bias)
        if self.variant=='gdn_scalar':return gdn_decay(logits,self.A_log,self.dt_bias)
        if self.variant=='sigmoid':return gsa_decay(logits,normalizer=1.)
        if self.variant=='gsa_slot':logits=logits.reshape(*x.shape[:2],base.H,base.M)
        return gsa_decay(logits,normalizer=8.)

    def forward(self,x):
        b,t,_=x.shape
        q,k,v=[f(x).reshape(b,t,base.H,base.K) for f in (self.q,self.k,self.v)]
        alpha=self.alpha(x)
        with torch.autocast('cuda',enabled=False):
            beta=torch.sigmoid(F.linear(x.float(),self.beta.weight,self.beta.bias))
        self.last_alpha=alpha.detach()
        slots=torch.arange(t,device=x.device)[None,:,None].expand(b,t,base.H).remainder(base.M).contiguous()
        keys=self.keys[None].expand(b,base.H,base.M,base.K).contiguous()
        initial=torch.zeros(b,base.H,base.M,base.K,device=x.device)
        h,p,r,_=base.ops.triton_geometry(q,k,slots,keys)
        o,_=base.ops.triton_chunk(h,p,r,v,alpha,beta,slots,initial)
        y=self.norm(o.float()).to(x.dtype).reshape(b,t,base.D)
        return self.out(y*F.silu(self.gate(x)))


class DecayLM(base.LM):
    def __init__(self,variant,seed):
        super().__init__('komp',seed)
        self.variant=variant
        for i,layer in enumerate(self.layers):
            layer.mix=DecayMixer(layer.mix,variant,seed,i)

    @torch.no_grad()
    def decay_diagnostics(self):
        result=[]
        for layer in self.layers:
            mix=layer.mix;a=mix.last_alpha
            row={'alpha_mean':a.mean().item(),'alpha_min':a.min().item(),'alpha_max':a.max().item(),
                 'key_norm':mix.keys.norm(dim=-1).mean().item()}
            if a.ndim==4:row['slot_std']=a.std(dim=-1).mean().item()
            if hasattr(mix,'A_log'):row['rate']=mix.A_log.exp().tolist()
            result.append(row)
        return result


def optimizer(model):
    groups={}
    for name,p in model.named_parameters():
        residual=name.endswith(('mix.out.weight','down.weight'))
        no_decay=name.endswith(('decay.bias','A_log','dt_bias'))
        groups.setdefault((residual,no_decay),[]).append(p)
    return torch.optim.AdamW([{'params':ps,'weight_decay':0. if nd else .1,'residual_output':res}
                             for (res,nd),ps in groups.items()],lr=5e-4,betas=(.9,.95),fused=True)


def metadata(model):
    sources=[Path(__file__),Path(base.__file__),*sorted((Path(base.__file__).parents[1]/'komp_slot').glob('*.py'))]
    common={n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
            for n,p in model.named_parameters() if '.decay.' not in n and not n.endswith(('A_log','dt_bias'))}
    return {'decay_variant':model.variant,'initial_alpha':.98,'initial_beta':.2,'geometry':'baseline IEEE, unchanged',
            'weight_decay_policy':'decay.bias/A_log/dt_bias=0; all remaining parameters=.1',
            'shared_parameter_sha256':common,
            'ablation_source_sha256':{str(p.relative_to(Path(base.__file__).parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}


def source_hashes():
    root=Path(base.__file__).parents[1]
    sources=[Path(__file__),Path(base.__file__),*sorted((root/'komp_slot').glob('*.py'))]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}


def preflight():
    ROOT.mkdir(exist_ok=True)
    sample=base.load('train')[:4];ids,_=base.batch(sample)
    report={'status':'running','variants':{}}
    shared=[];initial=[];initial_bf16=[]
    for kind in VARIANTS:
        model=DecayLM(kind,731).cuda();opt=optimizer(model)
        shared.append(metadata(model)['shared_parameter_sha256'])
        gate_initial=[]
        with torch.no_grad(),base.amp():
            for layer in model.layers:
                gate_input=layer.n1(model.embed(ids))
                a=layer.mix.alpha(gate_input)
                assert float((a-.98).abs().max())<2e-6
                with torch.autocast('cuda',enabled=False):
                    beta=torch.sigmoid(F.linear(gate_input.float(),layer.mix.beta.weight,layer.mix.beta.bias))
                assert float((beta-.2).abs().max())<2e-6
                gate_initial.append({'alpha_mean':float(a.mean()),'alpha_max_error':float((a-.98).abs().max()),
                                     'beta_mean':float(beta.mean()),'beta_max_error':float((beta-.2).abs().max())})
            before=model(ids);changed=ids.clone();changed[:,40:]=(changed[:,40:]+7)%base.VOCAB
            after=model(changed)
            torch.testing.assert_close(before[:,:40],after[:,:40],atol=2e-5,rtol=2e-5)
            initial_bf16.append(before.detach().float().cpu())
        # Compare the initial mathematical functions before BF16 rounding can
        # amplify tiny recurrence-order differences through four model layers.
        with torch.no_grad(),torch.autocast('cuda',enabled=False):
            initial.append(model(ids).detach().float().cpu())
        losses=[];slot_gradient=None
        for step in range(16):
            opt.zero_grad(set_to_none=True)
            with base.amp():loss=base.loss_items(model,sample)[0].mean()
            loss.backward()
            norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert bool(torch.isfinite(norm))
            if kind=='gsa_slot' and step==0:
                slot_gradient=[]
                for layer in model.layers:
                    grad=layer.mix.decay.weight.grad.reshape(base.H,base.M,base.D)
                    per_head=grad.std(dim=1).amax(dim=-1)
                    assert bool((per_head>1e-8).all()),'KOMP slot gate gradients failed to break symmetry'
                    slot_gradient.append(per_head.tolist())
            opt.step();losses.append(float(loss.detach()))
        assert losses[-1]<losses[0]
        report['variants'][kind]={'losses':losses,'slot_gradient_std':slot_gradient,'causality':'passed','initial_gates':gate_initial}
        print('PASS PREFLIGHT',kind,losses[0],losses[-1],flush=True)
        del model,opt;torch.cuda.empty_cache()
    assert all(x==shared[0] for x in shared)
    report['initial_equivalence']={}
    for i,kind in enumerate(VARIANTS[1:],1):
        out=initial[i]
        torch.testing.assert_close(out,initial[0],atol=3e-4,rtol=3e-4,msg=kind+'/fp32_initial')
        report['initial_equivalence'][kind]={
            'fp32_max_abs':float((out-initial[0]).abs().max()),
            'bf16_max_abs':float((initial_bf16[i]-initial_bf16[0]).abs().max()),
            'bf16_rms_relative':float((initial_bf16[i]-initial_bf16[0]).square().mean().sqrt()/initial_bf16[0].square().mean().sqrt())}
    report['status']='passed'
    report['source_sha256']=source_hashes()
    (ROOT/'preflight.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true')
    p.add_argument('--kind',choices=VARIANTS);p.add_argument('--seed',type=int,default=731)
    args=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if args.preflight:preflight()
    else:
        ticket=json.loads((ROOT/'preflight.json').read_text())
        assert ticket['status']=='passed' and ticket['source_sha256']==source_hashes(),'Preflight does not match training source'
        root=ROOT/'results';root.mkdir(exist_ok=True,parents=True)
        base.train(args.kind,args.seed,model_factory=DecayLM,result_root=root,
                   optimizer_factory=optimizer,config_extra=metadata,preflight_path=ROOT/'preflight.json')
