"""Matched frozen-kernel KAPA trials; no edits to shared experiment/kernel files."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import experiment as base
import torch
from torch import nn
import torch.nn.functional as F
from komp_slot.kapa_chunk import chunk_gaussian_protected_kapa

ROOT=Path(__file__).resolve().parent/os.environ.get('KAPA_RUNS','block_runs_v4/kapa')
VARIANTS={'kapa_scalar_p2_c16':False,'kapa_slot_p2_c16':True}


class KapaMixer(base.Mixer):
    def __init__(self,original,slot):
        super().__init__('komp');self.load_state_dict(original.state_dict());self.slot=slot
        if slot:
            self.decay=nn.Linear(base.D,base.H*base.M)
            nn.init.zeros_(self.decay.weight)
            alpha=.98**8
            nn.init.constant_(self.decay.bias,math.log(alpha/(1-alpha)))

    def forward(self,x):
        b,t,_=x.shape
        q,k,v=[f(x).reshape(b,t,base.H,base.K) for f in (self.q,self.k,self.v)]
        with torch.autocast('cuda',enabled=False):
            g=F.logsigmoid(F.linear(x.float(),self.decay.weight,self.decay.bias))
            if self.slot:g=(g/8).reshape(b,t,base.H,base.M)
            beta=torch.sigmoid(F.linear(x.float(),self.beta.weight,self.beta.bias))
        centers=self.keys[None].expand(b,base.H,base.M,base.K).contiguous()
        initial=torch.zeros(b,base.H,base.M,base.K,device=x.device)
        o,_=chunk_gaussian_protected_kapa(q,k,v,g,beta,centers,initial,
                                         anchors=2,regularization=.05,temperature=.5,gram_ridge=.01,chunk_size=16)
        y=self.norm(o.float()).to(x.dtype).reshape(b,t,base.D)
        return self.out(y*F.silu(self.gate(x)))


class KapaLM(base.LM):
    def __init__(self,kind,seed):
        super().__init__('komp',seed);self.variant=kind
        for layer in self.layers:layer.mix=KapaMixer(layer.mix,VARIANTS[kind])


def source_hashes():
    root=Path(base.__file__).parents[1]
    paths=[Path(__file__),Path(base.__file__),*sorted((root/'komp_slot').glob('*.py'))]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def metadata(model):
    return dict(kapa_variant=model.variant,source_sha256=source_hashes(),
                kernel='frozen r3 kapa_chunk; all-Triton first fwd/bwd',anchors=2,chunk_size=16,
                regularization=.05,temperature=.5,gram_ridge=.01,short_conv=False,
                routing='fixed trainable centers, independent of sequence; no eviction',
                normalization='raw Gaussian q/k/centers without L2 normalization',
                decay='slot logsigmoid/8' if VARIANTS[model.variant] else 'scalar logsigmoid',
                initial_alpha=.98,initial_beta=.2,weight_decay_policy='all parameters .1, same as block arms')


def preflight():
    ROOT.mkdir(parents=True,exist_ok=True)
    sample=base.load('train')[:4];ids,_=base.batch(sample)
    report={'variants':{}}
    for kind in VARIANTS:
        model=KapaLM(kind,731).cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
        causal=[]
        with torch.no_grad(),base.amp():
            before=model(ids)
            for cut in (15,16,17,40,64,65):
                changed=ids.clone();changed[:,cut:]=(changed[:,cut:]+7)%base.VOCAB
                after=model(changed);error=float((before[:,:cut]-after[:,:cut]).abs().max())
                torch.testing.assert_close(before[:,:cut],after[:,:cut],atol=1e-5,rtol=1e-5)
                causal.append(dict(cut=cut,max_abs=error))
        losses=[]
        for step in range(16):
            opt.zero_grad(set_to_none=True)
            with base.amp():loss=base.loss_items(model,sample)[0].mean()
            loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert bool(torch.isfinite(norm))
            if step==0:
                for layer in model.layers:
                    assert layer.mix.keys.grad is not None and bool(torch.isfinite(layer.mix.keys.grad).all())
            opt.step();losses.append(float(loss.detach()))
        assert losses[-1]<losses[0],(kind,losses)
        report['variants'][kind]=dict(losses=losses,causality=causal,params=sum(p.numel() for p in model.parameters()))
        print('PASS PREFLIGHT',kind,losses[0],losses[-1],flush=True)
        del model,opt;torch.cuda.empty_cache()
    report.update(status='passed',source_sha256=source_hashes())
    (ROOT/'preflight.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--kind',choices=VARIANTS);parser.add_argument('--seed',type=int,default=731)
    args=parser.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if args.preflight:preflight()
    else:
        ticket=json.loads((ROOT/'preflight.json').read_text())
        assert ticket['status']=='passed' and ticket['source_sha256']==source_hashes()
        assert args.kind in ticket['variants']
        base.train(args.kind,args.seed,model_factory=KapaLM,result_root=ROOT/'results',
                   config_extra=metadata,preflight_path=ROOT/'preflight.json')
