"""Native frozen GSA2 supplemental trials; shared experiment files stay unchanged."""
import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import experiment as base
import torch
from torch import nn

# Same native export restoration used by the validated GSA2 harness.
path=base.ops.FLA_ROOT/'fla/modules/__init__.py'
spec=importlib.util.spec_from_file_location('fla.modules',path,submodule_search_locations=[str(path.parent)])
module=sys.modules['fla.modules'];module.__file__=str(path);module.__package__='fla.modules'
module.__spec__=spec;module.__loader__=spec.loader;spec.loader.exec_module(module)
layers=types.ModuleType('fla.layers');layers.__path__=[str(base.ops.FLA_ROOT/'fla/layers')]
layers.__spec__=importlib.machinery.ModuleSpec('fla.layers',loader=None,is_package=True);sys.modules['fla.layers']=layers
from fla.layers.gsa2 import GatedSlotAttention2

ROOT=Path(os.environ['GSA2_TRIALS'])
VARIANTS={'gsa2_native_lr5e4':(True,5e-4),'gsa2_native_lr1e3':(True,1e-3),'gsa2_noconv_lr5e4':(False,5e-4)}
SCHEDULE=base.schedule


def hashes(model):
    return {n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for n,p in model.named_parameters() if '.mix.' not in n}


class NativeMixer(nn.Module):
    def __init__(self,i,use_conv):
        super().__init__()
        self.layer=GatedSlotAttention2(hidden_size=base.D,head_dim=base.K,num_heads=base.H,num_slots=64,
            expand_v=1.,mode='chunk',layer_idx=i,use_short_conv=use_conv,use_w_conv=use_conv)
    def forward(self,x):return self.layer(x,use_cache=False)[0]


class Model(base.LM):
    def __init__(self,kind,seed):
        super().__init__('gdn',seed);self.variant=kind
        for i,layer in enumerate(self.layers):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+7000+i);layer.mix=NativeMixer(i,VARIANTS[kind][0])


def optimizer(model):
    groups={}
    for name,p in model.named_parameters():
        residual=name.endswith(('mix.layer.o_proj.weight','.down.weight'))
        no_decay=bool(getattr(p,'_no_weight_decay',False))
        groups.setdefault((residual,no_decay),[]).append(p)
    return torch.optim.AdamW([dict(params=ps,weight_decay=0. if nd else .1,residual_output=res) for (res,nd),ps in groups.items()],lr=5e-4,betas=(.9,.95),fused=True)


def sources():
    root=Path(base.__file__).parents[1]
    paths=[Path(__file__),Path(base.__file__),root/'benchmark_fla.py',*sorted((root/'komp_slot').glob('*.py'))]
    out={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    out.update({'upstream/'+str(p.relative_to(base.ops.FLA_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((base.ops.FLA_ROOT/'fla').rglob('*.py'))})
    return out


def metadata(model):
    conv,lr=VARIANTS[model.variant]
    return dict(native_layer='fla.layers.gsa2.GatedSlotAttention2',slots=64,peak_lr=lr,
        lr=f'{lr:g} warmup5% cosine to10%; residual output multiplier .1->.3',
        short_conv=conv,use_w_conv=conv,conv_size=4,source_sha256=sources(),
        nonmixer_init_sha256=hashes(model),initialization='native mixer seed+7000+layer; same named nonmixer backbone',
        wd_exclusions=[n for n,p in model.named_parameters() if getattr(p,'_no_weight_decay',False)],
        recurrent_elements_per_head=8192,comparison_block_elements_per_head=4096,
        routing='native Oja2 -> GDN2, no KOMP eviction',
        selection='1e-3 selected in prior validation-only search; 5e-4 is common recipe, no new search')


def causal(model,ids):
    rows=[]
    for training in (True,False):
        model.train(training)
        with torch.no_grad(),base.amp():
            before=model(ids)
            for cut in (31,32,63,64,65,127,128,129):
                changed=ids.clone();changed[:,cut:]=(changed[:,cut:]+7)%base.VOCAB
                after=model(changed);err=float((before[:,:cut]-after[:,:cut]).abs().max())
                torch.testing.assert_close(before[:,:cut],after[:,:cut],atol=2e-5,rtol=2e-5)
                rows.append(dict(training=training,cut=cut,max_abs=err))
            changed=ids.clone();changed[1]=(changed[1]+11)%base.VOCAB
            after=model(changed)
            torch.testing.assert_close(before[0],after[0],atol=2e-5,rtol=2e-5)
            rows.append(dict(training=training,batch_isolation=float((before[0]-after[0]).abs().max())))
    model.train()
    return rows


def preflight():
    ROOT.mkdir(parents=True,exist_ok=True)
    sample=base.load('train')[:4];ids,_=base.batch(sample);report={'variants':{}}
    for kind in VARIANTS:
        model=Model(kind,731).cuda();reference=base.LM('gdn',731)
        assert hashes(model)==hashes(reference);del reference
        initial=causal(model,ids);opt=optimizer(model)
        residual={id(p) for g in opt.param_groups if g['residual_output'] for p in g['params']}
        assert residual=={id(p) for n,p in model.named_parameters() if n.endswith(('mix.layer.o_proj.weight','.down.weight'))}
        assert len(residual)==2*base.L
        for g in opt.param_groups:
            for p in g['params']:assert g['weight_decay']==(0. if getattr(p,'_no_weight_decay',False) else .1)
        x=torch.randn(2,129,base.D,device='cuda',requires_grad=True)
        with base.amp():y=model.layers[0].mix(x)
        dx=torch.autograd.grad(y[:,63].float().square().sum(),x)[0]
        assert torch.count_nonzero(dx[:,64:]).item()==0
        losses=[]
        for _ in range(16):
            opt.zero_grad(set_to_none=True)
            with base.amp():loss=base.loss_items(model,sample)[0].mean()
            loss.backward();assert bool(torch.isfinite(nn.utils.clip_grad_norm_(model.parameters(),1.)))
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            opt.step();losses.append(float(loss.detach()))
        assert losses[-1]<losses[0]
        report['variants'][kind]=dict(losses=losses,initial_causality=initial,trained_causality=causal(model,ids),future_gradient=0,params=sum(p.numel() for p in model.parameters()),nonmixer_init_matches=True)
        print('PASS',kind,losses[0],losses[-1],flush=True)
        del model,opt;torch.cuda.empty_cache()
    report.update(status='passed',source_sha256=sources())
    (ROOT/'preflight.json').write_text(json.dumps(report,indent=2))


def train(kind,seed):
    ticket=json.loads((ROOT/'preflight.json').read_text());assert ticket['status']=='passed' and ticket['source_sha256']==sources()
    assert kind in ticket['variants']
    peak=VARIANTS[kind][1]
    def schedule(step,total):
        lr,mult=SCHEDULE(step,total);return lr*(peak/5e-4),mult
    base.schedule=schedule
    base.train(kind,seed,model_factory=Model,optimizer_factory=optimizer,result_root=ROOT/'results',config_extra=metadata,preflight_path=ROOT/'preflight.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--kind',choices=VARIANTS);p.add_argument('--seed',type=int,default=731)
    args=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.preflight:preflight()
    else:train(args.kind,args.seed)
