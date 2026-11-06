"""Frozen Block-KOMP projection ablation and matched softmax attention experiments."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import block_experiment as be
import experiment as base
import torch
from torch import nn
import torch.nn.functional as F
from komp_slot.block_flex import block_komp_flex

ROOT = Path(os.environ.get('BLOCK_RUNS', Path(__file__).parent/'block_runs_proj'))
VARIANTS = {
    'fifo_proj_c64': dict(chunk=64, decay='slot', project=True),
    'fifo_noproj_c64': dict(chunk=64, decay='slot', project=False),
    'learned_proj_c64': dict(chunk=64, decay='slot', evict='learned', project=True),
    'learned_noproj_c64': dict(chunk=64, decay='slot', evict='learned', project=False),
    'swa_w32': dict(window=32, rope=True),
    'swa_w96': dict(window=96, rope=True),
    'swa_w96_nope': dict(window=96, rope=False),
    'full_attn': dict(window=192, rope=True),
    'fifo_proj_c128': dict(chunk=128, decay='slot', project=True),
    'fifo_noproj_c128': dict(chunk=128, decay='slot', project=False),
}


class ProjectionMixer(be.BlockMixer):
    def __init__(self, original, cfg):
        super().__init__(original, cfg)
        self.project = cfg['project']

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = [f(x).reshape(b, t, base.H, base.K) for f in (self.q, self.k, self.v)]
        with torch.autocast('cuda', enabled=False):
            g = (F.logsigmoid(F.linear(x.float(), self.decay.weight, self.decay.bias))/8.).reshape(b,t,base.H,base.M)
            beta = torch.sigmoid(F.linear(x.float(), self.beta.weight, self.beta.bias))
            scores = F.linear(x.float(), self.score.weight, self.score.bias) if self.score_gate else None
            if scores is not None:
                beta = beta*2*torch.sigmoid(scores)
        keys = self.keys[None].expand(b,base.H,base.M,base.K).contiguous()
        values = torch.zeros(b,base.H,base.M,base.K,device=x.device)
        initial = self.initial_score[None].expand(b,base.H,base.M) if self.score_gate else None
        o, _ = block_komp_flex(q,k,v,g,beta,(keys,values),chunk_size=self.chunk,evict=self.evict,
                              scores=scores,initial_scores=initial,project=self.project,
                              coherence=None,dedupe=None,projectron=False)
        y = self.norm(o.float()).to(x.dtype).reshape(b,t,base.D)
        return self.out(y*F.silu(self.gate(x)))


def rotary(x):
    """Adjacent coordinate pairs; base 10000, full head dim, positions start at zero."""
    t, d = x.shape[-2:]
    angles = torch.arange(t,device=x.device,dtype=torch.float32)[:,None] * (
        10000. ** (-torch.arange(0,d,2,device=x.device,dtype=torch.float32)/d))[None,:]
    z = x.float().reshape(*x.shape[:-1],d//2,2)
    a, b = z.unbind(-1)
    return torch.stack((a*angles.cos()-b*angles.sin(),a*angles.sin()+b*angles.cos()),-1).flatten(-2).to(x.dtype)


def window_mask(t, window, device):
    pos = torch.arange(t,device=device)
    distance = pos[:,None]-pos[None,:]
    return (distance>=0)&(distance<window)


class SWAMixer(nn.Module):
    def __init__(self, original, cfg):
        super().__init__()
        # Reuse the exact initialized modules, with no extra random draw or unused KOMP parameters.
        for name in ('q','k','v','out','gate','norm'):
            setattr(self,name,getattr(original,name))
        self.window, self.rope = cfg['window'], cfg['rope']

    def forward(self, x):
        b,t,_ = x.shape
        q,k,v = [f(x).reshape(b,t,base.H,base.K).transpose(1,2) for f in (self.q,self.k,self.v)]
        if self.rope:
            q,k = rotary(q),rotary(k)
        o = F.scaled_dot_product_attention(q,k,v,attn_mask=window_mask(t,self.window,x.device),
                                            dropout_p=0.,is_causal=False,scale=base.K**-.5)
        y = self.norm(o.transpose(1,2).float()).to(x.dtype).reshape(b,t,base.D)
        return self.out(y*F.silu(self.gate(x)))


def shared_hashes(model):
    return {n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
            for n,p in model.named_parameters()
            if not any(part in n.split('.') for part in ('decay','beta','keys','score','initial_score'))}


class AblationLM(base.LM):
    def __init__(self, kind, seed):
        super().__init__('komp',seed)
        before = shared_hashes(self)
        self.variant = kind
        cfg = VARIANTS[kind]
        for layer in self.layers:
            layer.mix = SWAMixer(layer.mix,cfg) if 'window' in cfg else ProjectionMixer(layer.mix,cfg)
        assert shared_hashes(self)==before, 'Shared initialization changed'
        self.shared_initial = before


def source_hashes():
    root = Path(base.__file__).parents[1]
    files = [Path(__file__),Path(base.__file__),Path(be.__file__),*sorted((root/'komp_slot').glob('*.py'))]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def metadata(model):
    cfg = VARIANTS[model.variant]
    swa = 'window' in cfg
    return dict(ablation_config=cfg,ablation_source_sha256=source_hashes(),
        shared_init_sha256=model.shared_initial,
        implementation='PyTorch SDPA BF16 autocast' if swa else 'frozen block_flex FP32 algebra; BF16 projected q/k/v',
        routing='causal window, current token included' if swa else 'block-synchronous '+cfg.get('evict','fifo'),
        rope=dict(base=10000.,dim=base.K,pairing='adjacent',positions='zero based absolute',qk_only=True) if cfg.get('rope') else None,
        attention_scale=base.K**-.5 if swa else None, short_conv=False,
        persistent_state_elements_per_head=(cfg['window'] if swa else base.M)*2*base.K,
        state_budget_note='element count, not byte parity: SWA BF16 KV versus Block FP32 state; transient Block dictionary M+C',
        temperature=None if swa else .5,ridge=None if swa else 1e-4,
        deterministic=True,tf32=False)


@torch.no_grad()
def causal_check(model,ids):
    records=[]
    with base.amp():
        before=model(ids)
        repeated=model(ids)
        torch.testing.assert_close(before,repeated,atol=0,rtol=0)
        for cut in (31,32,33,63,64,65,95,96,97,127,128,129):
            changed=ids.clone();changed[:,cut:]=(changed[:,cut:]+7)%base.VOCAB
            after=model(changed)
            torch.testing.assert_close(before[:,:cut],after[:,:cut],atol=1e-5,rtol=1e-5)
            records.append(dict(cut=cut,max_abs=float((before[:,:cut]-after[:,:cut]).abs().max())))
    return records


@torch.no_grad()
def window_check(model):
    mix=model.layers[0].mix
    if not isinstance(mix,SWAMixer):return None
    t=mix.window+9
    x=torch.randn(2,t,base.D,device='cuda')
    target=t-1
    with base.amp():
        a=mix(x)
        changed=x.clone();changed[:,:target-mix.window+1]+=torch.randn_like(changed[:,:target-mix.window+1])*3
        b=mix(changed)
        torch.testing.assert_close(a[:,target],b[:,target],atol=1e-5,rtol=1e-5)
        # Verify the oldest *included* position can influence the current output.
        changed=x.clone();changed[:,target-mix.window+1]+=3*torch.randn_like(changed[:,target-mix.window+1])
        c=mix(changed)
        inside=float((a[:,target]-c[:,target]).abs().max());assert inside>0
    return dict(window=mix.window,target=target,outside_max_abs=float((a[:,target]-b[:,target]).abs().max()),
                oldest_included_max_abs=inside,scope='single mixer; stacked model has a larger receptive field')


def preflight():
    ROOT.mkdir(parents=True,exist_ok=True)
    sample=base.load('train')[:4];ids,_=base.batch(sample)
    report=dict(variants={},source_sha256=source_hashes())
    common=None
    for kind in VARIANTS:
        model=AblationLM(kind,731).cuda()
        if common is None:common=model.shared_initial
        assert common==model.shared_initial
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
        initial=causal_check(model,ids);window_initial=window_check(model)
        losses=[];score_gradients=[]
        for step in range(16):
            opt.zero_grad(set_to_none=True)
            with base.amp():loss=base.loss_items(model,sample)[0].mean()
            loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert bool(torch.isfinite(norm))
            if step==0 and VARIANTS[kind].get('evict')=='learned':
                for layer in model.layers:
                    grad=layer.mix.score.weight.grad
                    assert grad is not None and bool(torch.isfinite(grad).all()) and float(grad.norm())>0
                    score_gradients.append(float(grad.norm()))
            opt.step();losses.append(float(loss.detach()))
        assert losses[-1]<losses[0],(kind,losses)
        report['variants'][kind]=dict(losses=losses,initial_causality=initial,trained_causality=causal_check(model,ids),
            window_initial=window_initial,window_trained=window_check(model),score_gradient_norms=score_gradients,
            params=sum(p.numel() for p in model.parameters()),shared_init_sha256=common)
        print('PASS PREFLIGHT',kind,losses[0],losses[-1],flush=True)
        del model,opt;torch.cuda.empty_cache()
    report['status']='passed'
    (ROOT/'preflight.json').write_text(json.dumps(report,indent=2))


def train(kind,seed):
    ticket=json.loads((ROOT/'preflight.json').read_text())
    assert ticket['status']=='passed' and ticket['source_sha256']==source_hashes() and kind in ticket['variants']
    eval_times=[];gen_times=[]
    original_eval,original_gen=base.evaluate,base.generate
    def timed(fn,records):
        def call(*args,**kwargs):
            torch.cuda.synchronize();start=time.monotonic()
            result=fn(*args,**kwargs)
            torch.cuda.synchronize();records.append(time.monotonic()-start)
            return result
        return call
    base.evaluate=timed(original_eval,eval_times);base.generate=timed(original_gen,gen_times)
    try:
        base.train(kind,seed,model_factory=AblationLM,result_root=ROOT/'results',config_extra=metadata,preflight_path=ROOT/'preflight.json')
    finally:
        base.evaluate,base.generate=original_eval,original_gen
    dest=ROOT/'results'/f'{kind}-seed{seed}'
    summary=json.loads((dest/'summary.json').read_text())
    summary['timing']=dict(train_loop_and_bookkeeping_seconds=summary['elapsed_seconds']-sum(eval_times[:-1])-sum(gen_times),
        evaluation_seconds=sum(eval_times[:-1]),generation_seconds=sum(gen_times),reload_evaluation_seconds=eval_times[-1],
        note='CUDA synchronized wall time; training excludes measured validation/evaluation/generation but includes loop/logging overhead; not kernel timing')
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--kind',choices=VARIANTS)
    p.add_argument('--seed',type=int,default=731);args=p.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.preflight:preflight()
    else:train(args.kind,args.seed)
