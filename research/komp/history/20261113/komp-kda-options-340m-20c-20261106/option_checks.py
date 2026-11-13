"""Option independence, backward compatibility and changed-gate operator checks."""
import gc,hashlib,itertools,json,math,traceback
import torch
from torch import nn
from torch.nn import functional as F
from models import LM,map_weights,ROOT,KOMPConfig,KOMPMixer,legacy,D,H,K,M,ALPHA,BETA,exact_komp,block_komp_fast
from mixer_options import head_log_decays,initialize_mixer_options
from komp_slot.block_reference import block_komp_reference,block_komp_token_reference


def report_diff(a,b):
    a,b=a.detach().double(),b.detach().double();d=a-b
    return {'relative_rms':float(d.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-6)),
            'normalized_max':float(d.abs().max()/b.abs().max().clamp_min(1.))}


def differential(kind,inputs,label,checks):
    inputs=[a.detach().requires_grad_() for a in inputs]
    gold=[a.detach().double().requires_grad_() for a in inputs]
    if kind=='exact':
        o,state,_=exact_komp(*inputs[:5],tuple(inputs[5:]),compute_chunk=16,projection_dtype='float32')
        ref,refstate=block_komp_token_reference(*gold[:5],tuple(gold[5:]),chunk_size=1,evict='fifo')
    else:
        o,state=block_komp_fast(*inputs[:5],tuple(inputs[5:]),chunk_size=128)
        ref,refstate=block_komp_reference(*gold[:5],tuple(gold[5:]),chunk_size=128)
    weight=[torch.randn_like(x.float()) for x in (o,*state)]
    grads=torch.autograd.grad((o,*state),inputs,weight)
    refgrads=torch.autograd.grad((ref,*refstate),gold,[x.double() for x in weight])
    low=inputs[0].dtype==torch.bfloat16
    for name,x,y in zip(['output','keys','state','dq','dk','dv','dg','dbeta','dk0','ds0'],(o,*state,*grads),(ref,*refstate,*refgrads)):
        row={'case':label,'name':name,**report_diff(x,y)};checks.append(row)
        assert torch.isfinite(x).all() and row['relative_rms']<=(.035 if low else .005) and row['normalized_max']<=(.04 if low else .005),row


def run_checks(kind,stream):
    checks=[];option_rows=[]
    try:
        torch.manual_seed(42)
        # Full CPU checkpoint mapping verifies unchanged shared initialization, not just mixer math.
        old=legacy.LM(kind,seed=42,checkpoint_layers=False)
        off=LM(kind,options={'output_sigmoid':False,'low_rank_gates':False,'multi_timescale_init':False})
        expected=map_weights(old.state_dict());actual=off.state_dict()
        assert set(expected)==set(actual)
        for name,value in expected.items():assert torch.equal(value,actual[name]),('baseline_initialization',name)
        del old,off,expected,actual;gc.collect()
        # Saved loader cursor: diagnostics do not consume formal training inputs.
        state=stream.state_dict();ids=stream.next()[:129].cuda();stream.load_state_dict(state)
        torch.manual_seed(42+int(hashlib.sha256(b'embed').hexdigest()[:8],16))
        embed=nn.Embedding(32000,D);nn.init.normal_(embed.weight,0,.02)
        with torch.no_grad():x=F.rms_norm(embed.cuda()(ids)[None],(D,),eps=1e-6)
        del embed
        for sigmoid,lowrank,multi in itertools.product([False,True],repeat=3):
            config=KOMPConfig(kind=kind,checkpoint_mixer=False,output_sigmoid=sigmoid,low_rank_gates=lowrank,multi_timescale_init=multi)
            torch.manual_seed(142);mix=KOMPMixer(config,0)
            # Initial short-conv weights as in seed42 shared backbone.
            for j,conv in enumerate(mix.convs):
                torch.manual_seed(42+6000+j);nn.init.normal_(conv.weight,0,.02)
            initialize_mixer_options(mix,config,42,0);mix=mix.cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                o,_,_=mix(x[:,:33]);loss=o.float().square().mean()
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in mix.parameters())
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                args=mix.operands(x);g=args[3];beta=args[4]
                expected=head_log_decays().to(g) if multi else torch.full((H,),math.log(ALPHA),device='cuda')
                assert torch.allclose(g,expected[None,None,:,None].expand_as(g),atol=1e-7,rtol=1e-5)
                assert abs(float(g.double().mean())-math.log(ALPHA))<1e-7
                assert torch.allclose(beta,torch.full_like(beta,BETA),atol=1e-7,rtol=1e-5)
                if not any((sigmoid,lowrank,multi)):
                    baseline=legacy.Mixer(kind).cuda();baseline.load_state_dict(mix.state_dict())
                    base=baseline(x[:,:33]);assert torch.equal(base,o),'Disabled options changed the legacy mixer'
                    del baseline
            option_rows.append(dict(sigmoid=sigmoid,lowrank=lowrank,multi_timescale=multi,finite_gradients=True,initial_mean_log_alpha=float(g.double().mean())))
            if all((sigmoid,lowrank,multi)):
                # First and last head cover the slowest and fastest initial decay.
                real=[a[:,:,[0,H-1]].contiguous() for a in args[:5]]+[a[:,[0,H-1]].contiguous() for a in args[5:]]
                differential(kind,real,'real_text_initial_activations_multiscale',checks)
            del mix,args,o,loss;gc.collect();torch.cuda.empty_cache()
        for dtype in [torch.float32,torch.bfloat16]:
            torch.manual_seed(731)
            q,k,v=[torch.randn(1,129,2,K,device='cuda',dtype=dtype) for _ in range(3)]
            g=head_log_decays()[[0,H-1]].float().cuda()[None,None,:,None].expand(1,129,2,M).contiguous()
            # Nonconstant token/slot perturbations exercise gates after they start learning.
            g=g*(.5+torch.rand_like(g))
            beta=torch.full((1,129,2),BETA,device='cuda')
            k0=F.normalize(torch.randn(1,2,M,K,device='cuda'),dim=-1);s0=torch.randn_like(k0)*.1
            differential(kind,[q,k,v,g,beta,k0,s0],str(dtype)+'_varying_slot_decay',checks)
        (ROOT/f'evidence/options-operator-{kind}.json').write_text(json.dumps(dict(status='passed',kind=kind,options=option_rows,checks=checks,scope='Specified real-text and random cases only; existing near-collinear FP32 limitations remain'),indent=2))
    except BaseException:
        (ROOT/f'evidence/options-operator-failure-{kind}.json').write_text(json.dumps(dict(status='failed',kind=kind,options=option_rows,checks=checks,traceback=traceback.format_exc()),indent=2));raise
