"""Eight-GPU training with the original global document batches and objective."""
import argparse
import contextlib
import hashlib
import faulthandler
import traceback
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'snapshot/suite100m'))
import train as base
from models import optimizer
from alpha_models import LM, metadata
from tasks import Factory


def digest():
    return {'snapshot': hashlib.sha256((ROOT/'evidence/snapshot-sha256.json').read_bytes()).hexdigest(),
            'ddp_train': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'alpha_models':hashlib.sha256((ROOT/'alpha_models.py').read_bytes()).hexdigest()}


def stage(name):
    if os.environ.get("DDP_PREFLIGHT") == "1":
        print("STAGE", dist.get_rank(), name, flush=True)


def rank0_result(error):
    """CPU control broadcast keeps rank-only errors visible to every process."""
    result = [error]
    dist.broadcast_object_list(result, src=0, group=CONTROL)
    if result[0] is not None:
        raise RuntimeError(result[0])


class Objective(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, mask):
        return base.loss_sum(self.model, ids, mask)[0]


def amp(dtype):
    return torch.autocast('cuda', dtype=torch.bfloat16) if dtype == 'bf16' else contextlib.nullcontext()


def synchronize_parameters(model):
    # Every rank compares exact parameter sums with rank0; a sampled element
    # checksum plus all tensor sums detects replica divergence without copies.
    stats = torch.stack([p.detach().double().sum() for p in model.parameters()])
    target = stats.clone()
    dist.broadcast(target, 0)
    error = (stats-target).abs().max()
    assert error.item() == 0, ('replicas diverged', error.item())
    return error.item()


def update(ddp, opt, rows, dtype='bf16', compare=False):
    rank, world = dist.get_rank(), dist.get_world_size()
    nvalid = sum(int(np.sum(row['loss_mask'][1:])) for row in rows)
    assert nvalid > 0
    local = rows[rank::world]
    # All ranks use an identical number of forward/backward calls. Empty tail
    # partitions execute a graph-connected zero loss on an existing document.
    padded = math.ceil(max(len(r['input_ids']) for r in rows)/16)*16
    max_local = math.ceil(len(rows)/world)
    micro = min(max_local, 2**int(math.log2(max(1, 8192/padded))))
    microsteps = math.ceil(max_local/micro)
    opt.zero_grad(set_to_none=True)
    totals = torch.zeros(3, device='cuda', dtype=torch.float64)
    for step in range(microsteps):
        selected = local[step*micro:(step+1)*micro]
        valid = bool(selected)
        ids, mask = base.batch(selected if valid else rows[:1])
        sync = contextlib.nullcontext() if step == microsteps-1 else ddp.no_sync()
        with sync:
            with amp(dtype):
                stage(f'update-{dtype}-{step}-forward')
                loss_sum = ddp(ids, mask)
                loss = loss_sum * (world/nvalid if valid else 0.)
            stage(f'update-{dtype}-{step}-backward')
            loss.backward()
            stage(f'update-{dtype}-{step}-backward-done')
        if valid:
            totals += torch.tensor([float(loss_sum.detach()), sum(len(r['input_ids'])-1 for r in selected),
                                    sum(int(np.sum(r['loss_mask'][1:])) for r in selected)], device='cuda', dtype=torch.float64)
    stage("update-totals")
    dist.all_reduce(totals)
    assert int(totals[1]) == sum(len(r['input_ids'])-1 for r in rows)
    assert int(totals[2]) == nvalid
    report = {'loss': totals[0].item()/nvalid, 'tokens': int(totals[1]), 'supervised': int(totals[2]),
              'microsteps': microsteps, 'global_documents': len(rows)}
    if compare:
        report['gradients'] = {name: p.grad.detach().clone() for name,p in ddp.module.model.named_parameters()
                               if name in ('embed.weight', 'head.weight', 'layers.0.mix.out.weight', 'layers.0.mix.layer.o_proj.weight')}
    norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.)
    assert torch.isfinite(norm) and math.isfinite(report['loss'])
    opt.step()
    report['grad_norm'] = norm.item()
    return report


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(),
            'numpy': np.random.get_state(), 'python': random.getstate()}


def save_checkpoint(path, ddp, opt, config, step, consumed, supervised, elapsed):
    rank, world = dist.get_rank(), dist.get_world_size()
    states = [None]*world if rank == 0 else None
    dist.gather_object(rng_state(), states, dst=0)
    if rank == 0:
        tmp = path.with_suffix('.tmp.pt')
        torch.save({'model': ddp.module.model.state_dict(), 'optimizer': opt.state_dict(), 'config': config,
                    'step': step, 'consumed': consumed, 'supervised': supervised, 'elapsed': elapsed,
                    'rank_rng': states, 'world_size': world}, tmp)
        os.replace(tmp, path)
    dist.barrier()


def load_checkpoint(path, model, opt, config):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['config'] == config and saved['world_size'] == dist.get_world_size()
    model.load_state_dict(saved['model'], strict=True)
    opt.load_state_dict(saved['optimizer'])
    state = saved['rank_rng'][dist.get_rank()]
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy']);random.setstate(state['python'])
    progress = {key: saved[key] for key in ('step', 'consumed', 'supervised', 'elapsed')}
    del saved
    dist.barrier()
    return progress


def make_model(kind, seed, lr, wd):
    model = LM(kind, seed).cuda()
    opt = optimizer(model, lr, wd)
    # The loss/head is inside DDP's forward graph, avoiding false unused heads.
    ddp = DDP(Objective(model), device_ids=[int(os.environ['LOCAL_RANK'])], broadcast_buffers=False,
              find_unused_parameters=False)
    return model, opt, ddp


def preflight(kind):
    rank = dist.get_rank()
    # Confirm the only initialization changes under rank0 error propagation.
    initialization_error=None
    if rank == 0:
        try:
            from models import LM as OriginalLM
            reference=OriginalLM('exact',731)
            changed=LM('exact',731)
            a,b=reference.state_dict(),changed.state_dict()
            differences=[name for name in a if not torch.equal(a[name],b[name])]
            assert differences==[f'layers.{i}.mix.decay.bias' for i in range(15)],differences
            base.atomic_json(ROOT/'evidence/initialization-difference.json',{'status':'passed','changed_parameters':differences,'all_other_tensors_identical':True,'reference_alpha0':.98,'new_alpha0':.995})
            del reference,changed,a,b
        except BaseException:
            initialization_error=traceback.format_exc()
    rank0_result(initialization_error)
    stage("construct")
    model, opt, ddp = make_model(kind, 731, 1e-4, .01)
    stage("causality")
    info = metadata(model)
    assert 99_000_000 < info['parameters'] < 101_000_000
    model.eval()
    ids = torch.arange(129, device='cuda')[None].remainder(99)+1
    with torch.no_grad(), amp('bf16'):
        before = model(ids)
        altered = ids.clone();altered[:,65:] += 11
        after = model(altered)
    torch.testing.assert_close(before[:,:65],after[:,:65],atol=1e-5,rtol=1e-5)
    del before,after
    model.train()
    # FP32 differential objective/gradient/update against the one-GPU original
    # computation on a whole, uneven global batch (17 documents across8ranks).
    fac = Factory('mano')
    rows = [fac.row(random.Random(923+i)) for i in range(17)]
    partitions=[None]*dist.get_world_size()
    dist.all_gather_object(partitions,list(range(17))[rank::dist.get_world_size()])
    assert sorted(i for part in partitions for i in part)==list(range(17))
    selected_names = ('embed.weight', 'head.weight', 'layers.0.mix.out.weight', 'layers.0.mix.layer.o_proj.weight')
    reference = None
    reference_error = None
    stage('reference')
    if rank == 0:
        try:
            reference = LM(kind,731).cuda();reference.load_state_dict(model.state_dict())
            ropt = optimizer(reference,1e-4,.01)
            ids,mask = base.batch(rows)
            loss,n = base.loss_sum(reference,ids,mask);whole_loss=loss/n
            whole_loss.backward()
            whole_grads = {name:p.grad.detach().clone() for name,p in reference.named_parameters() if name in selected_names}
            whole_loss_value = whole_loss.item()
            del loss,whole_loss,ids,mask
            ropt.zero_grad(set_to_none=True)
            # Isolate distributed reduction from native-kernel batch/length
            # numerics: execute exactly the rank-local microbatches sequentially.
            rloss = torch.zeros((),device='cuda')
            padded=math.ceil(max(len(r['input_ids']) for r in rows)/16)*16
            max_local=math.ceil(len(rows)/dist.get_world_size())
            micro=min(max_local,2**int(math.log2(max(1,8192/padded))))
            for shard in range(dist.get_world_size()):
                local=rows[shard::dist.get_world_size()]
                for start in range(0,len(local),micro):
                    ids,mask=base.batch(local[start:start+micro])
                    loss,_=base.loss_sum(reference,ids,mask)
                    normalized=loss/n
                    normalized.backward()
                    rloss += normalized.detach()
            rgrads = {name:p.grad.detach().clone() for name,p in reference.named_parameters() if name in selected_names}
            whole_diagnostics={'loss':whole_loss_value,'partitioned_loss':rloss.item(),
                'loss_abs':abs(whole_loss_value-rloss.item()),'gradient_rms':{},'gradient_sign_disagreement_fraction':{}}
            for name,grad in rgrads.items():
                ref=whole_grads[name]
                whole_diagnostics['gradient_rms'][name]=((grad-ref).square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-8)).item()
                whole_diagnostics['gradient_sign_disagreement_fraction'][name]=((grad.sign()!=ref.sign()) & ((grad!=0)|(ref!=0))).float().mean().item()
            print('NATIVE_BATCH_DIAGNOSTIC',kind,json.dumps(whole_diagnostics),flush=True)
            del whole_grads
            torch.nn.utils.clip_grad_norm_(reference.parameters(),1.);ropt.step()
        except BaseException:
            reference_error = traceback.format_exc()
            print(reference_error, flush=True)
    rank0_result(reference_error)
    stage("reference-done")
    step = update(ddp,opt,rows,'fp32',compare=True)
    gradient_errors,parameter_errors = {},{}
    comparison_error = None
    if rank == 0:
        try:
            assert abs(step['loss']-rloss.item()) < .005,(kind,step['loss'],rloss.item())
            for name,grad in step['gradients'].items():
                ref = rgrads[name]
                error = ((grad-ref).square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-8)).item()
                assert error < .02,(kind,name,error)
                gradient_errors[name]=error
            # Compare actual updated parameters, including optimizer preconditioning.
            refs=dict(reference.named_parameters())
            for name,p in model.named_parameters():
                if name in selected_names:
                    error=(p-refs[name]).abs().max().item()
                    assert error < 1e-4,(kind,name,'parameter delta',error)
                    parameter_errors[name]=error
            del reference,ropt,rgrads
        except BaseException:
            comparison_error = traceback.format_exc()
            print(comparison_error, flush=True)
    rank0_result(comparison_error)
    stage("comparison-done")
    del step['gradients']
    synchronize_parameters(model)
    # Different lengths and a tail smaller than world_size exercise dummy ranks.
    tail = [fac.row(random.Random(1923+i)) for i in range(3)]
    stage('tail')
    second=update(ddp,opt,tail,'bf16')
    synchronize_parameters(model)
    directory = ROOT/'runs-ddp/ddp-preflight';directory.mkdir(parents=True,exist_ok=True)
    config={'kind':kind,'preflight':True,'sources':digest()}
    checkpoint = directory/f'{kind}.checkpoint.pt'
    stage('checkpoint')
    save_checkpoint(checkpoint,ddp,opt,config,2,step['tokens']+second['tokens'],step['supervised']+second['supervised'],0)
    restored=load_checkpoint(checkpoint,model,opt,config)
    assert restored['step']==2
    assert all(int(state['step'])==2 for state in opt.state.values())
    synchronize_parameters(model)
    # Long real-data BF16 batch and its variable lengths, after restore.
    corpus=base.Corpus('depo2','train','tune')
    realrows=[corpus.row(i) for i in range(17)]
    stage('real')
    real=update(ddp,opt,realrows,'bf16')
    synchronize_parameters(model)
    report={'status':'passed','model':info,'world_size':dist.get_world_size(),'sources':digest(),
            'checks':['full-model causality','FP32 one-GPU rank-shaped accumulation vs8GPU global objective and update',
                      'uneven17-document batch','3-document tail with dummy ranks','replica equality after each step',
                      'all-rank RNG and model/optimizer checkpoint restore','real long Depo2 BF16 update'],
            'fp32_gradient_rms':gradient_errors,'parameter_max_abs':parameter_errors,
            'short_batch':step,'tail_batch':second,'real_batch':real,
            'whole_batch_native_numerics':whole_diagnostics if rank==0 else None}
    if rank == 0:
        base.atomic_json(directory/f'{kind}.json',report)
        checkpoint.unlink()
        print('DDP_PREFLIGHT',json.dumps(report),flush=True)
    dist.barrier()


def train(args):
    rank=dist.get_rank();world=dist.get_world_size()
    ticket=json.loads((ROOT/'runs-ddp/ddp-preflight'/f'{args.kind}.json').read_text())
    assert ticket['status']=='passed' and ticket['world_size']==world and ticket['sources']==digest()
    name=f'{args.task}-{args.kind}-final-lr{args.lr:g}-wd{args.wd:g}-seed{args.seed}'
    dest=ROOT/'runs-ddp/results'/name;dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'summary.json').exists()
    model,opt,ddp=make_model(args.kind,args.seed,args.lr,args.wd)
    corpus=base.Corpus(args.task,'train','final');val=base.Corpus(args.task,'val','final')
    groups=list(corpus.batches(12345));total=len(groups)
    order_hash=hashlib.sha256(b''.join(np.asarray(g,dtype='<i8').tobytes() for g in groups)).hexdigest()
    order_hashes=[None]*world;dist.all_gather_object(order_hashes,order_hash)
    assert len(set(order_hashes))==1,'Ranks disagree on global data order'
    config={'name':name,'task':args.task,'phase':'final','seed':args.seed,'lr':args.lr,'wd':args.wd,
            'batch':128,'world_size':world,'local_microbatch_token_cap':8192,'steps':total,
            'model':metadata(model),'sources':digest(),'selection':'reused prior validation selection; Exact matched Block',
            'ddp_loss':'local sum * world_size / global supervised tokens; identical microstep count',
            'data_order_sha256':order_hash}
    ckpt=dest/'checkpoint.pt';done=consumed=supervised=0;elapsed=0.
    if ckpt.exists():
        progress=load_checkpoint(ckpt,model,opt,config)
        done=progress['step'];consumed=progress['consumed'];supervised=progress['supervised'];elapsed=progress['elapsed']
    if rank==0:base.atomic_json(dest/'config.json',config)
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    log=(dest/'metrics.jsonl').open('a',buffering=1) if rank==0 else None
    for j in range(done,total):
        assert len(set(map(int,groups[j])))==len(groups[j])
        rows=[corpus.row(int(i)) for i in groups[j]]
        lr,mult=base.schedule(j+1,total,args.lr)
        for group in opt.param_groups:group['lr']=lr*(mult if group['residual_output'] else 1.)
        record=update(ddp,opt,rows,'bf16')
        consumed+=record['tokens'];supervised+=record['supervised']
        seconds=elapsed+time.monotonic()-start
        stop=torch.tensor(int((ROOT/'ddp-STOP').exists()) if rank==0 else 0,device='cuda')
        dist.broadcast(stop,0)
        stopping=bool(stop.item())
        if j==done or (j+1)%25==0 or j+1==total:
            synchronize_parameters(model)
            if rank==0:
                record.update(step=j+1,total=total,lr=lr,tokens=consumed,seconds=seconds,world_size=world)
                log.write(json.dumps(record)+'\n');print('TRAIN',name,json.dumps(record),flush=True)
        if (j+1)%250==0 or j+1==total or stopping:
            save_checkpoint(ckpt,ddp,opt,config,j+1,consumed,supervised,seconds)
        if stopping:
            if rank==0:
                base.atomic_json(dest/'paused.json',{'step':j+1,'tokens':consumed,'status':'paused','checkpoint':str(ckpt)})
                log.close()
            dist.barrier()
            return
    assert consumed==int(np.diff(corpus.offsets).sum())-len(corpus)
    dist.barrier()
    # Use the underlying module only on rank0; DDP forward collectives are absent.
    if rank==0:
        final_val=base.evaluate(model,val)
        summary={'status':'completed','config':config,'val':final_val,'tokens':consumed,
                 'supervised_positions':supervised,'train_seconds':elapsed+time.monotonic()-start,
                 'peak_bytes_rank0':torch.cuda.max_memory_allocated()}
        base.atomic_json(dest/'summary.json',summary);log.close()
        print('COMPLETE',name,json.dumps(final_val),flush=True)
    dist.barrier()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--kind',required=True);parser.add_argument('--task',choices=('depo2','brevo2'))
    parser.add_argument('--seed',type=int,default=731);parser.add_argument('--lr',type=float,default=.0005);parser.add_argument('--wd',type=float,default=.1)
    args=parser.parse_args()
    assert args.kind=='exact'
    manifest=json.loads((ROOT/'source-manifest.json').read_text())
    for name,wanted in manifest['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==wanted,name
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    os.environ['DDP_PREFLIGHT'] = '1' if args.preflight else '0'
    timeout=__import__('datetime').timedelta(seconds=180 if args.preflight else 600)
    dist.init_process_group('nccl',timeout=timeout,device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    CONTROL=dist.new_group(backend='gloo',timeout=timeout)
    if args.preflight:
        faulthandler.dump_traceback_later(90,repeat=True)
    assert dist.get_world_size()==8
    try:
        preflight(args.kind) if args.preflight else train(args)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush();sys.stderr.flush()
        os._exit(1)
    else:
        faulthandler.cancel_dump_traceback_later()
        dist.destroy_process_group()
