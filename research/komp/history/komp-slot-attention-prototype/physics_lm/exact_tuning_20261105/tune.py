"""Bounded Exact-KOMP screen; shares accepted DDP objective and frozen backbone."""
import argparse
import datetime
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent / 'suite100m_exact_20261105'
if not SUITE.exists():
    SUITE = HERE.parent
sys.path.insert(0, str(SUITE))
import ddp_train as d
import models
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ORIGINAL_EXACT = models.exact_komp


def sources():
    return dict(ddp=d.digest(), tuning={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(HERE.glob('*.py'))},
                plan=hashlib.sha256((HERE/'initial-screen.json').read_bytes()).hexdigest())


def configuration(name):
    plan = json.loads((HERE/'initial-screen.json').read_text())
    return next(c for c in plan['screen'] if c['name'] == name)


def configure(candidate):
    # Scoped to this process; frozen baseline sources remain byte-identical.
    def tuned_operator(*args, **kwargs):
        kwargs.update(temperature=candidate['temperature'], ridge=candidate['ridge'],
                      projection_dtype=candidate['projection_dtype'])
        return ORIGINAL_EXACT(*args, **kwargs)
    models.exact_komp = tuned_operator
    model = models.LM('exact', 731).cuda()
    with torch.no_grad():
        for layer in model.layers:
            mix = layer.mix
            mix.decay.bias.fill_(candidate['decay_bias'])
            mix.beta.bias.fill_(candidate['beta_bias'])
            assert torch.count_nonzero(mix.decay.weight) == 0
            assert torch.count_nonzero(mix.beta.weight) == 0
            alpha = torch.nn.functional.logsigmoid(mix.decay.bias).div(8).exp()
            torch.testing.assert_close(alpha, torch.full_like(alpha, candidate['alpha0']))
            torch.testing.assert_close(mix.beta.bias.sigmoid(), torch.full_like(mix.beta.bias, candidate['beta0']))
    return model


def evaluate(model, corpus):
    """Identical masked CE over the full validation set, distributed by document."""
    model.eval()
    indices = list(range(dist.get_rank(), len(corpus), dist.get_world_size()))
    totals = torch.zeros(2, device='cuda', dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, len(indices), 4):
            ids, mask = d.base.batch([corpus.row(i) for i in indices[start:start+4]])
            with d.amp('bf16'):
                loss, n = d.base.loss_sum(model, ids, mask)
            totals += torch.tensor([float(loss), n], device='cuda', dtype=torch.float64)
    dist.all_reduce(totals)
    model.train()
    return {'ce': float(totals[0]/totals[1]), 'supervised_positions': int(totals[1])}


def train(args):
    rank, world = dist.get_rank(), dist.get_world_size()
    source = sources()
    ticket = json.loads((SUITE/'runs-ddp/ddp-preflight/exact.json').read_text())
    assert ticket['status'] == 'passed' and ticket['world_size'] == world and ticket['sources'] == d.digest()
    numerical = json.loads((HERE/'evidence'/f'numerical-{args.task}.json').read_text())
    assert numerical['status'] == 'passed' and numerical['sources'] == source
    tuned_ticket = json.loads((HERE/'preflight'/args.task/'runs-ddp/ddp-preflight/exact.json').read_text())
    assert tuned_ticket['status'] == 'passed' and tuned_ticket['sources'] == source
    candidate = configuration(args.candidate)
    dest = HERE/'results'/args.task/args.candidate
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest/'summary.json').exists(), 'Never rerun a completed candidate'
    lr = {'depo2': .0005, 'brevo2': .001}[args.task]
    model = configure(candidate)
    opt = models.optimizer(model, lr, .1)
    ddp = DDP(d.Objective(model), device_ids=[int(os.environ['LOCAL_RANK'])],
              broadcast_buffers=False, find_unused_parameters=False)
    corpus = d.base.Corpus(args.task, 'train', 'tune')
    val = d.base.Corpus(args.task, 'val', 'tune')
    groups = list(corpus.batches(12345))
    total = len(groups)
    order_hash = hashlib.sha256(b''.join(np.asarray(g, dtype='<i8').tobytes() for g in groups)).hexdigest()
    hashes = [None]*world
    dist.all_gather_object(hashes, order_hash)
    assert len(set(hashes)) == 1
    expected = int(np.diff(corpus.offsets).sum())-len(corpus)
    assert 10_000_000 <= expected < 10_002_000, expected
    config = {'task': args.task, 'candidate': candidate, 'phase': 'tune', 'seed': 731,
              'lr': lr, 'wd': .1, 'batch': 128, 'world_size': world, 'steps': total,
              'model': {**models.metadata(model), 'projection_dtype': candidate['projection_dtype']}, 'sources': source, 'data_order_sha256': order_hash,
              'train_positions': expected, 'selection': 'full validation masked CE only; short-budget ranking',
              'schedule': '5% warmup then cosine to 10%; independent 10M horizon',
              'ddp_loss': 'local sum * world_size / global supervised tokens'}
    checkpoint = dest/'checkpoint.pt'
    done = consumed = supervised = 0
    elapsed = 0.
    if checkpoint.exists():
        saved = d.load_checkpoint(checkpoint, model, opt, config)
        done, consumed, supervised, elapsed = (saved[k] for k in ('step','consumed','supervised','elapsed'))
    if rank == 0:
        d.base.atomic_json(dest/'config.json', config)
    log = (dest/'metrics.jsonl').open('a', buffering=1) if rank == 0 else None
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    # Five comparable validation checkpoints, with no heldout/test selection.
    milestones = {max(1, math.ceil(total*x/5)) for x in range(1,6)}
    final_val = None
    for j in range(done, total):
        rate, mult = d.base.schedule(j+1, total, lr)
        for group in opt.param_groups:
            group['lr'] = rate*(mult if group['residual_output'] else 1.)
        record = d.update(ddp, opt, [corpus.row(int(i)) for i in groups[j]], 'bf16')
        consumed += record['tokens']
        supervised += record['supervised']
        stop = torch.tensor(int((SUITE/'ddp-STOP').exists()) if rank == 0 else 0, device='cuda')
        dist.broadcast(stop, 0)
        if j+1 in milestones:
            final_val = evaluate(model, val)
            record['val'] = final_val
        seconds = elapsed + time.monotonic()-start
        if j == done or (j+1)%25 == 0 or j+1 in milestones or stop.item():
            d.synchronize_parameters(model)
            if rank == 0:
                record.update(step=j+1,total=total,lr=rate,tokens=consumed,seconds=seconds)
                log.write(json.dumps(record)+'\n')
                print('TUNE',args.task,args.candidate,json.dumps(record),flush=True)
        if (j+1)%100 == 0 or j+1 in milestones or stop.item():
            d.save_checkpoint(checkpoint,ddp,opt,config,j+1,consumed,supervised,seconds)
        if stop.item():
            if rank == 0:
                d.base.atomic_json(dest/'paused.json', {'status':'paused','step':j+1,'tokens':consumed})
            raise RuntimeError('Safe stop saved checkpoint; candidate remains incomplete')
    assert consumed == expected
    if final_val is None:
        final_val = evaluate(model, val)
    if rank == 0:
        summary = {'status':'completed','config':config,'tokens':consumed,'supervised_positions':supervised,
                   'val':final_val,'elapsed_seconds':elapsed+time.monotonic()-start,
                   'peak_bytes_rank0':torch.cuda.max_memory_allocated()}
        d.base.atomic_json(dest/'summary.json', summary)
        log.close()
        print('COMPLETE',json.dumps(summary),flush=True)
    dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',required=True,choices=['depo2','brevo2'])
    parser.add_argument('--candidate',required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=10),
        device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    d.CONTROL = dist.new_group(backend='gloo', timeout=datetime.timedelta(minutes=10))
    assert dist.get_world_size() == 8
    try:
        train(args)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    else:
        dist.destroy_process_group()
