"""Validation-only LR/WD search and a prespecified tenfold scale-up."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import shortconv_experiment as conv
import experiment as base
import decay_experiment as decay
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent / 'lrwd_runs'
PREVIOUS = Path('/gpfs/users/linlizhi/komp-shortconv-20261031-r2/physics_lm/shortconv_runs')
LRS = (2.5e-4, 5e-4, 1e-3)
WDS = (0., .01, .1)
CONFIG_EXTRA = lambda model: {}


def trial_name(kind, seed, lr, wd, scale):
    return f'{kind}-{scale}-lr{lr:g}-wd{wd:g}-seed{seed}'


def dimensions(scale):
    base.D, base.H, base.K, base.M, base.L, base.FF = ((512, 8, 64, 32, 15, 1536)
                                                       if scale == 'large' else (128, 2, 64, 32, 4, 256))


def manifest(scale):
    path = ROOT / 'data100m/manifest.json' if scale == 'large' else Path(base.__file__).parent / 'data-manifest.json'
    return json.loads(path.read_text())


def load(split, scale):
    info = manifest(scale)
    root = ROOT / 'data100m' if scale == 'large' else base.DATA
    path = root / (split + '.jsonl')
    digest = hashlib.sha256()
    rows = []
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line)
            rows.append(json.loads(line))
    assert digest.hexdigest() == info['splits'][split]['sha256']
    return rows


def optimizer(model, wd):
    opt = decay.optimizer(model)
    for group in opt.param_groups:
        if group['weight_decay']:
            group['weight_decay'] = wd
    return opt


def environment():
    import triton
    import google.protobuf
    from google.protobuf.internal.api_implementation import Type
    return dict(torch=str(torch.__version__), triton=triton.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(), python=platform.python_version(),
                protobuf=google.protobuf.__version__, protobuf_backend=Type())


def preflight():
    ROOT.mkdir(parents=True, exist_ok=True)
    ticket = json.loads((PREVIOUS / 'preflight.json').read_text())
    assert ticket['status'] == 'passed' and ticket['source_sha256'] == conv.sources(), 'Frozen source changed'
    report = {'frozen_shortconv_checks': 'passed', 'checks': []}
    for kind in conv.KINDS:
        dimensions('small')
        model = conv.make_model(kind + '_conv', 731).cuda()
        hashes = conv.hashes(model, lambda n: True)
        previous = json.loads((PREVIOUS / 'results' / f'{kind}_conv-seed731/manifest.json').read_text())
        assert hashes == previous['nonconv_initial_sha256'] | previous['conv_initial_sha256']
        for wd in WDS:
            opt = optimizer(model, wd)
            actual = {id(p): group['weight_decay'] for group in opt.param_groups for p in group['params']}
            assert all(actual[id(p)] == (0. if n.endswith(('decay.bias', 'A_log', 'dt_bias')) else wd)
                       for n, p in model.named_parameters())
        sample = load('val', 'small')[:4]
        opt = optimizer(model, .1)
        with base.amp():
            loss = base.loss_items(model, sample)[0].mean()
        loss.backward()
        assert bool(torch.isfinite(nn.utils.clip_grad_norm_(model.parameters(), 1.)))
        opt.step()
        report['checks'].append({'kind': kind, 'initialization': 'matched previous', 'optimizer_groups': 'passed',
                                 'loss': float(loss.detach())})
        del model, opt
        torch.cuda.empty_cache()
    for peak in LRS:
        for step in (1, 111, 2231):
            lr, mult = base.schedule(step, 2231)
            effective = lr * peak / 5e-4
            assert 0 < effective <= peak and .0999 <= mult <= .3001
    report.update(status='passed', environment=environment(), sources=source_hashes())
    (ROOT / 'preflight.json').write_text(json.dumps(report, indent=2))


def source_hashes():
    return {**conv.sources(), **{f'physics_lm/{n}': hashlib.sha256((Path(__file__).parent / n).read_bytes()).hexdigest()
                                for n in ('lrwd_experiment.py', 'lrwd_launch.py', 'prepare_100m.py')}}


def train(kind, seed, lr, wd, scale):
    assert json.loads((ROOT / 'preflight.json').read_text())['sources'] == source_hashes()
    dimensions(scale)
    name = trial_name(kind, seed, lr, wd, scale)
    dest = ROOT / 'results' / name
    dest.mkdir(parents=True, exist_ok=False)
    model = conv.make_model(kind + '_conv', seed).cuda()
    base.check_slot_diversity(model, kind)
    rows, val = load('train', scale), load('val', scale)
    data = manifest(scale)
    order = list(range(len(rows)))
    random.Random(12345).shuffle(order)
    batch_size = 64
    total = math.ceil(len(rows) / batch_size)
    config = {'mixer': kind, 'scale': scale, 'seed': seed, 'peak_lr': lr, 'weight_decay': wd,
              'width': base.D, 'layers': base.L, 'heads': base.H, 'key_value_dim': base.K,
              'slots': base.M, 'ffn': base.FF, 'vocab': base.VOCAB, 'batch': batch_size,
              'params': sum(p.numel() for p in model.parameters()), 'steps': total,
              'data': data, 'environment': environment(), 'sources': source_hashes(),
              'initialization_sha256': conv.hashes(model, lambda n: True),
              'schedule': 'warmup5% cosine to10%; residual output LR multiplier .1->.3',
              'wd_exclusions': ['decay.bias', 'A_log', 'dt_bias'],
              'selection': 'final full validation mean content CE only; no test/confirm access during search',
              'short_conv': 'q/k/v depthwise causal width4 + SiLU, bias=False'}
    config.update(CONFIG_EXTRA(model))
    (dest / 'manifest.json').write_text(json.dumps(config, indent=2))
    opt = optimizer(model, wd)
    probe = io.BytesIO()
    torch.save({'model': model.state_dict(), 'config': config}, probe)
    probe.seek(0)
    restored = torch.load(probe, map_location='cpu', weights_only=True)
    assert restored['config'] == config
    del restored, probe
    causal = conv.check_model_causality(model, val[:2])
    (dest / 'initial-causality.json').write_text(json.dumps(causal, indent=2))
    started = time.monotonic()
    unique = 0
    torch.cuda.reset_peak_memory_stats()
    with (dest / 'metrics.jsonl').open('w', buffering=1) as log:
        log.write(json.dumps({'step': 0, 'val': base.evaluate(model, val[:128])}) + '\n')
        for step, offset in enumerate(range(0, len(order), batch_size), 1):
            batchrows = [rows[j] for j in order[offset:offset + batch_size]]
            unique += sum(len(r['input_ids']) for r in batchrows)
            scheduled_lr, mult = base.schedule(step, total)
            scheduled_lr *= lr / 5e-4
            for group in opt.param_groups:
                group['lr'] = scheduled_lr * (mult if group['residual_output'] else 1.)
            opt.zero_grad(set_to_none=True)
            with base.amp():
                loss = base.loss_items(model, batchrows)[0].mean()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not bool(torch.isfinite(norm)) or not bool(torch.isfinite(loss)):
                raise RuntimeError(f'Nonfinite {name} at step {step}; no automatic retry')
            opt.step()
            if step % 50 == 0 or step == total:
                record = {'step': step, 'loss': float(loss.detach()), 'grad_norm': float(norm),
                          'lr': scheduled_lr, 'output_mult': mult, 'unique_tokens': unique,
                          'seconds': time.monotonic() - started}
                if step % (2000 if scale == 'large' else 500) == 0 or step == total:
                    record['val'] = base.evaluate(model, val)
                log.write(json.dumps(record) + '\n')
                print(name, json.dumps(record), flush=True)
    assert unique == data['splits']['train']['tokens']
    del rows, order, opt
    final_val = base.evaluate(model, val)
    torch.save({'model': model.state_dict(), 'config': config}, dest / 'model.pt')
    saved = torch.load(dest / 'model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model'], strict=True)
    error = abs(base.evaluate(model, val)['mean_content_ce'] - final_val['mean_content_ce'])
    assert error == 0
    checks = conv.check_model_causality(model, val[:2])
    summary = {'status': 'completed', 'config': config, 'val': final_val, 'unique_tokens': unique,
               'elapsed_seconds': time.monotonic() - started, 'reload_validation_ce_error': error,
               'trained_causality': checks, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('COMPLETE', name, final_val['mean_content_ce'], flush=True)


def select():
    selection = {'criterion': 'minimum final full validation mean content CE at seed731', 'models': {}}
    for kind in conv.KINDS:
        candidates = []
        for lr in LRS:
            for wd in WDS:
                name = trial_name(kind, 731, lr, wd, 'small')
                summary = json.loads((ROOT / 'results' / name / 'summary.json').read_text())
                assert summary['status'] == 'completed' and summary['unique_tokens'] == 10001108
                candidates.append(dict(lr=lr, wd=wd, val_ce=summary['val']['mean_content_ce'], trial=name))
        selected = min(candidates, key=lambda row: (row['val_ce'], row['lr'], row['wd']))
        selection['models'][kind] = {'selected': selected, 'candidates': candidates}
    path = ROOT / 'selection.json'
    with path.open('x') as out:
        json.dump(selection, out, indent=2)
    print('LOCKED SELECTION', json.dumps(selection), flush=True)


def evaluate_selected(scale_filter):
    selection = json.loads((ROOT / 'selection.json').read_text())
    for kind, item in selection['models'].items():
        best = item['selected']
        for scale, seed in (('small', 731), ('small', 732), ('large', 731)):
            if scale != scale_filter:
                continue
            dimensions(scale)
            name = trial_name(kind, seed, best['lr'], best['wd'], scale)
            dest = ROOT / 'results' / name
            saved = torch.load(dest / 'model.pt', map_location='cpu', weights_only=True)
            model = conv.make_model(kind + '_conv', seed).cuda()
            model.load_state_dict(saved['model'], strict=True)
            confirm = load('confirm', scale)
            with torch.no_grad():
                metrics = {split: base.evaluate(model, load(split, scale)) for split in ('test', 'confirm')}
                generation, records = conv.generate(model, confirm)
            metrics.update(generation=generation, selection_sha256=hashlib.sha256((ROOT / 'selection.json').read_bytes()).hexdigest())
            (dest / 'heldout.json').write_text(json.dumps(metrics, indent=2))
            (dest / 'generations.json').write_text(json.dumps(records, indent=2))
            print('HELDOUT', name, json.dumps(metrics), flush=True)
            del model, saved
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=('preflight', 'train', 'select', 'evaluate'), required=True)
    parser.add_argument('--kind', choices=conv.KINDS)
    parser.add_argument('--seed', type=int, default=731)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=.1)
    parser.add_argument('--scale', choices=('small', 'large'), default='small')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(9041)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.action == 'preflight': preflight()
    elif args.action == 'select': select()
    elif args.action == 'evaluate': evaluate_selected(args.scale)
    else: train(args.kind, args.seed, args.lr, args.wd, args.scale)
