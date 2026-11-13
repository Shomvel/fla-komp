"""Fixed-input single-H100 profiles and paired timing of the production configuration."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=['operator', 'mixer', 'model'], required=True)
ap.add_argument('--snapshot', default='snapshot-baseline')
ap.add_argument('--tag', required=True)
ap.add_argument('--warmup', type=int, default=4)
ap.add_argument('--reps', type=int, default=10)
ap.add_argument('--profile', action='store_true')
ap.add_argument('--abba', action='store_true')
ap.add_argument('--compare-reference', default=None)
ap.add_argument('--cache-ut-layers', type=int, default=0)
ap.add_argument('--checkpoint', default='/gpfs/users/linlizhi/komp-learned123-1b-50b-20261109-mb16-direct/runs/block/checkpoint-010240')
ap.add_argument('--flags', default='{}')
args = ap.parse_args()
os.environ['KOMP_SNAPSHOT'] = str(ROOT / args.snapshot)
os.environ['KOMP_UT_CACHE_LAYERS'] = str(args.cache_ut_layers)
from models import LM, optimizer
import fla.ops.komp.block_fast as op

torch.set_num_threads(2)
torch.manual_seed(734)
torch.cuda.manual_seed_all(734)
for k, v in json.loads(args.flags).items():
    assert hasattr(op, k), k
    setattr(op, k, v)
checkpoint = Path(args.checkpoint)
assert (checkpoint / 'complete.json').exists(), checkpoint
fixture = ROOT / 'fixture/operator-trained.pt'
if args.mode == 'operator' and fixture.exists():
    cached = torch.load(fixture, map_location='cpu', weights_only=False)
    assert cached['checkpoint'] == str(checkpoint.resolve())
    parent, input_sha = cached['parent'], cached['input_sha']
    operands = [z.cuda() for z in cached['operands']]
    del cached
    model = mix = x = None
else:
    state = torch.load(checkpoint / 'model.pt', map_location='cpu', weights_only=False)
    model = LM('block')
    model.load_state_dict(state['model'], strict=True)
    parent = {k: state[k] for k in ['step', 'tokens', 'source_sha', 'data_sha']}
    del state
    model = model.cuda().train()
    manifest = json.loads((ROOT / 'configs/data-manifest.json').read_text())
    data = np.memmap(manifest['validation_tokens_path'], dtype=np.uint16, mode='r')
    ids = torch.tensor(np.array(data[:16*2049], dtype=np.int64).reshape(16, 2049), device='cuda')
    input_sha = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        x = model.model.layers[0].attn_norm(model.model.embeddings(ids[:, :-1])).detach()
    mix = model.model.layers[0].attn
    if args.mode == 'operator':
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            operands = mix.operands(x)
        assert fixture.parent.exists()
        torch.save(dict(checkpoint=str(checkpoint.resolve()), parent=parent, input_sha=input_sha,
                        operands=[z.detach().cpu() for z in operands]), fixture)
if args.mode == 'operator':
    q, k, v, g, beta, keys, values, scores, initial = operands
    inputs = [z.detach().clone().requires_grad_(True) for z in [q, k, v, g, beta, keys, values, scores]]
    del model, mix, operands, x
    def step():
        for z in inputs:
            z.grad = None
        q, k, v, g, beta, keys, values, scores = inputs
        out, final = op.block_komp_fast(q, k, v, g, beta, (keys, values), chunk_size=128,
                                      evict='learned', scores=scores, initial_scores=initial)
        # Production loss consumes outputs; returned dictionary does not contribute to the loss.
        out.float().square().mean().backward()
        return float(out.detach().float().square().mean())
elif args.mode == 'mixer':
    del model
    mix.checkpoint_mixer = True
    x = x.detach().requires_grad_(True)
    def step():
        mix.zero_grad(set_to_none=True)
        x.grad = None
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = mix(x)[0]
            loss = out.float().square().mean()
        loss.backward()
        return float(loss.detach())
else:
    del mix, x
    opt = optimizer(model)
    # Initialize real Adam state once before timing; weights remain fixed for paired measurements.
    for p in model.parameters():
        opt.state[p]['step'] = torch.tensor(0., device='cuda')
        opt.state[p]['exp_avg'] = torch.zeros_like(p)
        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p)
    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(ids[:, :-1], ids[:, 1:]) / 32768
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        # lr=0 measures the same fused Adam kernels without changing the paired checkpoint.
        for group in opt.param_groups:
            group['lr'] = 0.
        opt.step()
        return float(loss.detach())
gc.collect()
torch.cuda.empty_cache()
for i in range(args.warmup):
    value = step()
    torch.cuda.synchronize()
    print(json.dumps({'phase': 'warmup', 'index': i, 'value': value}), flush=True)
torch.cuda.reset_peak_memory_stats()
times = []
for i in range(args.reps):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = step()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)
result = dict(tag=args.tag, mode=args.mode, snapshot=args.snapshot, flags=json.loads(args.flags),
              parent=parent, input_sha=input_sha, shape=[16, 2048, 32, 64], slots=32, chunk=128,
              seconds=times, median_seconds=statistics.median(times), value=value,
              peak_bytes=torch.cuda.max_memory_allocated(), reserved_bytes=torch.cuda.max_memory_reserved(),
              operator_path=op.__file__, operator_sha=hashlib.sha256(Path(op.__file__).read_bytes()).hexdigest(),
              gpu=torch.cuda.get_device_name(), torch=torch.__version__)
if args.profile:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as prof:
        step()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(ROOT / 'evidence' / f'{args.tag}.trace.json'))
    (ROOT / 'evidence' / f'{args.tag}.profile.txt').write_text(prof.key_averages().table(sort_by='self_device_time_total', row_limit=60))
    events = [dict(name=e.key, count=e.count, cpu_us=e.self_cpu_time_total,
                   device_us=e.self_device_time_total) for e in prof.key_averages()]
    result['profile_events'] = sorted(events, key=lambda e: e['device_us'], reverse=True)[:60]
if args.abba or args.compare_reference:
    if args.compare_reference:
        assert args.mode == 'model'
        import importlib.util, sys
        import models as model_module
        import fla.models.komp.modeling_komp as architecture
        reference_path=ROOT / args.compare_reference / 'fla/ops/komp/block_fast.py'
        spec=importlib.util.spec_from_file_location('fla.ops.komp.block_fast_timing_reference',reference_path)
        reference=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reference;spec.loader.exec_module(reference)
        result['reference_sha']=hashlib.sha256(reference_path.read_bytes()).hexdigest()
    else:
        assert args.snapshot == 'snapshot' and hasattr(op, 'DECAY_SLOT_REDUCE')
    paired = []
    for enabled in [False, True, True, False]:
        if args.compare_reference:
            model_module.block_komp_fast=op.block_komp_fast if enabled else reference.block_komp_fast
            if hasattr(architecture,'ROUNDED_SWIGLU'):architecture.ROUNDED_SWIGLU=enabled
            if 'fla.modules.komp_rmsnorm' in sys.modules:sys.modules['fla.modules.komp_rmsnorm'].OPTIMIZED_NORM=enabled
        else:
            op.DECAY_SLOT_REDUCE = enabled
        for _ in range(2):
            step()
        torch.cuda.synchronize()
        elapsed = []
        for _ in range(args.reps):
            start = time.perf_counter()
            step()
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter()-start)
        paired.append(dict(candidate=enabled, seconds=elapsed, median_seconds=statistics.median(elapsed)))
    result['abba'] = paired
result['cache_ut_layers']=args.cache_ut_layers
result['source_files']={str(p.relative_to(ROOT/args.snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted((ROOT/args.snapshot).rglob('*.py'))}
(ROOT / 'evidence' / f'{args.tag}.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
