"""Same-GPU GDN / GSA / per-token KOMP / Block-KOMP operator timing at equal per-head state size."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import triton
import benchmark_fla as fla_bench
from benchmark_fla import chunk_gsa, chunk_gated_delta_rule, FLA_ROOT
from komp_bench.timing import graph_ms
from komp_slot.block_komp import block_komp
from komp_slot.block_fast import block_komp_fast
from komp_slot.geometry_chunk import triton_geometry_chunked
from komp_slot.fast_chunk import fast_chunk

ROOT = Path(__file__).resolve().parent


def make(b, t, h, dk, dv, m, seed=0):
    torch.manual_seed(seed)
    r = lambda *s: torch.randn(*s, device='cuda')
    q, k = r(b, t, h, dk).bfloat16(), r(b, t, h, dk).bfloat16()
    v = (r(b, t, h, dv)*.2).bfloat16()
    alpha = torch.full((b, t, h), .98, device='cuda')
    g = alpha.log()
    beta = torch.full_like(alpha, .2)
    gs = g[..., None].expand(b, t, h, m).contiguous()
    s = (1-gs.exp()).bfloat16()
    slots = (torch.arange(t, device='cuda') % m)[None, :, None].expand(b, t, h).contiguous()
    return {
        'gdn': [q, k, v, g, beta, r(b, h, dk, dv)*.05],
        'gsa': [q, k, v, s, gs, r(b, h, dk, m)*.05, r(b, h, m, dv)*.05],
        'komp': [q, k, v, alpha, beta, r(b, h, m, dk), r(b, h, m, dv)*.05],
        'block': [q, k, v, g, beta, r(b, h, m, dk), r(b, h, m, dv)*.05],
    } | {'block_v0': None}, slots


def run(name, xs, slots, c):
    if name == 'gdn':
        return chunk_gated_delta_rule(*xs[:5], initial_state=xs[5], output_final_state=True, use_qk_l2norm_in_kernel=True)
    if name == 'gsa':
        o, st = chunk_gsa(*xs[:5], initial_state=(xs[5], xs[6]), output_final_state=True)
        return o, *st
    if name == 'komp':
        hh, pp, rr, ks = triton_geometry_chunked(xs[0], xs[1], slots, xs[5])
        o, vs = fast_chunk(hh, pp, rr, xs[2], xs[3], xs[4], slots, xs[6])
        return o, ks, vs
    fn = block_komp_fast if name == 'block' else block_komp
    o, (ks, vs) = fn(*xs[:5], (xs[5], xs[6]), chunk_size=c)
    return o, ks, vs


def measure(name, xs, slots, c, unroll):
    row = {}
    x = [y.detach().clone().requires_grad_() for y in xs]
    with torch.no_grad():
        row['fwd_ms'] = graph_ms(lambda: run(name, x, slots, c), unroll=unroll, replays=max(2, 20//unroll))
        ws = [torch.randn_like(o)*.01 for o in run(name, x, slots, c)]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    torch.autograd.grad(run(name, x, slots, c), x, ws)
    torch.cuda.synchronize()
    row['train_peak_extra_gib'] = (torch.cuda.max_memory_allocated()-base)/2**30
    row['fwd_bwd_ms'] = graph_ms(lambda: torch.autograd.grad(run(name, x, slots, c), x, ws), unroll=unroll,
                                 replays=max(2, 20//unroll))
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default=str(ROOT/'evidence/block-benchmark.json'))
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    configs = [dict(b=2, t=2048, h=4, dk=64, dv=64, m=32, unroll=10),
               dict(b=1, t=8192, h=64, dk=128, dv=128, m=64, unroll=2)]
    report = {'rows': [], 'method': 'CUDA Graph replay, bounded unroll, median of 3; BF16 q/k/v, FP32 gates/states; '
              'equal per-head state: GDN Dk*Dv, GSA M*(Dk+Dv), KOMP M*(Dk+Dv)',
              'environment': {'torch': str(torch.__version__), 'triton': triton.__version__, 'gpu': torch.cuda.get_device_name()},
              'sources': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in [Path(__file__), *sorted((ROOT/'komp_slot').glob('*.py'))]}}
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        data, slots = make(cfg['b'], cfg['t'], cfg['h'], cfg['dk'], cfg['dv'], cfg['m'])
        variants = [('gdn', 'gdn', 64), ('gsa', 'gsa', 64), ('komp_per_token', 'komp', 16),
                    ('block_c64', 'block', 64), ('block_c32', 'block', 32), ('block_v0_c64', 'block_v0', 64)]
        for label, name, c in variants:
            row = {**{k: v for k, v in cfg.items() if k != 'unroll'}, 'operator': label, 'chunk': c}
            try:
                row.update(measure(name, data['block' if name == 'block_v0' else name], slots, c, cfg['unroll']))
            except Exception as e:
                row['error'] = repr(e)[:300]
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            report['rows'].append(row)
            dest.write_text(json.dumps(report, indent=2)+'\n')
            print('BENCH', json.dumps(row), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
