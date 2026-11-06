"""Key-gradient block width at the main and training shapes (8k prefers 64)."""
import torch
from komp_slot import block_fast as BF
from benchmark_block import make, run
from komp_bench.timing import graph_ms

for cfg in ((2, 2048, 4, 64, 64, 32), (64, 192, 2, 64, 64, 32)):
    data, slots = make(*cfg)
    for kb in (32, 64):
        BF.KEY_BLOCK = kb
        row = []
        for name, sc in (('block', None), ('block_slot', None), ('block_slot', data['scores'])):
            x = [y.detach().clone().requires_grad_() for y in data[name]]
            f = lambda: run(name, x, slots, 64, sc)
            with torch.no_grad():
                ws = [torch.randn_like(o)*.01 for o in f()]
            row.append(graph_ms(lambda: torch.autograd.grad(f(), x, ws), unroll=10, replays=2))
        print(cfg, 'key_block', kb, 'scalar/slot/learned fwd+bwd', ' '.join(f'{v:.3f}' for v in row), flush=True)
BF.KEY_BLOCK = 64
