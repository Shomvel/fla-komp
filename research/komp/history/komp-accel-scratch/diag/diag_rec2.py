import itertools, functools, torch, triton
import validate as base
from komp_bench.timing import graph_ms
from komp_slot import fast_chunk as F, scan as S
t, m = 2048, 32
xs, slots = base.core_inputs((2, t, 4, m, 64), 'cuda', 3)
xs[3] = xs[3].bfloat16()
x = base.clone_grad(xs)
call = lambda: F.fast_chunk(x[0], x[1], x[2], x[3], x[4], x[5], slots, x[6])
with torch.no_grad():
    ws = [torch.randn_like(o) for o in call()]
def measure(tag):
    with torch.no_grad():
        f = graph_ms(call)
    fb = graph_ms(lambda: torch.autograd.grad(call(), x, ws))
    print(f'{tag} fwd {f:.4f} fwd+bwd {fb:.4f}', flush=True)
measure('default')
for w in (1, 2, 8):
    F.WARPS['prepare'] = w; measure(f'prepare={w}')
F.WARPS['prepare'] = 4
for w in (1, 2):
    S.WARPS.update(compose=w, scan=w, expand=w); measure(f'scan={w}')
S.WARPS.update(compose=4, scan=4, expand=4)
for w in (1, 2):
    F.WARPS.update(offset=w, output=w); measure(f'offset/output={w}')
F.WARPS.update(offset=4, output=4)
for g in (4, 8, 16):
    call = functools.partial(F.fast_chunk, x[0], x[1], x[2], x[3], x[4], x[5], slots, x[6], group=g); measure(f'group={g}')
