import torch
from komp_slot import block_fast as BF, block_komp as BK
from benchmark_block import make, run
for name, cfg in (('main', (2, 2048, 4, 64, 64, 32)), ('8k', (1, 8192, 64, 128, 128, 64))):
    data, slots = make(*cfg)
    x = [y.detach().clone().requires_grad_() for y in data['block']]
    o = run('block', x, slots, 64)
    torch.autograd.grad(o, x, [torch.ones_like(y) for y in o])
kerns = [BF.att_kernel, BF.proj_kernel, BF.ut_kernel, BF.compose_kernel, BF.proj_bwd_kernel, BF.ut_bwd_kernel,
         BK.block_mapgrad_kernel, BK.block_keygrad_kernel]
for kern in kerns:
    for dev, caches in kern.device_caches.items():
        for key, k in caches[0].items():
            print(kern.fn.__name__, 'regs', k.n_regs, 'spills', k.n_spills, 'smem', k.metadata.shared, flush=True)
