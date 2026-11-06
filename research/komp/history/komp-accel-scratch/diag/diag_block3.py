import torch, triton
from komp_slot import block_komp as BK
from benchmark_block import make, run
for name, cfg in (('main', (2, 2048, 4, 64, 64, 32)), ('8k', (1, 8192, 64, 128, 128, 64))):
    data, slots = make(*cfg)
    x = [y.detach().clone().requires_grad_() for y in data['block']]
    o = run('block', x, slots, 64)
    torch.autograd.grad(o, x, [torch.ones_like(y) for y in o])
    for kern in (BK.block_maps_kernel, BK.block_mapgrad_kernel, BK.block_maps_bwd_kernel, BK.block_keygrad_kernel):
        for dev, caches in kern.device_caches.items():
            for key, k in caches[0].items():
                print(name, kern.fn.__name__, 'regs', k.n_regs, 'spills', k.n_spills, 'smem', k.metadata.shared, flush=True)
