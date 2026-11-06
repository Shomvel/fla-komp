import torch, triton
from komp_slot import block_komp as BK
from validate_block_komp import inputs
for (b, t, h, dk, dv, m, c) in ((2, 96, 2, 128, 128, 64, 64), (1, 256, 2, 64, 64, 32, 64), (1, 256, 2, 64, 64, 32, 32)):
    xs = [x.requires_grad_() for x in inputs(b, t, h, dk, dv, m, torch.bfloat16, 0)]
    for mw in (4, 8):
        BK.WARPS.update(maps=mw, maps_bwd=mw)
        try:
            o = BK.block_komp(*xs[:5], (xs[5], xs[6]), chunk_size=c)
            torch.autograd.grad((o[0], *o[1]), xs, [torch.ones_like(o[0]), torch.ones_like(o[1][0]), torch.ones_like(o[1][1])])
            ks = {k: v for k, v in BK.block_maps_kernel.device_caches[0][0].items()} if False else None
            print('ok', (b, t, h, dk, dv, m, c), 'warps', mw, flush=True)
        except Exception as e:
            print('FAIL', (b, t, h, dk, dv, m, c), 'warps', mw, repr(e)[:160], flush=True)
