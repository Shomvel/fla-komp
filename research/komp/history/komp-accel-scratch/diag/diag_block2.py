import traceback, torch
from komp_slot import block_komp as BK
from validate_block_komp import inputs
xs = [x.requires_grad_() for x in inputs(2, 96, 2, 128, 128, 64, torch.bfloat16, 0)]
try:
    o = BK.block_komp(*xs[:5], (xs[5], xs[6]), chunk_size=64)
    torch.autograd.grad((o[0], *o[1]), xs, [torch.ones_like(o[0]), torch.ones_like(o[1][0]), torch.ones_like(o[1][1])])
    print('ok')
except Exception:
    traceback.print_exc()
