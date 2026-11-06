"""Near-duplicate stress case of validate_block_komp.py under block_fast flag combinations."""
import itertools, torch
from komp_slot import block_fast as BF
from komp_slot.block_komp import block_komp as block_komp_v0
from komp_slot.block_reference import block_komp_reference
from validate_block_komp import inputs, run, rms

names = ('out', 'keys', 'values', 'dq', 'dk', 'dv', 'dg', 'dbeta', 'dk0', 'dv0')
for case_m, case_c in ((16, 32), (64, 64)):
    xs = inputs(1, 160, 2, 64, 64, case_m, torch.float32, 21)
    center = torch.randn(1, 1, 2, 64, device='cuda')
    xs[1] = center+.02*xs[1]
    xs[5] = center.transpose(1, 2)+.02*xs[5]
    ref_x = [x.detach().double().requires_grad_() for x in xs]
    ref = run(block_komp_reference, ref_x, case_c)
    ws = [torch.randn_like(x).float() for x in ref]
    gref = torch.autograd.grad(ref, ref_x, [w.double() for w in ws])

    def errs(fn):
        x = [y.detach().clone().requires_grad_() for y in xs]
        out = run(fn, x, case_c)
        gr = torch.autograd.grad(out, x, ws)
        return [rms(a, b) for a, b in zip((*out, *gr), (*ref, *gref))]

    base = errs(block_komp_v0)
    print(case_m, case_c, 'v0       ', ' '.join(f'{n}={e:.1e}' for n, e in zip(names, base)), flush=True)
    for r1, merge, fuse in itertools.product((False, True), (False, True), (False, True)):
        BF.RANK1_SWEEP, BF.MERGE_KEYGRAD, BF.FUSE_POOL_NORM = r1, merge, fuse
        e = errs(BF.block_komp_fast)
        worst = max(a/max(5e-3/1.25, b) for a, b in zip(e, base))
        print(case_m, case_c, f'r1={r1:d} merge={merge:d} fuse={fuse:d}', ' '.join(f'{n}={x:.1e}' for n, x in zip(names, e)),
              f'worst/limit {worst/1.25:.2f}', flush=True)
