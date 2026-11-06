"""GPU numerical acceptance for the GDN timescale gate (no training)."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from komp_slot.decay import gdn_decay, gsa_decay, matched_dt_bias

ROOT = Path(__file__).resolve().parent
REPORT = {'checks': []}


def check(name, got, expected, atol=2e-6, rtol=5e-5):
    assert bool(torch.isfinite(got).all()), name
    error = (got.double() - expected.double()).abs()
    REPORT['checks'].append({'name': name, 'max_abs': error.max().item(),
                              'atol': atol, 'rtol': rtol})
    torch.testing.assert_close(got.double(), expected.double(), atol=atol, rtol=rtol)


def reference(logits, A_log, dt_bias):
    return (-A_log.exp() * F.softplus(logits + dt_bias)).exp()


def run_case(shape, dtype, strided=False, extreme=False):
    torch.manual_seed(731 + len(shape))
    x = torch.randn(shape, device='cuda', dtype=dtype)
    a = torch.randn(shape[2:], device='cuda') * .5
    bias = torch.randn_like(a) + matched_dt_bias()
    if extreme:
        values = torch.tensor([-1000., -100., -30., -20., -14., -1., 0., 1., 20., 100., 1000.], device='cuda')
        x = values[torch.arange(x.numel(), device='cuda') % values.numel()].reshape(shape).to(dtype)
        a = torch.linspace(-80., 80., a.numel(), device='cuda').reshape(a.shape)
        bias.zero_()
    if strided:
        x = x.transpose(0, 1).contiguous().transpose(0, 1)
        a = torch.stack((a, a), -1)[..., 0]
        bias = torch.stack((bias, bias), -1)[..., 0]
    xs = [t.detach().requires_grad_() for t in (x, a, bias)]
    refs = [t.detach().double().requires_grad_() for t in xs]
    alpha = gdn_decay(*xs)
    assert alpha.dtype == torch.float32 and alpha.shape == x.shape
    expected = reference(*refs)
    tag = f'{shape}/{dtype}/strided{strided}/extreme{extreme}'
    check(tag + '/alpha', alpha, expected, atol=3e-6 if extreme else 2e-6)
    weights = torch.randn_like(alpha)
    grads = torch.autograd.grad(alpha, xs, weights)
    refgrads = torch.autograd.grad(expected, refs, weights.double())
    for name, got, ref, original in zip(('dx', 'dA', 'dbias'), grads, refgrads, xs):
        # Autograd returns the leaf dtype, including BF16/FP16 quantization.
        rounded = ref.to(original.dtype)
        low = original.dtype != torch.float32
        check(tag + '/' + name, got, rounded,
              atol=2e-3 if low else (3e-5 if extreme else 1e-5),
              rtol=8e-3 if low else (2e-4 if extreme else 5e-5))
    print('PASS', tag, flush=True)


def equivalence():
    x = torch.linspace(-15, 15, 257, device='cuda').reshape(1, 257, 1).requires_grad_()
    a = torch.zeros(1, device='cuda', requires_grad=True)
    b = torch.tensor([matched_dt_bias()], device='cuda', requires_grad=True)
    got = gdn_decay(x, a, b)
    check('rate1/sigmoid', got, torch.sigmoid(-x - b))
    zero = torch.zeros((2, 3, 4), device='cuda')
    init = gdn_decay(zero, torch.zeros(4, device='cuda'), torch.full((4,), matched_dt_bias(), device='cuda'))
    check('matched_init/alpha98', init, torch.full_like(init, .98), atol=1e-7, rtol=0.)
    check('matched_init/rate3',
          gdn_decay(zero, torch.full((4,), __import__('math').log(3.), device='cuda'),
                    torch.full((4,), matched_dt_bias(rate=3.), device='cuda')),
          torch.full_like(init, .98), atol=1e-7, rtol=0.)
    # Scalar expanded to slot gates is exactly the same function.
    base = gdn_decay(x, a, b)
    expanded = gdn_decay(x[..., None].expand(1, 257, 1, 5), a[:, None].expand(1, 5), b[:, None].expand(1, 5))
    check('scalar_slot_broadcast', expanded, base[..., None].expand_as(expanded), atol=0., rtol=0.)


def gsa_cases():
    torch.manual_seed(733)
    for shape in ((257,), (2, 129, 3), (2, 65, 2, 5)):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for normalizer in (1., 8.):
                x = (torch.randn(shape, device='cuda') * 8).to(dtype)
                if len(shape) > 1:
                    x = x.transpose(0, 1).contiguous().transpose(0, 1)
                x = x.requires_grad_()
                refx = x.detach().double().requires_grad_()
                got = gsa_decay(x, normalizer)
                ref = (F.logsigmoid(refx) / normalizer).exp()
                tag = f'gsa/{shape}/{dtype}/normalizer{normalizer}'
                check(tag + '/alpha', got, ref)
                weights = torch.randn_like(got)
                dx = torch.autograd.grad(got, x, weights)[0]
                refdx = torch.autograd.grad(ref, refx, weights.double())[0].to(dtype)
                check(tag + '/dx', dx, refdx, atol=2e-3 if dtype != torch.float32 else 2e-6,
                      rtol=8e-3 if dtype != torch.float32 else 5e-5)
                if normalizer == 1.:
                    check(tag + '/sigmoid', got, torch.sigmoid(x.float()))
    x = torch.tensor([-1000., -100., -20., 0., 20., 100., 1000.], device='cuda', requires_grad=True)
    refx = x.detach().double().requires_grad_()
    got = gsa_decay(x)
    ref = (F.logsigmoid(refx) / 8.).exp()
    check('gsa/extreme/alpha', got, ref)
    check('gsa/extreme/dx', torch.autograd.grad(got.sum(), x)[0], torch.autograd.grad(ref.sum(), refx)[0])
    print('PASS GSA gates', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'evidence/decay-gate.json')
    args = parser.parse_args()
    REPORT['environment'] = {'python': platform.python_version(), 'torch': str(torch.__version__),
                              'triton': str(triton.__version__), 'gpu': torch.cuda.get_device_name()}
    REPORT['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (Path(__file__), ROOT / 'komp_slot/decay.py')}
    try:
        for shape in ((1, 1, 1), (2, 129, 3), (2, 67, 2, 5), (64, 192, 2)):
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                run_case(shape, dtype, strided=shape[0] > 1)
        run_case((2, 65, 11), torch.float32, extreme=True)
        run_case((2, 65, 11), torch.bfloat16, extreme=True)
        equivalence()
        gsa_cases()
        REPORT['status'] = 'passed'
    except Exception as exc:
        REPORT['status'] = 'failed'
        REPORT['error'] = repr(exc)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(REPORT, indent=2) + '\n')
        print('REPORT', REPORT['status'], len(REPORT['checks']), flush=True)


if __name__ == '__main__':
    main()
