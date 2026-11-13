"""Source-bound learned-eviction checks against an independent FP64 algebra."""
import json

import torch
from torch.nn import functional as F

from models import ROOT
from value_learned import block_komp_value_learned as block_komp_fast
from value_learned import recurrent_value_learned as block_komp_flex


def compare(xs, scores, initial_scores, name, backend="triton"):
    actual = [x.detach().requires_grad_() for x in xs]
    reference = [x.detach().double().requires_grad_() for x in xs]

    def run(a, reference=False):
        fn = block_komp_flex if reference else block_komp_fast
        o, (k, s) = fn(*a[:5], (a[5], a[6]), chunk_size=128,
                      evict='learned', scores=scores, initial_scores=initial_scores, **({} if reference else {'backend': backend}))
        return o, k, s

    a, r = run(actual), run(reference, True)
    weights = [torch.randn_like(x) for x in a]
    da = torch.autograd.grad(a, actual, weights)
    dr = torch.autograd.grad(r, reference, [x.double() for x in weights])
    rows = []
    tolerance = .025 if xs[0].dtype == torch.bfloat16 else .003
    for label, x, y in zip(('out', 'keys', 'state', 'dq', 'dk', 'dv', 'dg', 'dbeta', 'dk0', 'ds0'),
                           (*a, *da), (*r, *dr)):
        x = x.double()
        d = x - y
        absolute = float(d.abs().max())
        relative = float(d.square().mean().sqrt() / y.square().mean().sqrt().clamp_min(1e-12))
        rows.append(dict(tensor=label, absolute=absolute, rmsrel=relative,
                         passed=bool(torch.isfinite(x).all()) and (absolute <= 3e-4 or relative <= tolerance)))
    return dict(case=name, tolerance=tolerance, absolute_tolerance=3e-4,
                rows=rows, passed=all(x['passed'] for x in rows))


@torch.no_grad()
def ranking_checks():
    # All keys are random distinct directions; nearest normalized row identifies
    # each retained atom independently of the kernel's physical-slot assignment.
    b, t, h, m, d = 1, 257, 2, 32, 64
    q, k, v = [torch.randn(b, t, h, d, device='cuda') for _ in range(3)]
    g = torch.full((b, t, h, d), -.005, device='cuda')
    beta = torch.full((b, t, h), .2, device='cuda')
    k0 = torch.randn(b, h, m, d, device='cuda')
    s0 = torch.zeros_like(k0)
    pool = F.normalize(torch.cat((k0, k.transpose(1, 2)), 2), dim=-1)
    rows = []
    for mode in ('ties_newest', 'retain_early', 'retain_initial'):
        scores = torch.zeros(b, t, h, device='cuda')
        initial = torch.zeros(b, h, m, device='cuda')
        if mode == 'retain_early':
            scores[:, :m] = 10
        elif mode == 'retain_initial':
            initial.fill_(10)
        _, (kept, _) = block_komp_fast(q, k, v, g, beta, (k0, s0), chunk_size=128,
                                     evict='learned', scores=scores, initial_scores=initial)
        found = (F.normalize(kept, dim=-1) @ pool.transpose(-1, -2)).argmax(-1).sort(-1).values
        expected = (torch.arange(t, t + m, device='cuda') if mode == 'ties_newest' else
                    torch.arange(m, 2 * m, device='cuda') if mode == 'retain_early' else
                    torch.arange(m, device='cuda')).expand(b, h, m)
        passed = torch.equal(found, expected)
        rows.append(dict(case=mode, passed=passed, retained_pool_rows=found.cpu().tolist(),
                         expected_pool_rows=expected.cpu().tolist(),
                         differs_from_fifo=mode != 'ties_newest'))
    return rows


def run_checks(model, stream):
    torch.manual_seed(1234)
    cases = []
    for dtype in (torch.float32, torch.bfloat16):
        t = 257  # crosses two complete chunks and a one-token tail
        xs = [torch.randn(1, t, 2, 64, device='cuda', dtype=dtype) for _ in range(3)]
        xs += [-torch.rand(1, t, 2, 64, device='cuda') * .02,
               torch.rand(1, t, 2, device='cuda') * 1.5,
               torch.randn(1, 2, 32, 64, device='cuda'),
               torch.randn(1, 2, 32, 64, device='cuda') * .1]
        scores = torch.randn(1, t, 2, device='cuda')
        initial = torch.randn(1, 2, 32, device='cuda')
        cases.append(compare(xs, scores, initial, 'random-learned-tail-' + str(dtype)))
    # Exercise the stable fallback and its seven gradients before long training.
    cases.append(compare(xs, scores, initial, 'forced-hybrid-bf16', backend='hybrid'))
    spread = [x.detach().float() for x in xs]
    spread[3] = torch.full_like(spread[3], -.005)
    spread[3][..., :32] = -.3  # cumulative relative spread exceeds 24
    cases.append(compare(spread, scores, initial, 'automatic-stable-fallback-fp32'))
    saved = stream.state_dict()
    ids = stream.next()[:130].cuda()[None]
    stream.load_state_dict(saved)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        hidden = model.model.embeddings(ids[:, :-1])
        hidden = model.model.layers[0].attn_norm(hidden)
        operands = list(model.model.layers[0].attn.operands(hidden))
    xs = [x[:, :, :2].contiguous() for x in operands[:5]] + [x[:, :2].contiguous() for x in operands[5:7]]
    scores, initial = operands[7][:, :, :2].contiguous(), operands[8][:, :2].contiguous()
    cases.append(compare(xs, scores, initial, 'real-first-layer-initial-ties-tail'))
    routing = ranking_checks()
    out = dict(status='passed' if all(c['passed'] for c in cases + routing) else 'failed',
               cases=cases, ranking=routing, gradient_scope='seven continuous operands; hard ranking detached',
               score_learning='separately checked through the model write-strength gate')
    (ROOT / 'evidence/operator-block.json').write_text(json.dumps(out, indent=2))
    assert out['status'] == 'passed', out
    print('GDN_SLOT_LEARNED_OPERATOR_PASS', flush=True)
