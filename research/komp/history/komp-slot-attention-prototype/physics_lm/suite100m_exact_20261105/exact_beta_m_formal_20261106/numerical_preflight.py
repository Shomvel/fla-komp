"""Representative new-shape CUDA checks; not an all-stress precision claim."""
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import torch

ROOT = Path(os.environ['VARIANT_ROOT'])
sys.path.insert(0, str(ROOT / 'snapshot/suite100m'))
from beta_m_models import LM, BETA0, SLOTS, ALPHA0, audit_initialization
from models import exact_komp, LM as OriginalLM
from reference import block_komp_token_reference
import train as base

checks = []


def check(name, actual, expected, low=False):
    a, b = actual.detach().double(), expected.detach().double()
    difference = a - b
    rms = float(difference.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-6))
    maximum = float(difference.abs().max() / b.abs().max().clamp_min(1.))
    row = dict(name=name, relative_rms=rms, normalized_max=maximum,
               rms_tolerance=.025 if low else .005, max_tolerance=.03 if low else .005)
    checks.append(row)
    assert torch.isfinite(a).all() and rms <= row['rms_tolerance'] and maximum <= row['max_tolerance'], row


def differential(inputs, label):
    inputs = [x.detach().requires_grad_() for x in inputs]
    gold = [x.detach().double().requires_grad_() for x in inputs]
    q, k, v, g, beta, keys, state = inputs
    outputs = exact_komp(q, k, v, g, beta, (keys, state), compute_chunk=16)[0:2]
    want = block_komp_token_reference(*gold[:5], tuple(gold[5:]), chunk_size=1, evict='fifo')
    actual = (outputs[0], *outputs[1])
    expected = (want[0], *want[1])
    weights = [torch.randn_like(x.float()) for x in actual]
    da = torch.autograd.grad(actual, inputs, weights)
    de = torch.autograd.grad(expected, gold, [x.double() for x in weights])
    for name, a, b in zip(('output', 'keys', 'state', 'dq', 'dk', 'dv', 'dg', 'dbeta', 'dk0', 'ds0'), (*actual, *da), (*expected, *de)):
        check(label + '/' + name, a, b, q.dtype == torch.bfloat16)
    print('NUMERICAL_PASS', label, flush=True)


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(812)
    audit = audit_initialization()
    (ROOT / 'evidence/initialization-difference.json').write_text(json.dumps(audit, indent=2) + '\n')
    for dtype in (torch.float32, torch.bfloat16):
        q, k, v = [torch.randn(1, 65, 1, 64, device='cuda', dtype=dtype) for _ in range(3)]
        keys = torch.linalg.qr(torch.randn(64, SLOTS, device='cuda')).Q.T[None, None].contiguous()
        state = torch.randn(1, 1, SLOTS, 64, device='cuda') * .1
        g = torch.full((1, 65, 1, SLOTS), math.log(ALPHA0), device='cuda')
        beta = torch.full((1, 65, 1), BETA0, device='cuda')
        differential([q, k, v, g, beta, keys, state], str(dtype) + '/random65')
    model = LM('exact', 731).cuda()
    model.eval()
    corpus = base.Corpus(os.environ['GROUP_TASK'], 'train', 'final')
    ids, mask = base.batch([corpus.row(0)])
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        layer = model.layers[0]
        x = layer.n1(model.embed(ids[:, :65]))
        mixer = layer.mix
        q, k, v = [conv(proj(x))[0].reshape(1, x.shape[1], 8, 64)[:, :, :1].contiguous()
                   for conv, proj in zip(mixer.convs, (mixer.q, mixer.k, mixer.v))]
    n = q.shape[1]
    differential([q, k, v, torch.full((1, n, 1, SLOTS), math.log(ALPHA0), device='cuda'),
                  torch.full((1, n, 1), BETA0, device='cuda'), mixer.keys.detach()[None, :1].contiguous(),
                  torch.zeros(1, 1, SLOTS, 64, device='cuda')], 'real_first_layer')
    if SLOTS == 32:
        original = OriginalLM('exact', 731).cuda().eval()
        original.load_state_dict(model.state_dict())
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            torch.testing.assert_close(model(ids[:, :129]), original(ids[:, :129]), atol=0, rtol=0)
        checks.append(dict(name='M32_adapter_original_model_forward', max_abs=0))
        del original
    # A 513-token call enters hierarchical state scan; split calls use serial scan.
    q, k, v = [torch.randn(1, 513, 1, 64, device='cuda') for _ in range(3)]
    g = torch.full((1, 513, 1, SLOTS), math.log(ALPHA0), device='cuda')
    beta = torch.full((1, 513, 1), BETA0, device='cuda')
    keys = mixer.keys.detach()[None, :1].contiguous()
    state = torch.zeros(1, 1, SLOTS, 64, device='cuda')
    inputs = [x.requires_grad_() for x in (q, k, v, g, beta, keys, state)]
    whole, ws, _ = exact_komp(*inputs[:5], tuple(inputs[5:]))
    first, fs, _ = exact_komp(*(x[:, :256] for x in inputs[:5]), tuple(inputs[5:]))
    last, ls, _ = exact_komp(*(x[:, 256:] for x in inputs[:5]), fs, start_position=256)
    streamed = (torch.cat((first, last), 1), *ls)
    full = (whole, *ws)
    weights = [torch.randn_like(x) for x in full]
    grads = torch.autograd.grad(full, inputs, weights)
    stream_grads = torch.autograd.grad(streamed, inputs, weights)
    for i, (a, b) in enumerate(zip((*full, *grads), (*streamed, *stream_grads))):
        check(f'hierarchical_vs_streamed/{i}', a, b)
    del model
    result = dict(status='passed', beta0=BETA0, slots=SLOTS, checks=checks,
                  scope='Representative random and real activation forward/state/gradient checks plus hierarchical streaming consistency; original near-collinear FP32 stress failure retained.')
    (ROOT / 'evidence/numerical.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        (ROOT / 'evidence/numerical-failure.json').write_text(json.dumps(dict(status='failed', traceback=traceback.format_exc(), checks=checks), indent=2) + '\n')
        raise
