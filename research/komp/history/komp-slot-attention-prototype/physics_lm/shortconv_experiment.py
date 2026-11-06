"""Paired short-convolution ablation, preserving the existing mixer equations."""
import argparse
import hashlib
import json
from pathlib import Path

import experiment as base
import decay_experiment as decay
import torch
from torch import nn
import torch.nn.functional as F
from fla.modules.causal_conv1d import ShortConvolution

ROOT = Path(__file__).resolve().parent / 'shortconv_runs'
KINDS = ('komp_slot', 'gsa', 'gdn')


class ConvProjection(nn.Module):
    def __init__(self, linear, seed):
        super().__init__()
        self.weight = linear.weight
        self.register_parameter('bias', linear.bias)
        # Identical convolution initialization across architectures, without
        # shifting any existing projection, gate, embedding, or FFN weights.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.conv = ShortConvolution(base.D, 4, bias=False, activation='silu', backend='triton')

    def forward(self, x):
        projected = F.linear(x, self.weight, self.bias)
        return self.conv(projected)[0]


def make_model(kind, seed):
    architecture, setting = kind.rsplit('_', 1)
    assert architecture in KINDS and setting in ('conv', 'plain')
    model = decay.DecayLM('gsa_slot', seed) if architecture == 'komp_slot' else base.LM(architecture, seed)
    model.shortconv_architecture = architecture
    model.use_short_conv = setting == 'conv'
    if model.use_short_conv:
        for i, layer in enumerate(model.layers):
            for j, name in enumerate(('q', 'k', 'v')):
                setattr(layer.mix, name, ConvProjection(getattr(layer.mix, name), seed + 5000 + i * 3 + j))
    return model


def hashes(model, selection):
    return {name: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
            for name, p in model.named_parameters() if selection(name)}


def sources():
    root = Path(base.__file__).parents[1]
    paths = [Path(__file__), Path(base.__file__), Path(decay.__file__), root / 'benchmark_fla.py',
             root / 'physics_lm/shortconv_launch.py', root / 'physics_lm/validate_data.py',
             root / 'physics_lm/physics_port.py',
             *sorted((root / 'komp_slot').glob('*.py'))]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def metadata(model):
    architecture = model.shortconv_architecture
    return {
        'mixer': architecture, 'use_short_conv': model.use_short_conv,
        'short_conv': {'width': 4, 'paths': ['q', 'k', 'v'], 'bias': False, 'activation': 'silu',
                       'backend': 'FLA Triton', 'initialization': 'FLA default, named seeds'},
        'gate_recipe': {'komp_slot': 'GSA slot decay /8; independent beta',
                        'gsa': 'logsigmoid slot decay /1, independent slot initialization',
                        'gdn': 'sigmoid scalar decay and beta, as previous mixer baseline'}[architecture],
        'weight_decay_policy': 'decay.bias/A_log/dt_bias=0, all other parameters=.1; same across arms',
        'geometry': 'frozen decay training archive, serial routes, IEEE solver',
        'nonconv_initial_sha256': hashes(model, lambda n: '.conv.' not in n),
        'conv_initial_sha256': hashes(model, lambda n: '.conv.' in n),
        'shortconv_source_sha256': sources(),
        'fla_commit': '97874af2400916e2885c33ea30559a70ff686b66',
        'brevo_scoring': 'ancestor closure and topological constraints from prompt graph only',
    }


def valid_brevo(ids, guess):
    boundary = ids.index(9997)
    edges = list(zip(ids[1:boundary:2], ids[2:boundary:2]))
    nodes = {ids[boundary + 1]}
    while True:
        more = nodes | {a for a, b in edges if b in nodes}
        if more == nodes:
            break
        nodes = more
    if not guess or guess[-1] != 9998:
        return False
    answer = guess[:-1]
    pos = {node: i for i, node in enumerate(answer)}
    return (len(answer) == len(nodes) and set(answer) == nodes
            and all(pos[a] < pos[b] for a, b in edges if b in nodes))


ORIGINAL_GENERATE = base.generate


def generate(model, rows, limit=128):
    summary, records = ORIGINAL_GENERATE(model, rows, limit)
    brevo = [r for r in rows if r['task'] == 'brevo'][:limit]
    generated = [r for r in records if r['task'] == 'brevo']
    assert len(brevo) == len(generated) == limit
    for row, record in zip(brevo, generated):
        independent = valid_brevo(row['input_ids'], record['guess'])
        assert independent == record['valid'], 'Prompt-only and legacy verifier disagree'
        record['valid'] = independent
    summary['brevo']['valid'] = sum(r['valid'] for r in generated) / limit
    return summary, records


base.generate = generate


def check_model_causality(model, rows):
    ids, _ = base.batch(rows[:2])
    checks = []
    model.eval()
    with torch.no_grad(), base.amp():
        original = model(ids)
        for cut in (1, 3, 4, 15, 16, 17, 31, 32, 33, 127, 128, 129, 191):
            changed = ids.clone()
            changed[:, cut:] = torch.randint(base.VOCAB, changed[:, cut:].shape, device='cuda')
            error = float((model(changed)[:, :cut] - original[:, :cut]).abs().max())
            assert error == 0, (model.shortconv_architecture, cut, error)
            checks.append({'cut': cut, 'prefix_max_abs': error})
    model.train()
    return checks


def convolution_checks():
    checks = []
    for dtype in (torch.float32, torch.bfloat16):
        conv = ShortConvolution(128, 4, bias=False, activation='silu', backend='triton').cuda()
        x = torch.randn(2, 65, 128, device='cuda', dtype=dtype, requires_grad=True)
        y, _ = conv(x)
        refx = x.detach().float().requires_grad_()
        refw = conv.weight.detach().clone().requires_grad_()
        ref = F.silu(F.conv1d(F.pad(refx.transpose(1, 2), (3, 0)), refw, groups=128).transpose(1, 2))
        cotangent = torch.randn_like(y)
        grads = torch.autograd.grad(y, (x, conv.weight), cotangent, retain_graph=True)
        refs = torch.autograd.grad(ref, (refx, refw), cotangent.float())
        tolerance = 2e-5 if dtype == torch.float32 else .015
        for name, actual, expected in [('output', y, ref), ('dx', grads[0], refs[0]), ('dw', grads[1], refs[1])]:
            relative = float((actual.float() - expected).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8))
            assert relative < tolerance, (dtype, name, relative)
            checks.append({'dtype': str(dtype), 'name': name, 'rms_relative': relative, 'tolerance': tolerance})
        grad = torch.autograd.grad(y[:, 31].sum(), x)[0]
        assert int(torch.count_nonzero(grad[:, 32:])) == 0
        with torch.no_grad():
            altered = x.detach().clone()
            altered[:, 32:] = torch.randn_like(altered[:, 32:])
            assert torch.equal(conv(altered)[0][:, :32], y[:, :32])
            prefix, state = conv(x.detach()[:, :32], output_final_state=True)
            steps = [prefix]
            for i in range(32, 65):
                out, state = conv(x.detach()[:, i:i + 1], cache=state, output_final_state=True)
                steps.append(out)
            streamed = torch.cat(steps, dim=1)
            torch.testing.assert_close(streamed, y, atol=1e-6 if dtype == torch.float32 else .008,
                                       rtol=1e-5 if dtype == torch.float32 else .008)
        checks.append({'dtype': str(dtype), 'future_gradient': 0, 'suffix_intervention': 0,
                       'stream_vs_full_max_abs': float((streamed - y).abs().max())})
    return checks


def preflight():
    from validate_data import check
    ROOT.mkdir(parents=True, exist_ok=True)
    seen = set()
    for split in ('train', 'val', 'test', 'confirm'):
        rows = base.load(split)
        for row in rows:
            check(row)
            key = tuple(row['input_ids'])
            assert key not in seen
            seen.add(key)
    sample = base.load('train')[:4]
    report = {'status': 'running', 'data_checked': len(seen), 'conv_checks': convolution_checks(), 'models': {}}
    shared = []
    conv_hashes = []
    for architecture in KINDS:
        paired = []
        for setting in ('plain', 'conv'):
            kind = architecture + '_' + setting
            model = make_model(kind, 731).cuda()
            base.check_slot_diversity(model, architecture)
            paired.append(hashes(model, lambda n: '.conv.' not in n))
            shared.append(hashes(model, lambda n: '.conv.' not in n and not any(s in n for s in ('.decay.', '.beta.', '.keys'))))
            if setting == 'conv':
                conv_hashes.append(hashes(model, lambda n: '.conv.' in n))
            causal = check_model_causality(model, sample)
            opt = decay.optimizer(model)
            losses = []
            for _ in range(16):
                opt.zero_grad(set_to_none=True)
                with base.amp():
                    loss = base.loss_items(model, sample)[0].mean()
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
                assert bool(torch.isfinite(norm)) and bool(torch.isfinite(loss))
                if setting == 'conv':
                    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().max()) > 0
                               for n, p in model.named_parameters() if '.conv.' in n)
                opt.step()
                losses.append(float(loss.detach()))
            assert losses[-1] < losses[0], (kind, losses)
            report['models'][kind] = {'losses': losses, 'causality': causal,
                                     'params': sum(p.numel() for p in model.parameters())}
            print('PREFLIGHT PASS', kind, losses[0], losses[-1], flush=True)
            del model, opt
            torch.cuda.empty_cache()
        assert paired[0] == paired[1], architecture + ': nonconv initialization mismatch'
    assert all(x == shared[0] for x in shared)
    assert all(x == conv_hashes[0] for x in conv_hashes)
    report.update(status='passed', source_sha256=sources(), shared_initialization='passed', paired_initialization='passed')
    (ROOT / 'preflight.json').write_text(json.dumps(report, indent=2))


def train(kind, seed):
    ticket = json.loads((ROOT / 'preflight.json').read_text())
    assert ticket['status'] == 'passed' and ticket['source_sha256'] == sources()
    base.train(kind, seed, model_factory=make_model, result_root=ROOT / 'results',
               optimizer_factory=decay.optimizer, config_extra=metadata, preflight_path=ROOT / 'preflight.json')
    dest = ROOT / 'results' / f'{kind}-seed{seed}'
    saved = torch.load(dest / 'model.pt', map_location='cpu', weights_only=True)
    model = make_model(kind, seed).cuda()
    model.load_state_dict(saved['model'], strict=True)
    checks = check_model_causality(model, base.load('confirm')[:4])
    (dest / 'trained-causality.json').write_text(json.dumps({'status': 'passed', 'checks': checks}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--kind', choices=[k + '_' + s for k in KINDS for s in ('plain', 'conv')])
    parser.add_argument('--seed', type=int, choices=(731, 732), default=731)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(9041)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.preflight:
        preflight()
    else:
        train(args.kind, args.seed)
