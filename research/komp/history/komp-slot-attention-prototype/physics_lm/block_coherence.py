"""Frozen coherence quality trials. Only harness instrumentation; all arms use flex FP32."""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import torch
import block_experiment as be

flex_module = importlib.import_module('komp_slot.block_flex')
FLEX = flex_module.block_komp_flex
ROOT = Path(os.environ['BLOCK_RUNS'])
THRESHOLDS = {'off': None, '99': .99, '95': .95, '80': .80}
VARIANTS = {}
for family in ('fifo', 'gate', 'learned'):
    for tag, theta in THRESHOLDS.items():
        kind = f'coh_{family}_{tag}'
        VARIANTS[kind] = dict(theta=theta, family=family)
        be.VARIANTS[kind] = dict(chunk=64, decay='slot', evict='learned' if family == 'learned' else 'fifo',
                                 score_gate=family != 'fifo')
be.TRITON_LEARNED = True  # Route learned through the same replacement below; never call fast kernel.
STATE = dict(kind=None, valid=None, layer=0, training_run=False, step=0, total=2231,
             collect=False, windows={}, train_tokens=0, diagnostic=False, diag_rows=[], diag_stats=[])
OLD_INIT, OLD_FORWARD = be.BlockLM.__init__, be.BlockLM.forward
OLD_SOURCES, OLD_META = be.source_hashes, be.metadata


def init(self, kind, seed):
    STATE['kind'] = kind
    OLD_INIT(self, kind, seed)


def forward(self, ids):
    STATE.update(kind=self.variant, valid=ids != 0, layer=0, collect=False)
    if STATE['training_run'] and self.training and torch.is_grad_enabled():
        STATE['step'] += 1
        STATE['train_tokens'] += int((ids != 0).sum())
        width = math.ceil(STATE['total'] * .1)
        step = STATE['step']
        window = 'first10pct' if step <= width else 'last10pct' if step > STATE['total']-width else None
        STATE.update(collect=window is not None, window=window)
    out = OLD_FORWARD(self, ids)
    assert STATE['layer'] == be.base.L
    return out


def state_rows(keys, coef, layer, start, n, sample_ids):
    # Keep the production FP32 state, evaluate its geometry in FP64 for a stable diagnostic.
    keys, coef = keys.detach().double().cpu(), coef.detach().double().cpu()
    cos = keys @ keys.transpose(-1, -2)
    gram = ((cos-1)/.5).exp() + 1e-4 * torch.eye(keys.shape[-2], dtype=torch.float64)
    eig = torch.linalg.eigvalsh(gram)
    assert bool((eig[..., 0] > 0).all())
    cond = eig[..., -1] / eig[..., 0]
    pairs = (cos > .95).triu(1).sum((-2, -1))
    norms = coef.norm(dim=(-2, -1))
    for bi, sample in enumerate(sample_ids):
        for head in range(keys.shape[1]):
            STATE['diag_rows'].append(dict(sample=sample, layer=layer, head=head, block_start=start,
                block_length=n, partial=n < 64, condition=float(cond[bi, head]),
                cosine_pairs_gt95=int(pairs[bi, head]), coefficient_frobenius=float(norms[bi, head])))


def route(q, k, v, g, beta, state, **kw):
    layer = STATE['layer']; STATE['layer'] += 1
    theta = VARIANTS[STATE['kind']]['theta']
    collect = STATE['collect'] or STATE['diagnostic']
    stats = {'valid': STATE['valid']} if collect and theta is not None else None
    captured = []
    if STATE['diagnostic']:
        # Read locals after physical-slot assignment, without replacing any tensor operation.
        target = flex_module._flex.__code__
        lines = Path(target.co_filename).read_text().splitlines()
        capture_line = next(i+1 for i,s in enumerate(lines) if 'born = torch.gather(torch.gather(allborn' in s)
        def trace(frame, event, arg):
            if frame.f_code is not target:
                return None
            if event == 'line' and frame.f_lineno == capture_line:
                x = frame.f_locals
                captured.append((x['K'].detach().clone(), x['S'].detach().clone(), x['c0'], x['n']))
            return trace
        previous = sys.gettrace()
        assert previous is None
        sys.settrace(trace)
    try:
        out = FLEX(q, k, v, g, beta, state, coherence=theta, stats=stats, **kw)
    finally:
        if STATE['diagnostic']: sys.settrace(previous)
    assert out[0].dtype == torch.float32 and out[1][1].dtype == torch.float32
    if collect:
        if theta is None: stats = dict(tokens=int(STATE['valid'].sum())*q.shape[2], merged=0)
        if STATE['collect']:
            row = STATE['windows'].setdefault(STATE['window'], {}).setdefault(str(layer), dict(tokens=0, merged=0, calls=0))
            for key in ('tokens', 'merged'): row[key] += stats[key]
            row['calls'] += 1
        if STATE['diagnostic']:
            assert len(captured) == math.ceil(q.shape[1]/64)
            STATE['diag_stats'].append(dict(layer=layer, tokens=stats['tokens'], merged=stats['merged']))
            for keys, coef, start, n in captured:
                state_rows(keys, coef, layer, start, n, STATE['sample_ids'])
    return out


be.BlockLM.__init__, be.BlockLM.forward = init, forward
be.block_komp_fast = route
be.block_komp_flex = route


def sources():
    result = OLD_SOURCES()
    result['physics_lm/block_coherence.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def metadata(model):
    return dict(OLD_META(model), implementation='block_flex FP32 for every arm; BF16 surrounding backbone',
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        coherence=VARIANTS[model.variant]['theta'], coherence_family=VARIANTS[model.variant]['family'],
        statistics='nonpadding token-head weighted; first/last ceil(2231*.1)=224 optimizer steps only',
        selection_metric='E1 seven-seed confirm mean_content_ce; best enabled threshold, off tested separately',
        source_sha256=sources())


be.source_hashes, be.metadata = sources, metadata


def preflight():
    be.KINDS = list(VARIANTS)
    be.preflight()
    # Additional future-gradient check in each threshold/family, plus the training-only counter contract.
    sample = be.base.load('train')[:2]
    ids, _ = be.base.batch(sample)
    checks = []
    for kind in VARIANTS:
        model = be.BlockLM(kind, 731).cuda()
        STATE.update(valid=ids != 0, layer=0)
        x = torch.randn(2, 129, be.base.D, device='cuda', requires_grad=True)
        STATE['valid'] = torch.ones(2, 129, dtype=torch.bool, device='cuda')
        with be.base.amp(): out = model.layers[0].mix(x)
        grad = torch.autograd.grad(out[:, 63].float().square().sum(), x)[0]
        assert torch.count_nonzero(grad[:, 64:]).item() == 0
        STATE.update(training_run=True, step=0, train_tokens=0, windows={})
        with torch.no_grad(), be.base.amp(): model(ids)
        assert STATE['step'] == 0
        with be.base.amp(): model(ids)
        assert STATE['step'] == 1 and len(STATE['windows']['first10pct']) == 4
        assert all(r['tokens'] == int((ids != 0).sum())*be.base.H for r in STATE['windows']['first10pct'].values())
        STATE['training_run'] = False
        # Exercise the exact E3 observation path before spending a training budget.
        real_ids = torch.tensor([sample[0]['input_ids']], device='cuda')
        STATE.update(diagnostic=True, sample_ids=[0], diag_rows=[], diag_stats=[])
        model.eval()
        with torch.no_grad(), be.base.amp(): observed = model(real_ids)
        STATE['diagnostic'] = False
        with torch.no_grad(), be.base.amp(): plain = model(real_ids)
        torch.testing.assert_close(observed, plain, atol=0, rtol=0)
        assert len(STATE['diag_rows']) == 4*2*math.ceil(real_ids.shape[1]/64)
        checks.append(dict(kind=kind, future_gradient=0, train_only_masked_statistics=True,
                           diagnostic_trace_bitwise_neutral=True))
        del model, x, grad, out; torch.cuda.empty_cache()
    ticket = json.loads((ROOT/'preflight.json').read_text())
    ticket['additional_checks'] = checks
    (ROOT/'preflight.json').write_text(json.dumps(ticket, indent=2))


def train(kind, seed):
    ticket = json.loads((ROOT/'preflight.json').read_text())
    assert ticket['status'] == 'passed' and ticket['source_sha256'] == sources()
    assert kind in ticket['variants'] and len(ticket['additional_checks']) == len(VARIANTS)
    STATE.update(training_run=True, step=0, train_tokens=0, windows={})
    be.base.train(kind, seed, model_factory=be.BlockLM, result_root=ROOT/'results',
                  config_extra=metadata, preflight_path=ROOT/'preflight.json')
    STATE['training_run'] = False
    assert STATE['step'] == 2231 and STATE['train_tokens'] == 10001108
    for window in STATE['windows'].values():
        assert len(window) == 4
        for r in window.values():
            assert r['calls'] == 224 and r['tokens'] > 0
            r['fraction'] = r['merged']/r['tokens']
    (ROOT/f'{kind}-seed{seed}-coherence.json').write_text(json.dumps(dict(kind=kind, seed=seed,
        coherence=VARIANTS[kind]['theta'], status='completed', steps=STATE['step'],
        training_tokens=STATE['train_tokens'], denominator='nonpadding token-heads',
        windows=STATE['windows']), indent=2))


def diagnose(kind, seed):
    assert seed == 731
    folder = ROOT/'results'/f'{kind}-seed{seed}'
    checkpoint = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
    assert checkpoint['config']['source_sha256'] == sources()
    model = be.BlockLM(kind, seed).cuda(); model.load_state_dict(checkpoint['model'], strict=True); model.eval()
    rows = be.base.load('confirm')[:64]
    STATE.update(diagnostic=True, diag_rows=[], diag_stats=[])
    # Group equal real lengths; no padding, no streaming reset, original chunk boundaries retained.
    lengths = sorted(set(len(r['input_ids']) for r in rows))
    for length in lengths:
        indexes = [i for i,r in enumerate(rows) if len(r['input_ids']) == length]
        ids = torch.tensor([rows[i]['input_ids'] for i in indexes], device='cuda')
        assert bool((ids != 0).all())
        STATE['sample_ids'] = indexes
        with torch.no_grad(), be.base.amp(): traced = model(ids)
        STATE['diagnostic'] = False
        with torch.no_grad(), be.base.amp(): plain = model(ids)
        torch.testing.assert_close(traced, plain, atol=0, rtol=0)
        STATE['diagnostic'] = True
    STATE['diagnostic'] = False
    diag = dict(kind=kind, seed=seed, coherence=VARIANTS[kind]['theta'], status='passed',
        samples=64, tasks={t:sum(r['task']==t for r in rows) for t in ('depo','brevo')},
        sample_ids=[r.get('id',i) for i,r in enumerate(rows)],
        input_lengths=[len(r['input_ids']) for r in rows],
        note='Full teacher-forced real documents; no padding. Every block end including final partial block; FP32 frozen states, Gram condition computed in FP64. Observational trace is bitwise output-neutral.',
        rows=STATE['diag_rows'], merge_counts=STATE['diag_stats'])
    (ROOT/f'{kind}-seed{seed}-diagnostic.json').write_text(json.dumps(diag, indent=2))
    print('DIAGNOSTIC PASSED', kind, len(diag['rows']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--preflight', action='store_true'); p.add_argument('--diagnose', action='store_true')
    p.add_argument('--kind', choices=VARIANTS); p.add_argument('--seed', type=int, default=731); args = p.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.preflight: preflight()
    elif args.diagnose: diagnose(args.kind, args.seed)
    else: train(args.kind, args.seed)
