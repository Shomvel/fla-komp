"""Exact C=1 KOMP on the unchanged, frozen Depo/Brevo training protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import experiment as base
from block_experiment import BlockMixer
from komp_slot.exact_komp import exact_komp

ROOT = Path(os.environ['EXACT_RUN_ROOT'])
VARIANTS = {
    'exact_scalar_fifo': {'chunk': 16},
    'exact_slot_fifo': {'chunk': 16, 'decay': 'slot'},
    'exact_slot_gate': {'chunk': 16, 'decay': 'slot', 'score_gate': True},
    'exact_slot_learned': {'chunk': 16, 'decay': 'slot', 'evict': 'learned'},
}


class ExactMixer(BlockMixer):
    reference = False

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = [f(x).reshape(b, t, base.H, base.K) for f in (self.q, self.k, self.v)]
        with torch.autocast('cuda', enabled=False):
            g = F.logsigmoid(F.linear(x.float(), self.decay.weight, self.decay.bias))
            if self.slot:
                g = (g / 8).reshape(b, t, base.H, base.M)
            beta = torch.sigmoid(F.linear(x.float(), self.beta.weight, self.beta.bias))
            scores = F.linear(x.float(), self.score.weight, self.score.bias) if self.score_gate else None
        keys = self.keys[None].expand(b, base.H, base.M, base.K).contiguous()
        values = torch.zeros(b, base.H, base.M, base.K, device=x.device)
        initial = self.initial_score[None].expand(b, base.H, base.M) if self.score_gate else None
        if self.reference:
            from exact_reference import block_komp_token_reference
            strength = beta * 2 * torch.sigmoid(scores) if self.score_gate else beta
            with torch.autocast('cuda', enabled=False):
                o, _ = block_komp_token_reference(
                    q.double(), k.double(), v.double(), g.double(), strength.double(),
                    (keys.double(), values.double()), chunk_size=1, evict=self.evict,
                    scores=scores, initial_scores=initial)
        else:
            o, _, _ = exact_komp(
                q, k, v, g, beta, (keys, values), eviction=self.evict,
                scores=scores, initial_scores=initial, score_write_gate=self.score_gate,
                compute_chunk=16, projection_dtype='float32')
        y = self.norm(o.float()).to(x.dtype).reshape(b, t, base.D)
        return self.out(y * F.silu(self.gate(x)))


class ExactLM(base.LM):
    def __init__(self, kind, seed):
        super().__init__('komp', seed)
        self.variant = kind
        for layer in self.layers:
            layer.mix = ExactMixer(layer.mix, VARIANTS[kind])


def hashes():
    root = Path(base.__file__).parents[1]
    files = [*sorted((root/'komp_slot').rglob('*.py')), Path(__file__), Path(base.__file__),
             Path(__file__).with_name('block_experiment.py'), Path(__file__).with_name('exact_reference.py')]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def metadata(model):
    cfg = VARIANTS[model.variant]
    return {'exact_variant': model.variant, 'exact_config': cfg, 'compression_tokens': 1,
            'compute_chunk': 16, 'projection_dtype': 'float32', 'temperature': .5, 'ridge': 1e-4,
            'routing': 'per-token FIFO' if cfg.get('evict') != 'learned' else 'per-token input-score top-M; ties newer',
            'score_gate': cfg.get('score_gate', False) or cfg.get('evict') == 'learned',
            'score_note': 'Hard selection detached; write gate supplies score gradient; initial score dormant.',
            'short_conv': False, 'recurrent_elements_per_head': 4096,
            'decay': 'logsigmoid/8 per physical slot' if cfg.get('decay') == 'slot' else 'logsigmoid scalar',
            'source_sha256': hashes()}


def preflight():
    import copy
    ROOT.mkdir(parents=True, exist_ok=True)
    splits = {s: base.load(s) for s in ('train', 'val', 'test', 'confirm')}
    assert sum(len(r['input_ids']) for r in splits['train']) == 10001108
    sample = splits['train'][:4]
    ids, _ = base.batch(sample)
    report = {'variants': {}, 'data_hashes_verified': list(splits), 'source_sha256': hashes()}
    for kind in VARIANTS:
        model = ExactLM(kind, 731).cuda()
        ref = copy.deepcopy(model)
        for layer in ref.layers:
            layer.mix.reference = True
        with torch.no_grad(), base.amp():
            actual = model(ids[:1, :33])
            expected = ref(ids[:1, :33])
        error = ((actual.float()-expected.float()).square().mean().sqrt() /
                 expected.float().square().mean().sqrt().clamp_min(1e-6)).item()
        assert error < .01, (kind, 'FP64 reference model output', error)
        del actual, expected, ref
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses, score_grads = [], []
        for step in range(16):
            opt.zero_grad(set_to_none=True)
            with base.amp():
                loss = base.loss_items(model, sample)[0].mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert bool(torch.isfinite(norm)) and bool(torch.isfinite(loss))
            if step == 0 and model.layers[0].mix.score_gate:
                for layer in model.layers:
                    grad = layer.mix.score.weight.grad
                    assert grad is not None and bool(torch.isfinite(grad).all()) and grad.abs().max() > 0
                    score_grads.append(float(grad.norm()))
            opt.step()
            losses.append(float(loss.detach()))
        assert losses[-1] < losses[0], (kind, losses)
        causal = []
        with torch.no_grad(), base.amp():
            before = model(ids)
            for cut in (15, 16, 17, 31, 32, 33, 63, 64, 65, 127):
                altered = ids.clone()
                altered[:, cut:] = (altered[:, cut:] + 7) % base.VOCAB
                after = model(altered)
                torch.testing.assert_close(before[:, :cut], after[:, :cut], atol=1e-5, rtol=1e-5)
                causal.append({'cut': cut, 'max_abs': float((before[:, :cut]-after[:, :cut]).abs().max())})
        # Actual training batch shape must compile and backpropagate before the full run.
        opt.zero_grad(set_to_none=True)
        with base.amp():
            loss = base.loss_items(model, splits['train'][:64])[0].mean()
        loss.backward()
        assert bool(torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)))
        report['variants'][kind] = {'reference_rms': error, 'losses': losses,
                                    'score_gradient_norms': score_grads, 'causality': causal, 'batch64': 'passed'}
        print('PREFLIGHT', kind, json.dumps(report['variants'][kind]), flush=True)
        del model, opt
        torch.cuda.empty_cache()
    report['status'] = 'passed'
    (ROOT/'preflight.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--kind', choices=VARIANTS)
    parser.add_argument('--seed', type=int, default=731)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.preflight:
        preflight()
    else:
        ticket = json.loads((ROOT/'preflight.json').read_text())
        assert ticket['status'] == 'passed' and ticket['source_sha256'] == hashes()
        assert args.kind in ticket['variants']
        base.train(args.kind, args.seed, model_factory=ExactLM, result_root=ROOT/'results',
                   config_extra=metadata, preflight_path=ROOT/'preflight.json')
