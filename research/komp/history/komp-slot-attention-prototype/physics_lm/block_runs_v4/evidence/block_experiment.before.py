"""Block-KOMP Depo/Brevo trials on the frozen corpus; same backbone, data, LR recipe as the KOMP arms."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import experiment as base
import torch
from torch import nn
import torch.nn.functional as F
from komp_slot.block_fast import block_komp_fast
from komp_slot.block_flex import block_komp_flex

import os
# Triton indexed-dictionary path for learned eviction (block_flex otherwise).
TRITON_LEARNED = os.environ.get('BLOCK_TRITON_LEARNED', '1') == '1'
ROOT = Path(__file__).resolve().parent/os.environ.get('BLOCK_RUNS', 'block_runs')
VARIANTS = {'block_c16': {'chunk': 16}, 'block_c32': {'chunk': 32}, 'block_c64': {'chunk': 64},
            # GSA-style data-dependent per-slot decay, as the best per-token KOMP arm (gsa_slot).
            'block_slot_c64': {'chunk': 64, 'decay': 'slot'},
            # Importance-based eviction (one-shot top-M of M+C at chunk ends) and projectron writes.
            'block_slot_res_c64': {'chunk': 64, 'decay': 'slot', 'evict': 'residual'},
            'block_slot_coef_c64': {'chunk': 64, 'decay': 'slot', 'evict': 'coef_norm'},
            'block_slot_learn_c64': {'chunk': 64, 'decay': 'slot', 'evict': 'learned'},
            'block_slot_proj_c64': {'chunk': 64, 'decay': 'slot', 'projectron': True}}


class BlockMixer(base.Mixer):
    """KOMP mixer with block-synchronous eviction; shared parameters copied from the KOMP arm."""

    def __init__(self, original, cfg):
        super().__init__('komp')
        self.load_state_dict(original.state_dict())
        self.chunk = cfg['chunk']
        self.slot = cfg.get('decay') == 'slot'
        self.evict = cfg.get('evict', 'fifo')
        self.projectron = cfg.get('projectron', False)
        if self.evict == 'learned':
            # Learned importance: s_t ranks the atom written at t; 2*sigmoid(s_t) also scales its
            # write strength so s receives gradient (factor 1 at initialization).
            self.score = nn.Linear(base.D, base.H)
            nn.init.zeros_(self.score.weight)
            nn.init.zeros_(self.score.bias)
            self.initial_score = nn.Parameter(torch.zeros(base.H, base.M))
        if self.slot:
            # Same initial function as the scalar arms: logsigmoid(bias)/8 = log .98 on every slot.
            self.decay = nn.Linear(base.D, base.H*base.M)
            nn.init.zeros_(self.decay.weight)
            a = .98**8
            nn.init.constant_(self.decay.bias, math.log(a/(1-a)))

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = [f(x).reshape(b, t, base.H, base.K) for f in (self.q, self.k, self.v)]
        with torch.autocast('cuda', enabled=False):
            # GDN-style log decay; identical initial alpha=.98 to the sigmoid KOMP arm.
            g = F.logsigmoid(F.linear(x.float(), self.decay.weight, self.decay.bias))
            if self.slot:
                g = (g/8.).reshape(b, t, base.H, base.M)
            beta = torch.sigmoid(F.linear(x.float(), self.beta.weight, self.beta.bias))
        keys = self.keys[None].expand(b, base.H, base.M, base.K).contiguous()
        values = torch.zeros(b, base.H, base.M, base.K, device=x.device)
        if self.evict == 'fifo' and not self.projectron:
            o, _ = block_komp_fast(q, k, v, g, beta, (keys, values), chunk_size=self.chunk)
        elif self.evict == 'learned' and not self.projectron and TRITON_LEARNED:
            with torch.autocast('cuda', enabled=False):
                scores = F.linear(x.float(), self.score.weight, self.score.bias)
            beta = beta*2*torch.sigmoid(scores)
            initial = self.initial_score[None].expand(b, base.H, base.M)
            o, _ = block_komp_fast(q, k, v, g, beta, (keys, values), chunk_size=self.chunk, evict='learned',
                                   scores=scores, initial_scores=initial)
        else:
            scores = initial = None
            if self.evict == 'learned':
                with torch.autocast('cuda', enabled=False):
                    scores = F.linear(x.float(), self.score.weight, self.score.bias)
                beta = beta*2*torch.sigmoid(scores)
                initial = self.initial_score[None].expand(b, base.H, base.M)
            o, _ = block_komp_flex(q, k, v, g, beta, (keys, values), chunk_size=self.chunk, evict=self.evict,
                                   projectron=self.projectron, scores=scores, initial_scores=initial)
        y = self.norm(o.float()).to(x.dtype).reshape(b, t, base.D)
        return self.out(y*F.silu(self.gate(x)))


class BlockLM(base.LM):
    def __init__(self, kind, seed):
        super().__init__('komp', seed)
        self.variant = kind
        for layer in self.layers:
            layer.mix = BlockMixer(layer.mix, VARIANTS[kind])


def source_hashes():
    root = Path(base.__file__).parents[1]
    sources = [Path(__file__), Path(base.__file__), *sorted((root/'komp_slot').glob('*.py'))]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}


def metadata(model):
    cfg = VARIANTS[model.variant]
    return {'block_variant': model.variant, 'block_config': cfg,
            'eviction': cfg.get('evict', 'fifo')+': top-M of M+C kept at chunk ends (ties to newer), evicted atoms '
            'jointly projected', 'implementation': 'block_fast Triton' if cfg.get('evict', 'fifo') == 'fifo' or (
                cfg.get('evict') == 'learned' and TRITON_LEARNED and not cfg.get('projectron')) else 'block_flex PyTorch',
            'projectron': 'beta *= detached novelty (Cholesky pivot of the union Gram)'
            if cfg.get('projectron') else None,
            'decay': ('per ring slot g=logsigmoid(linear)/8' if cfg.get('decay') == 'slot' else 'g=logsigmoid(linear)')
            + ', initial alpha .98 everywhere',
            'temperature': .5, 'ridge': 1e-4, 'source_sha256': source_hashes()}


def preflight():
    ROOT.mkdir(exist_ok=True)
    sample = base.load('train')[:4]
    ids, _ = base.batch(sample)
    report = {'variants': {}}
    for kind in KINDS:
        model = BlockLM(kind, 731).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with torch.no_grad(), base.amp():
            before = model(ids)
            changed = ids.clone()
            changed[:, 40:] = (changed[:, 40:]+7) % base.VOCAB
            after = model(changed)
            torch.testing.assert_close(before[:, :40], after[:, :40], atol=1e-5, rtol=1e-5)
        losses = []
        for _ in range(16):
            opt.zero_grad(set_to_none=True)
            with base.amp():
                loss = base.loss_items(model, sample)[0].mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert bool(torch.isfinite(norm))
            opt.step()
            losses.append(float(loss.detach()))
        assert losses[-1] < losses[0], (kind, losses)
        report['variants'][kind] = {'losses': losses, 'causality': 'passed'}
        print('PASS PREFLIGHT', kind, losses[0], losses[-1], flush=True)
        del model, opt
        torch.cuda.empty_cache()
    report['status'] = 'passed'
    report['source_sha256'] = source_hashes()
    (ROOT/'preflight.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--kinds', default=','.join(VARIANTS))
    p.add_argument('--kind', choices=tuple(VARIANTS))
    p.add_argument('--seed', type=int, default=731)
    args = p.parse_args()
    KINDS = args.kinds.split(',')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.preflight:
        preflight()
    else:
        ticket = json.loads((ROOT/'preflight.json').read_text())
        assert ticket['status'] == 'passed' and ticket['source_sha256'] == source_hashes(), 'Preflight does not match source'
        root = ROOT/'results'
        root.mkdir(exist_ok=True, parents=True)
        base.train(args.kind, args.seed, model_factory=BlockLM, result_root=root, config_extra=metadata,
                   preflight_path=ROOT/'preflight.json')
