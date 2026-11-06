"""Coherence-admission trials: Block-KOMP per-slot FIFO with and without the coherence criterion (cosine
threshold), all arms on the FP32 PyTorch path (block_flex). Same recipe as block_experiment.py; merge counts
per layer are written beside each run."""
import argparse
import atexit
import hashlib
import json
from pathlib import Path

import torch
import block_experiment as be
from komp_slot.block_flex import block_komp_flex

COH = {'flex_slot_c64': None, 'flex_slot_coh99_c64': .99, 'flex_slot_coh95_c64': .95, 'flex_slot_coh80_c64': .80}
for kind in COH:
    be.VARIANTS[kind] = {'chunk': 64, 'decay': 'slot'}
STATE = {'coherence': None, 'calls': 0, 'stats': [], 'valid': None}
_forward = be.BlockLM.forward


def _lm_forward(self, ids, *a, **kw):
    STATE['valid'] = ids != 0   # right padding is token 0: merge counts cover real tokens only
    return _forward(self, ids, *a, **kw)


be.BlockLM.forward = _lm_forward


def _flex(q, k, v, g, beta, state, chunk_size=64, temperature=.5, ridge=1e-4, evict='fifo', scores=None,
          initial_scores=None):
    st = {'valid': STATE['valid']} if STATE['coherence'] is not None else None
    out = block_komp_flex(q, k, v, g, beta, state, chunk_size=chunk_size, temperature=temperature, ridge=ridge,
                          evict=evict, scores=scores, initial_scores=initial_scores, coherence=STATE['coherence'],
                          stats=st)
    if st is not None:
        if len(STATE['stats']) <= STATE['calls'] % 4:
            STATE['stats'].append([])
        STATE['stats'][STATE['calls'] % 4].append(st['merged']/max(1, st['tokens']))
        STATE['calls'] += 1
    return out


be.block_komp_fast = _flex
_hashes, _meta = be.source_hashes, be.metadata
be.source_hashes = lambda: {**_hashes(), 'physics_lm/block_coh.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
be.metadata = lambda model: {**_meta(model), 'implementation': 'block_flex PyTorch FP32 (all arms; coherence trials)',
                             'coherence': COH[model.variant]}


def dump(kind, seed):
    # Per layer (forward calls cycle through the 4 layers): merged fraction, early vs late in training.
    rows = STATE['stats']
    out = {f'layer{i}': {'first10pct': sum(r[:max(1, len(r)//10)])/max(1, len(r[:max(1, len(r)//10)])),
                         'last10pct': sum(r[-max(1, len(r)//10):])/max(1, len(r[-max(1, len(r)//10):])), 'calls': len(r)}
           for i, r in enumerate(rows)}
    (be.ROOT/f'{kind}-seed{seed}-coherence.json').write_text(json.dumps({'coherence': COH[kind], 'merged': out}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--kinds', default=','.join(COH))
    p.add_argument('--kind', choices=tuple(COH))
    p.add_argument('--seed', type=int, default=731)
    args = p.parse_args()
    be.KINDS = args.kinds.split(',')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.preflight:
        # Each arm's preflight runs with its own threshold (causality + loss decrease).
        orig = be.BlockLM.__init__

        def init(self, kind, seed):
            STATE['coherence'] = COH[kind]
            orig(self, kind, seed)
        be.BlockLM.__init__ = init
        be.preflight()
    else:
        STATE['coherence'] = COH[args.kind]
        ticket = json.loads((be.ROOT/'preflight.json').read_text())
        assert ticket['status'] == 'passed' and ticket['source_sha256'] == be.source_hashes(), 'Preflight does not match source'
        root = be.ROOT/'results'
        root.mkdir(exist_ok=True, parents=True)
        atexit.register(dump, args.kind, args.seed)
        be.base.train(args.kind, args.seed, model_factory=be.BlockLM, result_root=root, config_extra=be.metadata,
                      preflight_path=be.ROOT/'preflight.json')
