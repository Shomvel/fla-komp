"""Option B quality probe: Block-KOMP with eviction period C=128 against C=64, Depo/Brevo, same recipe as
block_experiment.py. Every arm runs on the FP32 PyTorch path (block_flex), so the two periods are compared
on one implementation (block_fast supports C <= 64)."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import block_experiment as be
from komp_slot.block_flex import block_komp_flex


def _flex(q, k, v, g, beta, state, chunk_size=64, temperature=.5, ridge=1e-4, evict='fifo', scores=None,
          initial_scores=None):
    return block_komp_flex(q, k, v, g, beta, state, chunk_size=chunk_size, temperature=temperature, ridge=ridge,
                           evict=evict, scores=scores, initial_scores=initial_scores)


be.block_komp_fast = _flex
be.VARIANTS.update({
    'flex_slot_c64': {'chunk': 64, 'decay': 'slot'},
    'flex_slot_c128': {'chunk': 128, 'decay': 'slot'},
    'flex_slot_learn_c64': {'chunk': 64, 'decay': 'slot', 'evict': 'learned'},
    'flex_slot_learn_c128': {'chunk': 128, 'decay': 'slot', 'evict': 'learned'},
})
_hashes, _meta = be.source_hashes, be.metadata
be.source_hashes = lambda: {**_hashes(), 'physics_lm/block_c128.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
be.metadata = lambda model: {**_meta(model), 'implementation': 'block_flex PyTorch FP32 (all arms; option B probe)'}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--kinds', default='flex_slot_c64,flex_slot_c128,flex_slot_learn_c64,flex_slot_learn_c128')
    p.add_argument('--kind', choices=tuple(be.VARIANTS))
    p.add_argument('--seed', type=int, default=731)
    args = p.parse_args()
    be.KINDS = args.kinds.split(',')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.preflight:
        be.preflight()
    else:
        ticket = json.loads((be.ROOT/'preflight.json').read_text())
        assert ticket['status'] == 'passed' and ticket['source_sha256'] == be.source_hashes(), 'Preflight does not match source'
        root = be.ROOT/'results'
        root.mkdir(exist_ok=True, parents=True)
        be.base.train(args.kind, args.seed, model_factory=be.BlockLM, result_root=root, config_extra=be.metadata,
                      preflight_path=be.ROOT/'preflight.json')
