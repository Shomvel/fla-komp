"""Merge fraction per layer and threshold on the keys of a trained per-slot FIFO model (no coherence in training)."""
import sys, torch
import block_experiment as be
import experiment as base
from komp_slot.block_flex import block_komp_flex

CAPTURE = []
be.block_komp_fast = lambda q, k, v, g, beta, state, **kw: (CAPTURE.append((q, k, v, g, beta, state)),
                                                          block_komp_flex(q, k, v, g, beta, state, **kw))[1]
model = be.BlockLM('block_slot_c64', 731).cuda()
model.load_state_dict(torch.load(sys.argv[1], map_location='cpu', weights_only=True)['model'])
model.eval()
ids, _ = base.batch(base.load('confirm')[:64])
with torch.no_grad(), base.amp():
    model(ids.cuda() if not ids.is_cuda else ids)
    for li, (q, k, v, g, beta, state) in enumerate(CAPTURE):
        row = []
        for th in (.99, .95, .9, .8, .7, .6, .5):
            st = {'valid': ids != 0}
            block_komp_flex(q, k, v, g, beta, state, chunk_size=64, coherence=th, stats=st)
            row.append(f'{th}:{st["merged"]/st["tokens"]:.1%}')
        print('layer', li, ' '.join(row), flush=True)
