import json, torch
from benchmark_block import make, measure
data, slots = make(2, 2048, 4, 64, 64, 32)
rows = []
for label, name, c in (('gdn', 'gdn', 64), ('gsa', 'gsa', 64), ('block_c64', 'block', 64), ('block_c32', 'block', 32)):
    r = {'operator': label, 'chunk': c, **measure(name, data[name], slots, c, 10)}
    rows.append(r); print('BENCH', json.dumps(r), flush=True)
data, slots = make(1, 8192, 64, 128, 128, 64)
for label, name, c in (('gdn', 'gdn', 64), ('block_c64', 'block', 64)):
    r = {'operator': label, 'chunk': c, 'config': '8k', **measure(name, data[name], slots, c, 2)}
    rows.append(r); print('BENCH', json.dumps(r), flush=True)
open('evidence/block-benchmark-robust.json', 'w').write(json.dumps(rows, indent=2))
