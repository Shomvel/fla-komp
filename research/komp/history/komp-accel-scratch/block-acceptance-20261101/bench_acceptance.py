"""Same-device operator timing, retaining individual samples and source snapshot."""
import json
from pathlib import Path
import torch
from benchmark_block import make, measure

dest = Path('evidence/benchmark.json')
report = {'rows': [], 'timing': 'CUDA graphs; compilation excluded; 3 independent measurements per row',
          'note': 'Equal carried-state elements/head. Block algorithm retains up to M+C atoms inside a chunk.'}
configs = [(2, 2048, 4, 64, 64, 32), (1, 8192, 64, 128, 128, 64), (64, 192, 8, 64, 64, 32)]
for cfg in configs:
    data, slots = make(*cfg)
    variants = [('gdn', 'gdn', 64), ('gsa', 'gsa', 64), ('block_c16', 'block', 16),
                ('block_c32', 'block', 32), ('block_c64', 'block', 64)]
    if cfg == configs[0]:
        variants.append(('per_token_komp', 'komp', 16))
    for label, name, c in variants:
        row = dict(config=dict(zip(('b', 't', 'h', 'dk', 'dv', 'm'), cfg)), operator=label, chunk=c)
        report['rows'].append(row)
        try:
            row['samples'] = [measure(name, data[name], slots, c, 2 if cfg[1] == 8192 else 5) for _ in range(3)]
            row['status'] = 'passed'
            print('BENCH', json.dumps(row), flush=True)
        except Exception as exc:
            row['status'] = 'failed'; row['error'] = repr(exc)
            raise
        finally:
            dest.write_text(json.dumps(report, indent=2))
        torch.cuda.empty_cache()
report['status'] = 'passed'
dest.write_text(json.dumps(report, indent=2))
