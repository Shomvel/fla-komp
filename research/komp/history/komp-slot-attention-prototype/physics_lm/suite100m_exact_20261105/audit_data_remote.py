"""Read-only checks of the existing frozen corpus before any new GPU training."""
import hashlib
import json
from pathlib import Path
import struct

SOURCE = Path('/gpfs/users/linlizhi/physics100m-20261101/data')
ROOT = Path('/gpfs/users/linlizhi/physics100m-exact-20261105')
report = {'status': 'passed', 'source': str(SOURCE), 'tasks': {}}
for task in ('depo2', 'brevo2'):
    folder = SOURCE/task
    manifest = json.loads((folder/'manifest.json').read_text())
    checks = {}
    for split in ('train', 'val', 'test'):
        info = manifest['splits'][split]
        tokens = folder/f'{split}.tokens.u16'
        masks = folder/f'{split}.mask.u8'
        raw = (folder/f'{split}.offsets.u64').read_bytes()
        offsets = struct.unpack('<'+'Q'*(len(raw)//8), raw)
        assert len(offsets)-1 == info['documents']
        assert offsets[0] == 0 and all(a < b for a, b in zip(offsets, offsets[1:]))
        assert offsets[-1]*2 == tokens.stat().st_size
        assert offsets[-1] == masks.stat().st_size
        assert offsets[-1]-len(offsets)+1 == info['next_token_positions']
        sha = hashlib.sha256()
        with tokens.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                sha.update(block)
        assert sha.hexdigest() == info['token_sha256']
        checks[split] = {'documents': info['documents'], 'positions': info['next_token_positions'],
                         'token_sha256': sha.hexdigest(), 'max_length': max(b-a for a,b in zip(offsets,offsets[1:]))}
    assert checks['train']['positions'] >= 100_000_000
    (ROOT/'evidence'/f'{task}-data-manifest.json').write_text(json.dumps(manifest, indent=2))
    report['tasks'][task] = checks
(ROOT/'evidence/data-audit.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
