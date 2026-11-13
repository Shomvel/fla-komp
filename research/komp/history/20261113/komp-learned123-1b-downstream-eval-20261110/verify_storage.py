"""Run from independent SSH storage client after worker probes appear."""
import json, os
from pathlib import Path
from sharding import digest,write_atomic
ROOT=Path(__file__).resolve().parent
rows=[]
for shard in range(8):
    claim=ROOT/'sharded/claims-v2'/f'learned123_1b-s{shard}'
    owner=json.loads((claim/'owner.json').read_text())
    probe=claim/'publish-probe.json'
    before=digest(probe)
    try: claim.mkdir()
    except FileExistsError: pass
    else: raise AssertionError('Duplicate claim unexpectedly succeeded')
    temp=claim/'ssh-test-source'
    temp.write_text('collision probe')
    try:
        try: os.link(temp,probe)
        except FileExistsError: pass
        else: raise AssertionError('Existing probe unexpectedly overwritten')
    finally: temp.unlink()
    assert digest(probe)==before
    rows.append(dict(shard=shard,owner=owner,probe_sha=before))
write_atomic(ROOT/'sharded/cross-node-verified.json',dict(status='passed',workers=rows))
print(json.dumps(dict(status='passed',workers=len(rows))))
