from pathlib import Path
import hashlib,json
r=Path('/gpfs/users/linlizhi/komp-coh-trials-20261101');s=r/'snapshot-fine'
files={str(p.relative_to(s)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(s.rglob('*')) if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in str(p)}
old=json.loads((r/'snapshot-sha256.json').read_text());changed=[n for n,h in files.items() if old.get(n)!=h]
assert changed==['physics_lm/block_coherence.py'],changed
(r/'fine-snapshot-sha256.json').write_text(json.dumps(files,indent=2));print(json.dumps(dict(changed=changed,files=len(files))))
