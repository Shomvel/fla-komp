"""Create the candidate manifest once; existing frozen manifests are never overwritten."""

import json
from pathlib import Path

from verify_sources import fingerprint

root = Path(__file__).resolve().parent
files = {str(p.relative_to(root)): fingerprint(p) for pattern in ('*.py', '*.sh', '*.md') for p in root.rglob(pattern)
         if not any(part in ('.venv', '__pycache__', 'evidence', '.pytest_cache', 'attempts') for part in p.relative_to(root).parts)}
files['validation-plan.json'] = fingerprint(root/'validation-plan.json')
manifest = dict(version=1, files=dict(sorted(files.items())))
with (root/'source-manifest.json').open('x') as stream:
    stream.write(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(dict(source_manifest_sha256=fingerprint(root/'source-manifest.json'), files=len(files))))
