"""Read-only verification of the complete candidate and immutable model/checkpoint pins."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_candidate():
    path = ROOT/'source-manifest.json'
    manifest = json.loads(path.read_text())
    expected = set(manifest['files'])
    actual = {str(p.relative_to(ROOT)) for pattern in ('*.py', '*.sh', '*.md') for p in ROOT.rglob(pattern)
              if not any(part in ('.venv', '__pycache__', 'evidence', '.pytest_cache', 'attempts') for part in p.relative_to(ROOT).parts)}
    actual.add('validation-plan.json')
    if expected != actual:
        raise ValueError(dict(error='candidate file set differs', missing=sorted(expected-actual), extra=sorted(actual-expected)))
    for name, digest in manifest['files'].items():
        if fingerprint(ROOT/name) != digest:
            raise ValueError(f'candidate source fingerprint mismatch: {name}')
    return dict(source_manifest_sha256=fingerprint(path), file_count=len(expected))


def plan():
    return json.loads((ROOT/'validation-plan.json').read_text())


def verify_model(name, base, *, checkpoint=True):
    record = plan()['models'][name]
    source = Path(base)/record['folder']
    manifest_path = source/'source-manifest.json'
    if fingerprint(manifest_path) != record['source_manifest_sha256']:
        raise ValueError(f'{name}: source manifest mismatch')
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest['files'].items():
        if fingerprint(source/relative) != expected:
            raise ValueError(f'{name}: frozen source changed: {relative}')
    if fingerprint(source/'configs/data-manifest.json') != record['data_manifest_sha256']:
        raise ValueError(f'{name}: data manifest mismatch')
    result = dict(model=name, source=str(source), source_manifest_sha256=record['source_manifest_sha256'],
                  data_manifest_sha256=record['data_manifest_sha256'], source_file_count=len(manifest['files']))
    if checkpoint:
        path = Path(base)/record['checkpoint_relative']
        complete = path.parent/'complete.json'
        required = [path, complete]+[path.parent/f'rank{i}.pt' for i in range(8)]
        if not all(p.is_file() for p in required):
            raise ValueError(f'{name}: pinned checkpoint is incomplete')
        marker = json.loads(complete.read_text())
        if marker.get('step') != record['step'] or marker.get('tokens', marker.get('supervised_tokens')) != record['tokens']:
            raise ValueError(f'{name}: checkpoint completion marker step/token mismatch: {marker}')
        if marker.get('source_sha') != record['source_manifest_sha256'] or marker.get('data_sha') != record['data_manifest_sha256']:
            raise ValueError(f'{name}: checkpoint completion source/data mismatch')
        digest = fingerprint(path)
        if digest != record['checkpoint_sha256']:
            raise ValueError(f'{name}: checkpoint fingerprint mismatch')
        result.update(checkpoint=str(path), checkpoint_sha256=digest, step=record['step'], tokens=record['tokens'],
                      completion_marker_sha256=fingerprint(complete))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='/gpfs/users/linlizhi')
    parser.add_argument('--model', action='append')
    parser.add_argument('--source-only', action='store_true')
    parser.add_argument('--candidate-only', action='store_true')
    args = parser.parse_args()
    candidate = verify_candidate()
    models = [] if args.candidate_only else [verify_model(name, args.base, checkpoint=not args.source_only)
                                            for name in args.model or plan()['order']]
    print(json.dumps(dict(status='passed', candidate=candidate, models=models), indent=2))


if __name__ == '__main__':
    main()
