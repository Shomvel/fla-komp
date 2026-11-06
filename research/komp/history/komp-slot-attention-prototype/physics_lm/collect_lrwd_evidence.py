"""Archive completed experiment receipts and checkpoint hashes, leaving checkpoints in place."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--study', choices=('lrwd_runs', 'gsa2_runs'), required=True)
    parser.add_argument('--expected-runs', type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    study = root / 'physics_lm' / args.study
    completed = json.loads((study / 'completed.json').read_text())
    assert completed['status'] == 'completed'
    summaries = sorted((study / 'results').glob('*/summary.json'))
    assert len(summaries) == args.expected_runs
    checkpoints = {}
    checks = 0
    for path in summaries:
        result = json.loads(path.read_text())
        assert result['status'] == 'completed' and result['reload_validation_ce_error'] == 0
        assert result['unique_tokens'] == result['config']['data']['splits']['train']['tokens']
        assert all(c['prefix_max_abs'] == 0 for c in result['trained_causality'])
        checks += len(result['trained_causality'])
        initial = json.loads((path.parent / 'initial-causality.json').read_text())
        assert all(c['prefix_max_abs'] == 0 for c in initial)
        checks += len(initial)
        checkpoint = path.parent / 'model.pt'
        digest = hashlib.sha256()
        with checkpoint.open('rb') as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        checkpoints[str(checkpoint.relative_to(root))] = digest.hexdigest()
    selection = json.loads((study / 'selection.json').read_text())
    heldouts = list((study / 'results').glob('*/heldout.json'))
    assert len(heldouts) == 3 * len(selection['models'])
    selection_sha = hashlib.sha256((study / 'selection.json').read_bytes()).hexdigest()
    for path in heldouts:
        result = json.loads(path.read_text())
        assert result['selection_sha256'] == selection_sha
        records = json.loads((path.parent / 'generations.json').read_text())
        for task in ('depo', 'brevo'):
            rows = [r for r in records if r['task'] == task]
            assert len(rows) == 128
            assert sum(r['valid'] for r in rows) / 128 == result['generation'][task]['valid']
    receipt = {'status': 'passed', 'training_runs': len(summaries), 'heldout_evaluations': len(heldouts),
               'prefix_interventions': checks, 'checkpoint_sha256': checkpoints,
               'selection_sha256': selection_sha,
               'script_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((root / 'physics_lm').glob('*.py'))}}
    receipt_path = root / 'final-verification.json'
    receipt_path.write_text(json.dumps(receipt, indent=2))
    archive = root / 'final-evidence.tgz'
    with tarfile.open(archive, 'w:gz') as target:
        for path in sorted(study.rglob('*')):
            if 'data100m' in path.parts or not path.is_file() or path.suffix not in ('.json', '.jsonl', '.log', '.md', '.yaml'):
                continue
            target.add(path, arcname=str(path.relative_to(root)))
        for name in ('final-verification.json', 'job.log', 'payload-sha256.txt'):
            path = root / name
            if path.exists():
                target.add(path, arcname=name)
    print(json.dumps({k: v for k, v in receipt.items() if not k.endswith('sha256')}, indent=2))
    print(archive, archive.stat().st_size)


if __name__ == '__main__':
    main()
