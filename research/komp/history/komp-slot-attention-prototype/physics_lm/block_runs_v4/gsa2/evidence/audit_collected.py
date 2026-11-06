"""Audit the completed local evidence without importing CUDA or changing runs."""
from pathlib import Path
import hashlib
import json
import math


def read(path):
    return json.loads(path.read_text())


def main():
    evidence = Path(__file__).resolve().parent
    root = evidence.parent
    project = evidence.parents[3]
    plan = read(evidence / 'plan.json')
    ticket = read(root / 'preflight.json')
    assert ticket['status'] == 'passed'
    expected = {(arm, seed) for arm in plan['arms'] for seed in plan['seeds']}
    actual = set()
    for path in sorted((root / 'results').glob('*/summary.json')):
        result = read(path)
        cfg = result['config']
        identity = (cfg['architecture'], cfg['seed'])
        assert identity in expected and identity not in actual, path
        actual.add(identity)
        assert result['status'] == 'completed'
        assert cfg == read(path.parent / 'manifest.json'), path
        assert cfg['source_sha256'] == ticket['source_sha256'], path
        assert cfg['steps'] == 2231 and result['unique_tokens'] == 10001108
        assert cfg['batch'] == 64 and cfg['epoch'] == 1
        assert (cfg['width'], cfg['layers'], cfg['heads'], cfg['key_value_dim'], cfg['slots']) == (128, 4, 2, 64, 64)
        assert cfg['recurrent_elements_per_head'] == 8192
        conv = 'noconv' not in cfg['architecture']
        assert cfg['short_conv'] == cfg['use_w_conv'] == conv
        assert cfg['peak_lr'] == (1e-3 if cfg['architecture'].endswith('lr1e3') else 5e-4)
        assert result['reload_validation_ce_error'] < 1e-4
        metrics = [json.loads(line) for line in (path.parent / 'metrics.jsonl').read_text().splitlines()]
        steps = [row for row in metrics if row['step'] > 0]
        assert steps[-1]['step'] == 2231 and steps[-1]['unique_tokens'] == 10001108
        assert all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in steps)
        for task in ('depo', 'brevo'):
            assert result['generation'][task]['n'] == 128
            assert 0 <= result['generation'][task]['valid'] <= 1
    assert actual == expected, expected - actual
    assert not list(root.rglob('*.pt'))
    for name, digest in read(evidence / 'prior-report-sha256.json').items():
        assert hashlib.sha256((project / name).read_bytes()).hexdigest() == digest, name
    source_audit = read(evidence / 'final-source-audit.json')
    generation_audit = read(evidence / 'independent-generation-audit.json')
    assert source_audit['status'] == generation_audit['status'] == 'passed'
    assert source_audit['completed_runs'] == generation_audit['completed_runs'] == 21
    assert generation_audit['checked_generations'] == 21 * 256
    assert read(evidence / 'completion.json')['status'] == 'completed'
    assert (evidence / 'exit-code.txt').read_text().strip() == '0'
    receipt = dict(status='passed', completed_runs=len(actual), steps_per_run=2231,
                   tokens_per_run=10001108, generation_records=21 * 256,
                   source_matches_preflight=True, prior_reports_unchanged=True,
                   checkpoint_reload_passed=True, no_checkpoints=True)
    (evidence / 'result-audit.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
