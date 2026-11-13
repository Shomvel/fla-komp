"""Compare collected snapshots without mutating remote evaluation state."""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def main():
    previous_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    previous = read(previous_path)
    current = read(ROOT / 'evidence/latest.json')
    lookup = {(v['model'], v['case']): v for v in current['progress']}
    issues = []
    for old in previous['progress']:
        key = old['model'], old['case']
        new = lookup.get(key)
        if new is None:
            issues.append([*key, 'missing'])
            continue
        for field in ['plan_sha', 'checkpoint_sha']:
            if old[field] != new[field]:
                issues.append([*key, field])
        if new['count'] < old['count']:
            issues.append([*key, 'count_regressed'])
        if not set(old.get('batch_hashes', [])).issubset(new.get('batch_hashes', [])):
            issues.append([*key, 'batch_hashes'])
    old_summaries = {(v['model'], v['case']): v for v in previous['summaries']}
    new_summaries = {(v['model'], v['case']): v for v in current['summaries']}
    for key, old in old_summaries.items():
        new = new_summaries.get(key)
        if new is None or any(old.get(k) != new.get(k) for k in ['count', 'score', 'checkpoint_sha', 'plan_sha']):
            issues.append([*key, 'completed_summary_changed'])
    models = {'block', 'block123', 'learned', 'learned123'}
    states = {k: v for k, v in current['states'].items() if k.split('-s')[0] in models}
    count = sum(v['count'] for v in current['progress'])
    batches = lambda x: sum(len(v.get('batch_hashes', [])) for v in x['progress'])
    summary = dict(
        captured_utc=current['captured_utc'], previous_utc=previous['captured_utc'],
        count=count, total=163656, percent=count / 163656 * 100,
        delta=count - sum(v['count'] for v in previous['progress']),
        completed_items=len(current['summaries']),
        delta_items=len(current['summaries']) - len(previous['summaries']),
        hashed_batches_in_complete_items=batches(current), delta_hashed_batches_in_complete_items=batches(current) - batches(previous),
        jobs=current['jobs'],
        acceptance_passed=sum(v.get('status') == 'passed' for k, v in states.items() if k.endswith('-acceptance.json')),
        workers_complete=sum(k.endswith('-complete.json') for k in states),
        failures=[k for k in states if k.endswith('-failure.json')],
        integrity_issues=issues, plan_unchanged=current['plan_sha'] == previous['plan_sha'],
        new_completed=[{k: v[k] for k in ['model', 'case', 'count', 'score']} for key, v in new_summaries.items() if key not in old_summaries],
        unfinished=[{k: v[k] for k in ['model', 'case', 'count', 'total']} for v in current['progress'] if not v['complete']],
        worker_progress={k: {a: v.get(a) for a in ['case', 'batch_key', 'time', 'latest_batch_seconds']} for k, v in states.items() if k.endswith('-progress.json')},
    )
    output_path.write_text(json.dumps(summary, indent=2) + '\n')
    with (ROOT / 'PROGRESS.md').open('a') as f:
        f.write('\n本次追踪：' + json.dumps(summary, ensure_ascii=False) + '\n\n用户最新要求：已立即切为每小时跟踪eval与1B训练；eval完成核验后仅跟踪1B训练。\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
