"""Atomic evaluation resume records, bound to the exact run and evaluator."""
import json, os

def load_progress(path, identity, probes, grade):
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    assert state['identity'] == identity, 'evaluation resume identity mismatch'
    records = state['records']
    assert len(records) <= len(probes) and (len(records) % 8 == 0 or len(records) == len(probes))
    assert [r['probe'] for r in records] == probes[:len(records)], 'evaluation resume probe mismatch'
    assert all(r['valid'] == grade(r['probe'], r['guess']) for r in records), 'evaluation resume score mismatch'
    assert state['seconds'] >= 0
    return state

def save_progress(path, identity, metrics, records, seconds):
    state = dict(identity=identity, test=metrics, records=records, seconds=seconds)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(state))
    os.replace(temp, path)
