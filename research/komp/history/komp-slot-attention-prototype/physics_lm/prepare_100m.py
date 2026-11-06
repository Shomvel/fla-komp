"""Extend the frozen 10M corpus to 100M unique tokens, keeping evaluation splits fixed."""
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import physics_port as port
from validate_data import check

ROOT = Path(__file__).resolve().parent / 'lrwd_runs'
SOURCE = Path('/gpfs/users/linlizhi/softmax-10m-20261023/data')


def fingerprint(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).digest()


def main():
    dest = ROOT / 'data100m'
    dest.mkdir(parents=True, exist_ok=False)
    original = json.loads((Path(__file__).parent / 'data-manifest.json').read_text())
    info = {k: v for k, v in original.items() if k != 'splits'}
    info.update(target_unique_input_tokens=100_000_000, splits={}, nested_10m=True,
                generation_seed=2026103105, dedup_key='SHA256 of canonical input token array')
    seen = set()
    # Preserve exclusions used to produce the original 10M corpus.
    prior = Path('/gpfs/users/linlizhi/gdn-component-lr-20261023/data')
    for split in ('train', 'val', 'test', 'probe', 'confirm'):
        for line in (prior / (split + '.jsonl')).open():
            seen.add(fingerprint(json.loads(line)['input_ids']))
    info['excluded_prior_documents'] = len(seen)
    for split in ('val', 'test', 'confirm'):
        source = SOURCE / (split + '.jsonl')
        assert hashlib.sha256(source.read_bytes()).hexdigest() == original['splits'][split]['sha256']
        for line in source.open():
            row = json.loads(line)
            check(row)
            key = fingerprint(row['input_ids'])
            assert key not in seen
            seen.add(key)
        shutil.copyfile(source, dest / source.name)
        info['splits'][split] = original['splits'][split]
    source = SOURCE / 'train.jsonl'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original['splits']['train']['sha256']
    counts = dict(examples=0, tokens=0, supervised_tokens=0, content_supervised_tokens=0,
                  max_length=0, rejected_duplicates=0, task_examples={'depo': 0, 'brevo': 0})
    digest = hashlib.sha256()
    started = time.monotonic()
    rng = random.Random(info['generation_seed'])
    with (dest / 'train.jsonl').open('x') as out:
        def append(row, line):
            check(row)
            ids, mask = row['input_ids'], row['loss_mask']
            assert len(ids) <= 192
            key = fingerprint(ids)
            assert key not in seen
            seen.add(key)
            out.write(line)
            digest.update(line.encode())
            counts['examples'] += 1
            counts['tokens'] += len(ids)
            counts['supervised_tokens'] += sum(mask[1:])
            counts['content_supervised_tokens'] += sum(m and t < 9000 for m, t in zip(mask[1:], ids[1:]))
            counts['max_length'] = max(counts['max_length'], len(ids))
            counts['task_examples'][row['task']] += 1

        for line in source.open():
            append(json.loads(line), line)
        assert counts['tokens'] == 10001108
        while counts['tokens'] < 100_000_000 or counts['examples'] % 16:
            row = (port.gen_depo_one(rng, port.DepoSpec('depo1-small', 32, 4)) if counts['examples'] % 2 == 0 else
                   port.gen_brevo_one(rng, port.BrevoSpec('brevo1-small', 24, False)))
            if fingerprint(row['input_ids']) in seen:
                counts['rejected_duplicates'] += 1
                continue
            append(row, json.dumps(row, sort_keys=True) + '\n')
            if counts['examples'] % 100000 == 0:
                print('DATA', counts['examples'], counts['tokens'], round(time.monotonic() - started), flush=True)
    counts.update(sha256=digest.hexdigest(), cross_split_duplicates=0,
                  seed=info['generation_seed'], includes_frozen_10m=original['splits']['train'])
    assert counts['task_examples']['depo'] == counts['task_examples']['brevo']
    info['splits']['train'] = counts
    info['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (Path(__file__), Path(port.__file__))}
    (dest / 'manifest.json').write_text(json.dumps(info, indent=2))
    print('DATA COMPLETE', json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
