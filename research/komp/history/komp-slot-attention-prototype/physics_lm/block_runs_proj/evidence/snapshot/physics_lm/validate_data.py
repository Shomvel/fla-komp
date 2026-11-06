"""Small, deterministic PhysicsLM4 Part 4.1 tasks; no packing/truncation."""
import hashlib
import json
import random
from pathlib import Path
import physics_port as p

ROOT = Path(__file__).resolve().parent


def check(row):
    ids = row['input_ids']
    assert len(ids) == len(row['loss_mask']) and ids[0] == 9999
    assert max(ids) < 32000 and len(ids) <= 256
    if row['task'] == 'depo':
        # Independently parse the single/multi-token dictionary and verify every hop.
        first = next(i for i, t in enumerate(ids) if 9000 < t < 9500)
        words, word = [], []
        for t in ids[1:first]:
            word.append(t)
            if t > 50:
                words.append(tuple(word)); word = []
        assert not word and len(words) % 2 == 0
        succ = dict(zip(words[::2], words[1::2]))
        assert len(succ) == row['meta']['n']
        i = first
        while ids[i] != 9998:
            k = ids[i] - 9000; i += 1
            start = []
            while ids[i] != 9500:
                start.append(ids[i]); i += 1
            assert row['loss_mask'][i]
            i += 1; ans = []
            while True:
                ans.append(ids[i]); i += 1
                if ans[-1] > 50:
                    break
            cur = tuple(start)
            for _ in range(k):
                cur = succ[cur]
            assert tuple(ans) == cur
    else:
        q = ids.index(9997)
        edges = list(zip(ids[1:q:2], ids[2:q:2]))
        query = ids[q + 1]
        answer = ids[q + 3:-1]
        ancestors = {query}
        while True:
            more = ancestors | {a for a, b in edges if b in ancestors}
            if more == ancestors:
                break
            ancestors = more
        assert set(answer) == ancestors and len(answer) == len(ancestors)
        pos = {v: i for i, v in enumerate(answer)}
        assert all(pos[a] < pos[b] for a, b in edges if b in ancestors)


def main():
    out = ROOT / 'data'; out.mkdir(exist_ok=True)
    info = {'source': 'facebookresearch/PhysicsLM4@6c7bbda1d9c9ab104cd4a75814cd83bd29db77f0',
            'depo': {'N': 32, 'K': 4, 'M': 10, 'mini_vocab': 50, 'min_tlen': 1, 'max_tlen': 2},
            'brevo': {'N': 24, 'multi': False}, 'splits': {}}
    sets = []
    for split, seed, count in [('train', 1001, 32768), ('val', 2002, 512), ('test', 3003, 512)]:
        rng = random.Random(seed)
        rows = []
        for i in range(count):
            row = (p.gen_depo_one(rng, p.DepoSpec('depo1-small', 32, 4)) if i % 2 == 0 else
                   p.gen_brevo_one(rng, p.BrevoSpec('brevo1-small', 24, False)))
            check(row); rows.append(row)
        content = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows)
        (out / (split + '.jsonl')).write_text(content)
        # Ensure no identical full documents cross split boundaries.
        hashes = {hashlib.sha256(json.dumps(r['input_ids']).encode()).hexdigest() for r in rows}
        assert all(not (hashes & other) for other in sets)
        sets.append(hashes)
        info['splits'][split] = {'seed': seed, 'examples': count,
            'tokens': sum(len(r['input_ids']) for r in rows),
            'max_length': max(len(r['input_ids']) for r in rows),
            'sha256': hashlib.sha256(content.encode()).hexdigest()}
    (out / 'manifest.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
