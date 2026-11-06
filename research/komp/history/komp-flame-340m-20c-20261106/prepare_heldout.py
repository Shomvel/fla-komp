"""Build document-disjoint fixed FineWeb-Edu evaluation streams on CPU."""
import array
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

import tokenizers
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/gpfs/shared_data/qiji/linlizhi/datasets/fineweb-edu')
TOKENIZER = Path('/gpfs/shared_data/qiji/huggingface.co/fla-hub/transformer-1.3B-100B')
TARGET_IDS = 4_194_305


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def main():
    for name in ('data', 'configs', 'evidence'):
        (ROOT / name).mkdir(exist_ok=True)
    if (ROOT / 'configs/data-manifest.json').exists():
        raise RuntimeError('Existing data manifest: verify/reuse, never overwrite a live split')
    files = sorted(SOURCE.glob('*.jsonl'))
    assert len(files) > 0
    tok = Tokenizer.from_file(str(TOKENIZER / 'tokenizer.json'))
    assert tok.get_vocab_size() == 32000
    # Distribute evaluation documents across all available crawl snapshots.
    representatives = {}
    inventory = []
    for p in files:
        representatives.setdefault(p.name.split('_train-')[0], p)
        st = p.stat()
        with p.open('rb') as handle:
            prefix_hash = hashlib.sha256(handle.read(65536)).hexdigest()
        inventory.append(dict(path=str(p), bytes=st.st_size, mtime_ns=st.st_mtime_ns,
                              first_65536_bytes_sha256=prefix_hash))
    streams = [(str(p), p.open()) for p in representatives.values()]
    buffers = {k: array.array('H') for k in ('validation', 'test')}
    records = {k: [] for k in buffers}
    excluded = set()
    seen = set()
    samples = []
    duplicates = 0
    try:
        while any(len(v) < TARGET_IDS for v in buffers.values()):
            for path, stream in streams:
                offset = stream.tell()
                raw = stream.readline()
                if not raw:
                    raise RuntimeError('Representative shard exhausted before heldout budget')
                row = json.loads(raw)
                text = row['text']
                digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
                if digest in seen:
                    duplicates += 1
                    continue
                seen.add(digest)
                split = 'validation' if int(digest[:16], 16) % 2 == 0 else 'test'
                if len(buffers[split]) >= TARGET_IDS:
                    continue
                ids = tok.encode(text, add_special_tokens=True).ids
                assert ids and ids[0] == 1 and max(ids) < 32000
                excluded.add(digest)
                records[split].append(dict(source=path, offset=offset, text_sha256=digest,
                                           raw_tokens=len(ids), retained_tokens=min(len(ids), TARGET_IDS-len(buffers[split]))))
                buffers[split].extend(ids[:TARGET_IDS-len(buffers[split])])
                if len(samples) < 12:
                    samples.append(dict(text=text, ids=ids))
            if sum(map(len, records.values())) % 100 < len(streams):
                print(json.dumps({k:len(v) for k,v in buffers.items()}), flush=True)
    finally:
        for _, stream in streams:
            stream.close()
    paths = {}
    for split, values in buffers.items():
        assert len(values) == TARGET_IDS
        if sys.byteorder != 'little':
            values.byteswap()
        out = ROOT / 'data' / (split + '.uint16')
        with out.open('wb') as f:
            values.tofile(f)
        paths[split] = str(out)
    exclusion = ROOT / 'data/heldout_text_sha256.json'
    save_json(exclusion, sorted(excluded))
    save_json(ROOT / 'evidence/heldout-documents.json', records)
    save_json(ROOT / 'evidence/tokenizer-parity-samples.json', samples)
    save_json(ROOT / 'evidence/source-inventory.json', inventory)
    evidence = dict(
        status='prepared; GPU image tokenizer parity must pass before release',
        train_shards=len(files), source_bytes=sum(r['bytes'] for r in inventory),
        source_inventory_sha256=sha(ROOT / 'evidence/source-inventory.json'),
        source_hash_scope='size/mtime and first 64 KiB per shard; not a full 1 TB content hash',
        tokenizers_version=tokenizers.__version__, tokenizer_files={
            n:sha(TOKENIZER/n) for n in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','tokenizer.model')},
        text_hash='sha256(text.encode(utf-8)); no normalization',
        split_policy='Round-robin first shard of every crawl snapshot; unique entire documents assigned by SHA256 first64bits parity; all selected document hashes excluded from training globally',
        document_count={k:len(v) for k,v in records.items()}, excluded_documents=len(excluded),
        skipped_duplicate_documents=duplicates,
        split_hashes_disjoint=not bool({r['text_sha256'] for r in records['validation']} & {r['text_sha256'] for r in records['test']}),
        token_files={k:dict(path=p, sha256=sha(Path(p)), dtype='little-endian uint16',
                           raw_ids=TARGET_IDS, supervised_tokens=TARGET_IDS-1) for k,p in paths.items()},
        exclusion_sha256=sha(exclusion), packing='BOS per document, no extra EOS; continuous stream, 2049-token windows advancing by2048',
        limitations=['No global training deduplication or near-duplicate semantic deduplication; exact selected heldout text duplicates are filtered by training loader.',
                     'Deterministic heldout sampling within crawl snapshots, not an external benchmark.'])
    save_json(ROOT/'evidence/data-audit.json', evidence)
    save_json(ROOT/'configs/data-manifest.json', dict(
        train_files=[str(p) for p in files], tokenizer_path=str(TOKENIZER),
        excluded_text_sha256_path=str(exclusion), validation_tokens_path=paths['validation'],
        test_tokens_path=paths['test'], dtype='uint16', byte_order='little',
        validation_supervised_tokens=TARGET_IDS-1, test_supervised_tokens=TARGET_IDS-1,
        train_exclusion_required=True, data_audit_sha256=sha(ROOT/'evidence/data-audit.json')))
    print(json.dumps({k:evidence[k] for k in ('document_count','excluded_documents','split_hashes_disjoint','token_files')},indent=2))


if __name__ == '__main__':
    main()
