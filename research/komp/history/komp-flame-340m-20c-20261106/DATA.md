# FineWeb-Edu data contract

The two models share the same 253 JSONL shards under
`/gpfs/shared_data/qiji/linlizhi/datasets/fineweb-edu`, the original tokenizer
at `/gpfs/shared_data/qiji/huggingface.co/fla-hub/transformer-1.3B-100B`,
and the same deterministic rank assignment and packing order.

`configs/data-manifest.json` contains the absolute runtime paths. Source
inventory records size, mtime, and a SHA256 of the first 64 KiB of every shard;
this is not a full content hash of the approximately 1.04 TB source corpus.
Tokenizer files and both completed heldout token arrays have full SHA256 hashes.

## Heldout split

`prepare_heldout.py` reads the first shard from every available crawl snapshot
in round-robin order. It assigns unique whole documents by the parity of the
first 64 bits of `sha256(text.encode('utf-8'))`, without text normalization.
Validation contains 3,516 documents; test contains 3,363 documents. Their hash
sets are disjoint. Every selected document, including the last partially used
document in each token cache, is excluded from training by its full text hash.

Each cache contains 4,194,305 little-endian uint16 token IDs, supplying
4,194,304 supervised next-token positions in 2,048 windows. A 2,049-ID window
advances by 2,048, sharing the boundary token. The tokenizer JSON adds BOS ID 1
per document and does not append an extra EOS. The cache uses this exact
post-processing contract.

This prevents exact heldout document duplicates from entering training. It
does not claim global training deduplication or semantic near-duplicate removal.
The heldout sets are internal corpus evaluations, not external benchmarks.
Test is reserved for final evaluation; validation monitors training.

## Runtime acceptance

CPU preparation uses tokenizers 0.21.0. A separate check with transformers
4.49.0 / LlamaTokenizerFast exactly reproduced all token IDs for 12 complete
documents; cache lengths, vocabulary bounds, split disjointness, and the union
of training exclusions also passed (`evidence/data-independent-check.json`).

The first GPU-image environment check found an AutoTokenizer mismatch and
stopped before numerical or distributed training tests. CPU-only diagnosis in
that same image identified transformers 5.3.0 / tokenizers 0.22.2 rewriting the
original normalizer and pre-tokenizer: Prepend + Replace with no pre-tokenizer
became no normalizer + Metaspace. BOS/EOS IDs were unchanged, but leading-space
Chinese text changed token IDs. See `evidence/tokenizer-diagnosis.json`.

The training contract therefore uses the frozen native tokenizer JSON, retaining
the original transformers 4.49 semantics. Do not rebuild the caches to match
transformers 5's changed automatic conversion. Release requires GPU-image native
token IDs to match the CPU-prepared document and boundary samples, plus actual-text
loader exclusion and checkpoint cursor/buffer resume checks. Preserve the first
failure and subsequent diagnosis; the original AutoTokenizer parity assertion
remains a recorded failure, not a claim that transformers 5 now agrees.

The subsequent GPU-image check passed all 16 CPU-gold samples using the native
JSON tokenizer. The CPU loader check also witnessed the exclusion of real
heldout documents, exact cursor/token-buffer restoration, and disjoint rank
file assignments covering all 253 shards. Formal full-model training release
is a separate gate and must not be inferred from these data checks.
