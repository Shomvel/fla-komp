# Validation scope

The upstream base is `9e82a80fdaefcd7c9cc3da51ea36d6a9e1587bd1`. This consolidation preserves the frozen candidate's operator mathematics, defaults and precision; native package integration and the portable trainer are new integration work.

## Integration result

Ten host tests passed on Linux CPU (Python 3.14.8, PyTorch 2.15.0+cpu, Triton 3.9.0, Transformers 5.22.1). Editable installation and the trainer CLI passed. Core pre-commit checks, copyright headers, source hashes and operator equivalence checks passed. Config serialization now uses `kernel_temperature`, while loading the frozen `temperature=0.5` field remains supported. See the [integration receipt](evidence/integration.json).

## Reproducible checks

```bash
python validation/komp/verify_sources.py
python scripts/check_header.py --check
python -m pytest -q validation/komp/test_contracts.py validation/komp/test_geometry_cache.py
```

The source check validates 2,571 retained archival paths (1,523 unique contents) and 31 active operator/cache files. It compares ASTs after removing imports/docstrings and normalizing equivalent lambda formatting. It preserves branch structure and arithmetic, including the Triton constant branch in Block scan.

Host contracts check normal package imports and Transformers registration, config serialization, the complete 339,966,848-parameter legacy/native state layout for Exact and Block, mapped MLP values and gradients, rank-sharded token batches and resume indexing, reference gradients, and actual PyTorch non-reentrant checkpoint cache replay/lifetime/error behavior.

For a future CUDA validation run in an explicitly allocated GPU environment:

```bash
python -m validation.komp.numerical_cuda
EXACT_KOMP_PRUNE_GC=1 python -m validation.komp.numerical_cuda
```

The CUDA program retains the frozen recipe's acceptance inputs and tolerances, including random FP32/BF16 forward/state/all-gradient comparisons against double references, actual mixer projections and T=2048 causal-prefix checks. It writes `evidence/numerical.json`. It has not been executed against this integrated checkout during consolidation. The portable CUDA/DDP trainer has not received a new end-to-end training run.

## Historical H100 measurements

These results belong to the original frozen runtime and are retained for provenance; they are not a GPU certification of the latest-upstream integration.

- Actual 340M model, B=16 × accumulation 2, T=2048, checkpoint step 3072 and fixed real inputs, one H100.
- ABBA median: 5.2414 s baseline versus 4.4673 s candidate; throughput +17.33%, elapsed time −14.77%.
- Peak allocated memory: 30.948 → 35.242 GiB. Cache-only delivered the measured speedup; gc pruning alone showed no meaningful speedup.
- Twenty FP32/BF16 operator cases passed; full/tail checkpoint differential tests passed. Fourteen repeated fixed-batch updates had maximum loss difference 1.76e-4.
- Eight-rank/DDP performance and long-term quality were not established by that one-GPU experiment.

Receipts: [ABBA timings](evidence/frozen-20261106/abba-summary.json), [GPU summary](evidence/frozen-20261106/gpu-summary.json), [repeated-input losses](evidence/frozen-20261106/repeated-input-loss.json), [resource ledger](evidence/frozen-20261106/resource-ledger.json), and [original report](history/exact-komp-efficiency-research-20261106/REPORT.md).

The historical 100M/340M trainers, source variants and launchers are preserved for inspection and reconstruction. They have not all been rerun, and may depend on data, checkpoints, package versions and environment-specific paths outside this repository.
