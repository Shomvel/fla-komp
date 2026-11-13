# KOMP operators

The active implementation is imported from the November 6, 2026 KOMP340M optimization candidate. Numerical function bodies are preserved; packaging changes replace experiment-local imports and add formatting. `validation/komp/verify_sources.py` checks the imported bodies against the archived sources.

| Family | Entry point | Scope |
| --- | --- | --- |
| Exact KOMP | `fla.ops.komp.exact_komp` | Per-token compression, FIFO or learned deletion, compute chunks 16/32, FP32 or diagnostic FP64 geometry |
| Block KOMP | `fla.ops.komp.block_komp` | Chunk-boundary dictionary compression |
| Fast Block | `fla.ops.komp.block_komp_fast` | Fused projections, scans, backward and configurable tape precision; model uses C=128 |
| Flexible Block | `fla.ops.komp.block_komp_flex` | Research routing and dictionary-selection variants |
| Projected slots | `fla.ops.komp.projected_slot_attention` | Recurrent/chunk implementations and PyTorch references |
| BSGD | `fla.ops.komp.bsgd`, `fla.ops.komp.bsgd_chunk` | Budgeted merging, recurrent and chunk implementations |
| KAPA | `fla.ops.komp.kapa`, `fla.ops.komp.kapa_chunk` | Protected directions, Gaussian features and chunk implementations |
| References | `block_reference`, `reference`, `bsgd_reference`, `kapa_reference` | Pure PyTorch oracles and research variants |

## Exact input contract

```python
from fla.ops.komp import exact_komp

output, (final_keys, final_values), metadata = exact_komp(
    q, k, v, log_decay, beta, (initial_keys, initial_values),
    eviction='fifo', compute_chunk=16, projection_dtype='float32',
)
```

- `q/k`: `[B,T,H,K]`; `v`: `[B,T,H,V]`; `beta`: `[B,T,H]`.
- `log_decay`: scalar per head `[B,T,H]` or per slot `[B,T,H,M]`.
- Initial keys: `[B,H,M,K]`; initial values: `[B,H,M,V]`.
- CUDA; FP32/BF16 inputs; `T >= 1`, `1 <= M <= 64`, `1 <= K,V <= 128`. First-order gradients only.
- Learned deletion additionally accepts scores and streaming score/birth metadata. See the function docstring for the full contract.
- Exact compression always happens at every token. `compute_chunk` changes execution scheduling, not compression frequency. Block compression changes the algorithm and must be evaluated separately.

## Existing optimizations

`fla.ops.komp.geometry_cache.geometry_contexts` caches immutable FIFO FP32 geometry during non-reentrant checkpoint replay. It is opt-in through `KOMPConfig(cache_geometry=True)`, requires eager execution and mixer checkpointing, and is scoped to one checkpoint graph. Learned selection and FP64 diagnostics do not use this cache.

Set `EXACT_KOMP_PRUNE_GC=1` **before importing FLA** to omit the unused global cumulative-decay tape. The local cumulative sums and backward computation are unchanged. This is also off by default.

The frozen single-H100 experiment measured 17.33% higher throughput with caching, at an additional approximately 4.29 GiB peak allocated memory. The tape-pruning change alone showed no measured speedup. These are historical measurements, not measurements of the latest-upstream integration. See [validation notes](../../../research/komp/VALIDATION.md).

Mathematical notes and older optimization branches are indexed in [the source collection](../../../research/komp/README.md).
