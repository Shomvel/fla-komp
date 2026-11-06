# KOMP340M training

The maintained entry point is `python -m examples.komp.train`, run from this checkout. It supports Exact and Block KOMP, single GPU or `torchrun`, BF16 autocast, gradient accumulation, native FLA loss, AdamW, mixer checkpointing and resumable checkpoints. The architecture is the frozen 339,966,848-parameter recipe: width 1024, 24 layers, 16 heads of dimension 64, 32 slots, FFN 1824 and vocabulary 32000.

Install the fork in a Linux CUDA environment:

```bash
pip install -e '.[cuda,test]'
pip install numpy
```

Supply a flat little-endian `uint16` token file with IDs below 32000. The runner uses contiguous 2049-token windows for 2048 inputs and aligned next-token labels, sharing the boundary token. Each rank receives a disjoint part of the global batch. Full epochs repeat deterministically; incomplete final global batches in the token file are omitted. No dataset download is performed.

```bash
# Small run; adjust memory use with --microbatch.
python -m examples.komp.train --kind exact --tokens /data/train.bin \
  --output /checkpoints/komp-exact --global-batch 16 --microbatch 1 --steps 10

# Standard batch shape on eight GPUs.
EXACT_KOMP_PRUNE_GC=1 torchrun --standalone --nproc_per_node=8 \
  -m examples.komp.train --kind exact --tokens /data/train.bin \
  --output /checkpoints/komp-exact --cache-geometry
```

Use `--kind block` for the C=128 Block model. Geometry caching is Exact-only. Omit `--cache-geometry` and `EXACT_KOMP_PRUNE_GC` for the unchanged default operator path. The CUDA training loop has not been rerun as part of repository consolidation.

Fresh training initializes the seeded legacy model and maps all tensors strictly into the native FLA layout. `--init-checkpoint` instead loads a native weights-only state dict, optionally under `model`. `--resume` reloads this runner's `latest.pt`, including optimizer, step, rank RNG and an exact model/data/batching/schedule contract. Resume requires the same arguments, data hash and world size. Checkpoints are atomically replaced; only `latest.pt` is retained. Do not load checkpoints from untrusted sources.

This portable runner trains a configured number of complete steps. It does not reproduce the formal trainer's exact 6.8-billion-token tail budget, held-out-text exclusion pipeline or release gates. Those complete trainers and data loaders are retained in [the frozen runtime](../../research/komp/history/komp-flame-340m-20c-20261106/migration-fla-mb16/runtime) and [the optimized candidate](../../research/komp/history/exact-komp-efficiency-research-20261106/candidate). They retain their original environment paths and require their frozen inputs/receipts.

## Model and checkpoint API

```python
from fla.models.komp import KOMPConfig, KOMPForCausalLM
from fla.models.komp.conversion import map_weights, map_optimizer

model = KOMPForCausalLM(KOMPConfig(kind="exact", cache_geometry=True))
model.load_state_dict(map_weights(legacy_state_dict), strict=True)
```

Importing `fla.models.komp` registers the configuration and models with Transformers Auto classes. The wrapper preserves the original parameter names, separate gate/up BF16 rounding, and summed aligned-label loss in `TrainingLM`. It is a pretraining adapter: generation cache, masked input sequences, hybrid attention and attention residuals are unsupported. Dimensions are fixed to the validated recipe. Other model sizes and experimental architectures remain in the historical trainers.

The constructors preserve the frozen adapter's initialization rather than invoking a new Hugging Face initialization pass. Use the runner's seeded legacy initialization or load converted weights when reproducing an experiment.

The serialized kernel width parameter is `kernel_temperature=0.5`. Old configs using `temperature=0.5` are accepted and migrated on load; the reserved Transformers generation field is not written back. This changes no kernel parameter or weight.

## Value-decay models

Select `--decay-axis value --kind exact` or `--decay-axis value --kind block` for the independent `komp_value_decay` model. It defaults to FF2272, low-rank output/decay gates, sigmoid output gating and multiple initial head timescales, totaling 341,552,000 parameters. It starts from its own seeded initialization and requires its own checkpoint; slot-model checkpoints cannot be resumed into it. Geometry caching is currently unsupported for this variant. See [the operator/model contract](../../fla/ops/komp/VALUE_DECAY.md) for configuration and direct AutoModel usage.
