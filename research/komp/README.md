# KOMP source collection

Recent kernel changes, complete model scores and current experiment status: [2026-11-13 results](RECENT_RESULTS.md). The [kernel report](KERNEL_RESULTS.md), [184 downstream scores](results/20261113/MODEL_RESULTS.md), and [dated source archive](history/20261113/README.md) link each measurement to its original receipt.

This fork's `main` includes the maintained operators, native FLA model and trainer, plus the local research implementations. Upstream FLA is pinned at `9e82a80fdaefcd7c9cc3da51ea36d6a9e1587bd1` for this import.

| Component | Location |
| --- | --- |
| Exact, Block, flexible Block, projected slots, BSGD, KAPA | [fla/ops/komp](../../fla/ops/komp) |
| Native 340M mixer | [fla/layers/komp.py](../../fla/layers/komp.py) |
| Transformers config/model and legacy checkpoint conversion | [fla/models/komp](../../fla/models/komp) |
| Portable single-GPU/DDP trainer | [examples/komp](../../examples/komp) |
| Host and CUDA acceptance programs | [validation/komp](../../validation/komp) |
| Original variants, optimization trials, trainers and benchmarks | [history](history) |
| Per-file provenance and hashes | [source-manifest.json](source-manifest.json) |

## Historical sources

The archive preserves original source text and directory relationships. Identical files share a relative symlink to a retained copy. It is a source archive, not an installed Python package: launchers retain machine-specific paths and old package assumptions. In particular, archived `sys.modules` bootstrap code belongs to the frozen experiment environment, not the current package API.

| Source directory | Retained files | Contents |
| --- | ---: | --- |
| [komp-slot-attention-prototype](history/komp-slot-attention-prototype) | 1330 | Original slot, Exact/Block, BSGD/KAPA, 100M physics training and variants |
| [komp-flame-340m-20c-20261106](history/komp-flame-340m-20c-20261106) | 322 | 340M recipe, loaders, DDP, native FLA migration, checkpoint/optimizer conversion |
| [komp-exact-opt-20261105](history/komp-exact-opt-20261105) | 71 | First Exact optimization and acceptance |
| [komp-exact-triton-20261105](history/komp-exact-triton-20261105) | 14 | Exact Triton implementation and oracle |
| [komp-exact-opt2-20261105](history/komp-exact-opt2-20261105) | 95 | Hierarchical scan, direct/gather geometry, learned versions, ablations |
| [exact-komp-efficiency-research-20261106](history/exact-komp-efficiency-research-20261106) | 146 | Geometry checkpoint cache, gc tape pruning, performance and numerical acceptance |
| [komp-gdn-8k-20261105](history/komp-gdn-8k-20261105) | 10 | 8K GatedDeltaNet comparison |
| [physics100m-komp-variants-20261102](history/physics100m-komp-variants-20261102) | 241 | Physics 100M variant training and evaluation |
| [mesanet-komp-research-20261105](history/mesanet-komp-research-20261105) | 10 | MesaNet comparison and research |
| [komp-c128-triton-20261101](history/komp-c128-triton-20261101) | 84 | C128 Block Triton tuning |
| [komp-accel-scratch](history/komp-accel-scratch) | 248 | Block fusion, precision, indexing, coherence and scratch variants |

Source, launch scripts, mathematical notes, patches and configuration manifests are included. Frozen third-party FLA/flame source trees, virtual environments, caches, datasets, checkpoints, compiled binaries and generated run logs are excluded. They are not required to inspect the KOMP code; archived experiments may need their pinned dependencies to run. The current upstream FLA code is present at repository root.

Run `python validation/komp/verify_sources.py` to validate all archived hashes and numerical-source AST equivalence for the active operators. The check ignores import relocation, docstrings and equivalent lambda-to-function formatting. It does not certify CUDA execution, performance or compatibility of every historical experiment.

See [validation status](VALIDATION.md) for the distinction between the previous H100 experiment and this repository integration.
