# Exact-KOMP efficiency research

Start with `REPORT.md`. Stage 1 completed on one H100 under the cumulative45-minute authorization; reported job time totaled24 minutes2 seconds, and its jobs have been cleaned up. The 8-GPU Stage 2 is **not authorized or run**. This directory is independent from the frozen training runtime; do not run the copied `train.py`, `run_fla.py`, or migration scripts. They are retained solely as source evidence.

CPU checks, available now:

```bash
cd /Users/shom/Downloads/exact-komp-efficiency-research-20261106
python3 -m unittest test_cache_cpu test_compare_cpu -v
python3 cost_model.py
bash -n run_stage1.sh run_stage2.sh
```

Key artifacts: `candidate.patch`, `candidate/geometry_cache.py`, `evidence/cost-model.json`, `evidence/local-validation.json`, `evidence/gpu-summary.json`, `evidence/resource-ledger.json`, `REPORT.md`. The candidate cache defaults OFF; the runners explicitly enable it with `--cache`. `source-manifest.json` inside the copies describes the baseline provenance; it is intentionally not a candidate release receipt. `research-manifest.json` binds this research package.

For a separately authorized future rerun, stage this directory at `/gpfs/users/linlizhi/exact-komp-efficiency-research-20261106`. Do not upload into the formal runtime. The recorded Stage 1 used `job-stage1-r2.yaml` after a documented input-preparation failure. The prepared jobs expect an immutable fixture at `fixture/` and new result directories. Select a concrete complete native-FLA Exact checkpoint; do not bind to a moving `latest` during a benchmark.

CPU fixture preparation on the data host (replace the checkpoint placeholder with the chosen exact path):

```bash
python prepare_fixture.py \
  --checkpoint /ABSOLUTE/PINNED/EXACT/CHECKPOINT \
  --manifest inputs/data-manifest.json --audit inputs/data-audit.json \
  --out fixture
python profile_input.py --fixture fixture --rank 0 --out input-profile-rank0.json
```

Before submission, inspect the two YAMLs, verify project `WeaverRSI` / queue `proj-32`, pin the GPU type/topology through the actual allocation, and check GPFS capacity. `job-stage1.yaml` requests one GPU and has a45-minute deadline; `job-stage2.yaml` requests eight GPUs and has a30-minute deadline. The pinned image is inherited from the frozen training job. Both jobs use private Triton caches, stdout logs and output directories. No auto-submit command is included.

Equivalent commands inside the authorized matching GPU environment:

```bash
export EXACT_RESEARCH_FIXTURE="$PWD/fixture"
export EXACT_RESEARCH_RESULTS="$PWD/results-stage1"
bash run_stage1.sh
python summarize_bench.py results-stage1 --out stage1-speed.json

# Only after stage1 passes and the separate 8-GPU allocation is authorized:
export EXACT_RESEARCH_STAGE1="$PWD/results-stage1"
export EXACT_RESEARCH_RESULTS="$PWD/results-stage2"
bash run_stage2.sh
python summarize_bench.py results-stage2 --out stage2-speed.json
```

The first failure terminates each pipeline. Preserve failed outputs, inspect the full class of contracts statically, add regressions and fix in one batch before rerunning. Do not reuse output directories or retry until something passes. Collect terminal job identity/status and logs before any cleanup through the Northjob skill; never delete or pause either formal training job.

Interpretation: operator/layer measurements do not establish full-model speed. Single rank does not establish global256 behavior. Warm ABBA excludes compile/profile overhead, real streaming I/O and checkpoint/evaluation I/O. Numerical full/tail output, gradient and update comparisons use the original strict thresholds. No existing compatibility failure is relabeled. Production deployment and any longer quality experiment require a separate decision after measured results.

The measured combined candidate uses both `--cache --prune-gc`. In an integrating model, set `layer.attn.cache_geometry=True` for each Exact layer and set `EXACT_KOMP_PRUNE_GC=1` before importing the Exact operator. Keep mixer checkpointing enabled. These controls are opt-in; benchmark baseline/control runs leave them off. No batch, precision or model-math change is needed. Do not enable the research candidate in formal training before the outstanding DDP/quality validation.
