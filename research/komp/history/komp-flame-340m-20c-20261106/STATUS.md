# 340M × 20-token/parameter KOMP pretraining

New experiment; original PhysicsLM runs and frozen snapshots are unchanged.

- Paired models: Exact-KOMP and Block-KOMP, each requested8H100GPUs and6,800,000,000 effective inputs/next-token targets. Current nativeFLA mb16: both Exact and Block formally training; both passed the predeclared native acceptance, while strict trajectory compatibility remains failed and documented.
- Shared backbone: width 1024, 24 layers, 16 heads ×64, M32, FF1824, 339,966,848 parameters each, confirmed by actual CUDA model numel.
- Flame recipe: FineWeb-Edu local253shards, frozen32K tokenizer, seq2048, global256 sequences (requested current: microbatch16 per GPU,2 accumulation steps), AdamW lr3e-4 betas(.8,.95), eps1e-15, wd.1, warmup1024, cosine floor.1, seed42.
- Model precision: BF16 network; FP32 projection/state. Exact FP32 chunk tape; Block default BF16 chunk tape and attention operands. The precision difference remains explicit.
- Formal jobs are gated on numerical tests, 8rank finite gradients/update/replica consistency, checkpoint/RNG restore, same-local-shape loss-scaling witness, native tokenizer parity, real-data release validation and source/data hashes.

Preflight failures preserved:
1. `komp340m-preflight-20261106`: stopped before numerical work. Transformers5.3 AutoTokenizer rebuilt Llama normalizer unlike the frozen JSON/Transformers4.49 contract. Native tokenizers0.22.2 matches all16 CPU reference samples exactly; training deliberately uses the frozen JSON, retaining original data IDs.
2. `komp340m-preflight-r2-20261106`: Exact checks passed; Block keys failed because validator compared age-order returned FIFO state against physical-slot reference. CPU regression over6 sequence lengths, varying per-slot gates, output/state and all gradients proves the reference permutation equivalence to ≤4.45e-15. Validator now uses original age-order Block reference. Kernel and numeric tolerances unchanged.
3. `komp340m-preflight-r3-20261106`: Completed. All62 numerical checks and both8rank full-model finite updates/replica consistency/model+Adam+RNG restore checks passed. Terminal evidence saved and job removed.

Heldout caches and all source metadata live under `data/`, `configs/data-manifest.json`, and `evidence/`. Whole selected documents are excluded from training by exact UTF8 text SHA256. Full corpus near-duplicate deduplication is not performed. Cache/tokenizer/exclusion full hashes are checked at training startup;1TB source inventory hashes only the first64KiB plus size/mtime.

Numerical tolerances fixed before the first CUDA numeric run: FP32 relativeRMS/normalizedMax .005/.005; BF16 .035/.04. These are representative shape checks, not guarantees for all ill-conditioned Gram systems.

## Original mb4 formal startup accepted (historical)

Both jobs run independently on8 H10080GB GPUs (16 total):

| Model | Alias | Identity | Startup progress | Warm update median | Rank0 peak allocated |
|---|---|---|---:|---:|---:|
| Exact | komp340m-exact-20c-20261106 | job-da21b786c37b-20261106045921 |12steps /6,291,456tokens |5.669s |8.95GB |
| Block | komp340m-block-20c-20261106 | job-afaf5c905dc4-20261106045941 |12steps /6,291,456tokens |3.335s |8.99GB |

At startup, training losses reached10.3964/10.3688 with finite gradient norms. These are early warmup training losses, not validation quality results. Early full-update medians imply approximately20.4h Exact and12.0h Block compute time for12970steps; periodic evaluation/checkpoint I/O and scheduling add overhead. Budget roughly21–24h /13–16h, subject to later measured throughput.

Each model starts from seed42 initial weights, independently of validation/benchmark updates. First formal losses match release exactly; second Exact loss differs by9.48e-6 across runs, consistent with the nondeterministic floating-point accumulation path (no bitwise cross-run claim).

Release receipts bind source `70508897cf56d89882f6f64cdcff0f243c848a93c47aa21c07fb9934c975e00a`, data manifest `2c6f0fa997a87c85d2b7940c99051523751fb725d3d221b775ed6d894262aa0f`, and batching `74a286aa3551adc41ab330d94f29363fa169e068c70934128c7f8a084bff228b`. Current sources are checked against these before training. All preflight/release jobs are removed after recorded terminal status; only the two formal jobs remain.

See `evidence/startup-acceptance.json`, `evidence/release-{exact,block}.json`, `evidence/numerical.json`, `evidence/ddp-scaling.json`, and `evidence/batch-benchmark.json`. Full formal metrics are `runs/{exact,block}/metrics.jsonl`; validation every512steps, resumable checkpoint every256steps (latest two retained). Final val/test each4,194,304supervised tokens. The final partial update has508,928effective tokens and15,360masked extra compute positions, preserving exactly6.8B effective inputs and supervision.

## Previous mb16 × accumulation2 migration (superseded updates)

Both models now run microbatch16/GPU ×8GPUs ×2 accumulation =256 sequences/update, with the same6.8B effective-token target each. No sweep was performed. Total resources remain16 H10080GB GPUs.

- Exact cutover: step2304 /1,207,959,552tokens, new job `komp340m-exact-mb16-20261106`, identity `job-2553abb5e603-20261106084248`.
- Block cutover: step4096 /2,147,483,648tokens, new job `komp340m-block-mb16-20261106`, identity `job-2d7b70dde370-20261106084519`.

Both old jobs stopped at a complete checkpoint, with all8 workers and launcher paused. Last metric exactly matched checkpoint step; no logged complete update was rolled back. Original checkpoints, metrics, source and old batching config remain intact. Old jobs were deleted and absence verified before replacement jobs requested resources.

New source is isolated at `migration-mb16/runtime`, manifest `88f7c3e984d21829cb62fd4da7795eb6d2a04d7cd7f466909588ea5765cee635`; batching hash `c7ad3e6653337d0a322e7bd2c92a84e079c87b3c95be1abef0f59bf9f096938e`. Per-model migration receipts bind every source-checkpoint file SHA to the new source/data/batching. Full model, Adam moments/steps, CPU/CUDA/Python/NumPy RNG, JSONL byte offset and buffered token IDs were restored. Each performed one temporary real-data global256 update with finite gradients, then discarded it and restored original state again. Trial steps count as zero formal training tokens. Both first formal losses exactly match their corresponding trial loss before updating.

Latest acceptance: Exact2346 (42new updates), Block4124 (28new updates), continuous step numbering. Original30step medians versus new warm samples:

| Model | Old mb4 | New mb16 | New peak allocated/rank0 | New warm samples |
|---|---:|---:|---:|---:|
| Exact |5.717s |5.544s |13.98GB /13.02GiB |30 |
| Block |3.225s |2.772s |14.97GB /13.94GiB |27 |

Exact update time improves about3.0%; Block about14.0%. This is an early before/after comparison, not a controlled hardware benchmark. The changed batch shape can change floating-point trajectories; bitwise equivalence to uninterrupted mb4 is not claimed. State/projection precision and compression rules are unchanged.

Current metrics: `migration-mb16/runtime/runs/{exact,block}/metrics.jsonl`; prior trajectory remains `runs/{exact,block}/metrics.jsonl`. Current evidence: `migration-mb16/runtime/evidence/migration-{exact,block}.json`, `migration-complete.json`; resource receipt `migration-mb16/jobs-after.json`. A collective pause hook now supports future graceful checkpoint stops. Compute positions are counted using actual microbatch sizes, separately from effective/target tokens.

## Historical: requested mb32 × accumulation1, resumed from existing checkpoints

Per user's explicit immediate-stop instruction, both mb16 jobs were deleted without waiting for another checkpoint. Frozen final branches are `migration-mb32/superseded-mb16-{exact,block}-metrics.jsonl`:

| Model | Last old logged step | Resume checkpoint | Discarded complete updates | Recomputed tokens |
|---|---:|---:|---:|---:|
| Exact |2395 |2304 |91 |47,710,208 |
| Block |4226 |4096 |130 |68,157,440 |

These mb16 updates are superseded; never sum them into current effective progress. Valid trajectory consists of original mb4 through2304/4096 plus the mb32 segment, using the absolute token counters in current metrics. Original mb4/mb16 checkpoints, sources, configs and logs remain preserved.

Current microbatch32 perGPU ×8GPUs ×accumulation1 =global256; each model still targets exactly6.8B effective inputs/targets. Only two jobs run (total16GPUs):

- Exact `komp340m-exact-mb32-20261106`, identity `job-f54d0855a537-20261106085532`.
- Block `komp340m-block-mb32-20261106`, identity `job-fb3d4abed8aa-20261106085543`.

Source `3e3125acd34a2a3284952aae679fff9d2b55a104a0c4c825fc5a1bb038a76ee4`, isolated path `migration-mb32/runtime`. Receipts explicitly bind preceding mb16 source/checkpoint hashes and user-authorized rollback. Full model, Adam, RNG, byte/token-buffer loader state restored; temporary real B32 update passed all-parameter finite-gradient/update checks, was discarded, then original checkpoint restored again. Both first formal losses match their trial losses exactly; trials count zero budget tokens.

Latest acceptance: Exact2322 (18newsteps), Block4128 (32newsteps). Old last30 mb16 steps compared with new warm mb32:

| Model | mb16 median | mb32 median | Warm samples | Peak allocated/rank0 |
|---|---:|---:|---:|---:|
| Exact |5.532s |5.602s |17 |19.92GB /18.55GiB |
| Block |2.754s |2.857s |30 |21.96GB /20.45GiB |

The larger batch uses more memory but is slightly slower in these early observations (~1.3% /3.7% step-time increase); user-requested mb32 remains active. This before/after observation is not a controlled hardware benchmark. Batch shapes can change floating-point training trajectories; no bitwise equivalence claim.

Current logs, validation and checkpoints: `migration-mb32/runtime/runs/{exact,block}/`. Current migration evidence: `migration-mb32/runtime/evidence/migration-{exact,block}.json` and `migration-complete.json`. Resources: `migration-mb32/jobs-after.json`. Follow these paths instead of either old runtime.


## Current migration: native FLA, mb16 × accumulation2

User requested the native FLA model path and then microbatch16/GPU ×2 accumulation, preserving global256 sequences. The new isolated source is `migration-fla-mb16/runtime`; manifest SHA `1dde0fbb5668f1105425de6774a4d188535b278819d3af3f1c569a987359f5ea`.

- Exact resumes native FLA checkpoint2492 /1,306,525,696tokens directly, including Adam. The intermediate B32 native run completed exactly one formal update after the2491 legacy checkpoint, then gracefully paused. Zero complete updates rolled back.
- Block resumes legacy checkpoint4458 /2,337,275,904tokens, mapping parameters and Adam moments to native FLA split projections. No prior FLA Block formal steps ran.
- New jobs: `komp340m-exact-fla-mb16-20261106` (`job-fc9daba3ba43-20261106094504`) and `komp340m-block-fla-mb16-20261106` (`job-0cae6617a482-20261106094507`),8GPUs each.
- Registered KOMPConfig/KOMPModel/KOMPForCausalLM; native FLA block/residual/loop, native fused linear cross entropy. Native GatedMLP projection layout uses compatibility SiLU rounding; fused SwiGLU remains disabled. Selective mixer checkpointing, no full-layer or256-token head checkpoint loop.
- New actual full/tail tests use2microbatches/rank, same32sequences, firstmicro no_sync, loss divided by total local valid tokens. Tail uses32768+30848 valid positions/rank. Migration trials are discarded and full model/Adam/RNG/loader restored before formal continuation.

The frozen acceptance policy distinguishes strict trajectory compatibility (same original per-tensor .035RMS/.04max,1e-7nearzero allowance) from native training acceptance (loss absolute difference<.005, whole-model gradient and update relativeL2<.035, finite gradients/parameters, causal prefix,8replica equality and full state restoration). Policy was frozen before any mb16 GPU result. Strict failures remain failures even when native training is accepted; complete distributions and absolute tensor norms are retained.

Previous native B32 evidence remains in `migration-fla/runtime/evidence`: Exact strict full/tail passed; Block strict full failed11tensor comparisons, tail failed9. Worst gradient RMS5.226%/6.850%; worst Adam-update delta4.221%/4.209%. Loss differences1.49e-7/1.79e-7. Those Block failures stopped that pipeline before training. Original T257 diagnostic failures and all earlier sources remain intact.

Current nativeFLA mb16 acceptance: Exact has25continuous formal updates2493–2517 after complete checkpoint2492, latest1,319,632,896effective inputs/targets, loss2.97177948. First formal loss3.1075834333896637 exactly matches the discarded migration trial. Full model/Adam/RNG/loader restored and zero rollback. Last24warm updates have median5.3433s/global524288tokens,98,120tokens/s across8GPUs; rank0 peak allocated34,464,085,504B=32.098GiB. Compared with historical legacy mb16 median5.532s and legacy mb32 median5.602s, this early observation is about3.4% and4.6% faster respectively; it is not a same-hardware controlled benchmark. NativeFLA B32 had only one formal step and has no steady-state throughput comparator.

Exact strict trajectory compatibility remains FAILED: full/tail each2failures, both gradient and update of `model.layers.0.attn.k.weight`. Whole-model gradient relativeL2 is0.345%/0.439%, update relativeL2 is0.399%/0.448%; loss difference0, causal prefix difference0, all8replicas identical. This passes the independently predeclared native training policy, with strict failures and absolute norms fully retained. Across8rank trial peak allocation is34,460,676,096B. B16 measured storage estimates justify mixer-only checkpointing:34.94GiB selective vs97.15GiB without checkpointing.

Block has now started automatically on the same original job identity and completed the full migration, with no resubmission or configuration changes. Full/tail native training acceptance passed: whole-model gradient relativeL2 1.219%/1.247%, update relativeL2 0.841%/0.876%, loss absolute differences1.34e-7/1.69e-7, causal prefix difference0,8replicas identical, finite parameters/gradients. Strict compatibility remains FAILED with11tensor comparisons in each case; full worst gradient5.226% (`layers.19.mix.convs.0.weight`), tail worst gradient4.814% (`layers.18.mix.k.weight`). All original thresholds and complete distributions remain preserved.

Block source/checkpoint/compat/batching hashes, full Adam/RNG/loader restoration, zero rollback and zero trial-token budget verified. Actual first formal update4459 follows checkpoint4458, and its loss2.8338289856910706 exactly equals the discarded trial loss. Captured36continuous steps4459–4494; latest2,356,150,272effective tokens, loss2.86105401814. The first20warm steps after startup have median2.54575s/global524288tokens,205,946tokens/s across8GPUs. Peak rank0 allocated35,515,019,264B=33.076GiB. Compared with historical legacy mb16 median2.754s and legacy mb32 median2.857s, this is about7.6%/10.9% faster in an uncontrolled before/after observation.

Both original nativeFLA mb16 jobs now Running,8GPUs each,16total; no queue mutations were needed for Block startup. Exact continues uninterrupted and reached checkpoint2560 with validation CE3.02045023651/PPL20.50051969 over4,194,304tokens. Existing30minute monitor tracks both current run directories. Complete paired migration receipt: `migration-fla-mb16/migration-complete.json`; Block raw startup36steps/20warm statistics: `block-startup-metrics.jsonl`/`block-startup.json`; resource snapshot: `jobs-both-running.json`.

Current paths are `migration-fla-mb16/runtime/runs/{exact,block}/`. Startup receipt is `migration-fla-mb16/startup-acceptance.json`; captured25step metrics and warm statistics are `exact-startup-metrics.jsonl` and `exact-startup.json`. `jobs-current.json` records the queue snapshot.
