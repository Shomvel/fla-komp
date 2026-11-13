# KOMP / KDA downstream evaluation

Frozen reference: BP Dev `/gpfs/users/linlizhi/flash-linear-attention/run_eval_batches.sh`
and its local lm-evaluation-harness fork (git a43c1dba6751cfaf8ffb886f2342c0cf3c2f5239,
actual source copied into `source/lm_eval`). Flame training repository HEAD
ecdcbed725e293184d129acb75791122dfaef969. Neither shared checkout is modified.

One independent Northjob 8-H100 evaluation job in WeaverRSI/proj-32, one model
per GPU. Six completed final checkpoints initially available; old Value waits
for final complete.json. Original Exact checkpoint7168 is reported separately
as partial 3.758B tokens, never as a final 6.8B peer. New value+learned+123 training
is not included. Strict source/weight/step/token/data validation before evaluation.

Exact original task prompts, prefixes, generation limits and contains/string-match
scoring are materialized once with the original conda Transformers4.49 tokenizer;
raw tokenizers JSON encodings are checked against it. No chat template/BOS.
NIAH single1/2/3 at1024/2048/4096/8192,500 each; PhoneBook32/64/100/128/150/200/256,
1000 each; SWDE1111,FDA1102,NQ3157,DROP2087. Total20457/model. NIAH max128 new
symbols, PhoneBook20 with newline stop, BASED48 with newline stop. Greedy decoding,
EOS stop. Training context2048:4096/8192 are extrapolation tests. Existing PhoneBook
source is `phonebook_datasets_single_break`: task suffixes are legacy labels,
actual prompts only ~80–244 tokens; do not call them long-context tests.

The original adapters return hidden states, so evaluation explicitly applies
lm_head at the final real token. No generation cache is invented. Full prefix is
recomputed. R1 padding compatibility failed for original Exact (RMSrel.00414,
top1same); r1 was stopped, all evidence preserved under evidence/r1-padding.
R2 uses true sequence lengths and only batches equal-length prompts. Acceptance
uses real prompts, same-shape logit repeatability and tiny greedy continuations;
new lengths can cold-compile. Source/checkpoint/requests hashes retained.

Outputs append one record/sample and are fsynced; each case has summary only
when all requests finish. `python3 collect.py` is read-only and saves latest status.
No scores from incomplete samples are presented as full scores. SQuAD and general
LM sanity tasks from the wider old script are deferred; this run covers all four
user-named families plus NQ/DROP. All historical task sources remain available.

R3 consolidated contract audit: historical actual model config and HFLM property
resolve BASED max_length=2048, reserving48 new tokens and left-truncating to2000.
NIAH cap65536 and PhoneBook cap32768 are explicit historical model arguments.
All20457 request encodings are checked against original HF fast tokenizer;
representative shortest/longest requests are compared directly with historical
HFLM tok_batch_encode. Full-vocabulary decoding parity, cleanup=false, no BOS,
EOS2, greedy defaults and original task kwargs/filters are checked before launch.
Each actual runtime writes its config and asserts max_position_embeddings2048.

Numerical protocol deliberately retains frozen training FP32 parameters/residuals
and gate projections, BF16 autocast, and no cache. Historical models were loaded
as BF16 with caching. Frozen learned low-rank projections require FP32 weights
inside autocast-disabled gate code; blindly converting all parameters is invalid.
Therefore these are comparable KOMP/KDA runs using historical task definitions,
not claims of bitwise replay of historical model inference. Batch size<=8 with
identical effective lengths avoids unsupported attention masks and unsafe padding.

R2 was stopped before BASED execution when its generic context cap was found
inconsistent. Its scripts, acceptances and PhoneBook samples are archived under
`evidence/r2-context-audit`, not mixed into r3 formal scores. Source manifest r3
includes the audit and generation scripts. Exact original checkpoint7168 remains
partial and is always labelled separately from final-checkpoint comparisons.

## R4: authorized 24-GPU fixed-batch sharding

R3 was intentionally stopped by exact identity `job-e0d03ca6e2b4-20261107133839`
and confirmed absent before migration. This was a resource expansion, not a
numerical failure. Its result bytes and evidence remain in `sharded/r3-archive`.
Migration retained41,606 completed sample records, no incomplete tails/partial
batches. `sharded/migration-audit.json` records per-model inherited counts.

Three8-GPU jobs `komp340m-downstream-eval-20261107-r4-s0/s1/s2` each assign one
GPU per model. Each model's original whole batches are allocated round-robin
across3 workers:6795/6830/6832 samples total per shard before resumed exclusions.
Original prompt order, batch shape, greedy generation and scoring stay unchanged;
AST comparison proves logits/generate/score/context helpers identical to r3.
`sharded/plan.json` is immutable and checkpoint hashes are checked at runtime.
Each model/shard has an exclusive flock claim and owns disjoint atomic batch files;
interrupted incomplete batch work is recomputed with the original complete batch.
No training job, source or checkpoint is modified.

Current outputs are `sharded/batches/<model>/s<shard>/<batch>.json`.
`collect_sharded_remote.py` validates exact IDs, checkpoint/plan provenance and
no duplicates before publishing complete-case summaries. `collect.py` invokes it
and the Northjob status helper, saves all job statuses and worker states into
local `evidence/latest.json`, and refreshes `PROGRESS.md`. Read these r4 artifacts
rather than stale r3 progress files. R4 `sharded/source-r4.json` and job receipts
pin the implementation and three identities. Do not run prepare_shards.py twice.

## R4b: cross-node namespace ownership

R4's flock was found not to exclude access from the BP SSH client; its three exact
job identities were stopped and confirmed absent. Diagnostics are preserved in
`sharded/r4-flock-archive/flock-audit.json`; all completed atomic batches remain.
R4b uses persistent atomic mkdir claims under `sharded/claims-v2` with exact
owner job identity. Never reclaim by TTL. A claim can be archived/replaced only
after its named owner job is verified terminal/absent. A duplicate worker fails
before loading or generation. Batch publication fsyncs a private inode and uses
hard-link creation without overwrite; existing content must match provenance,
IDs and results. Collector uses independent mkdir claim `sharded/collector-claim`
with host/pid owner and finally cleanup; a crash requires verifying that owner
before removing its claim. Atomic replace applies only to mutable summaries and
status snapshots, never to published batch outputs.

24 worker claims and publication probes were created in3 GPU pods. An independent
BP SSH storage client confirmed all24 duplicate mkdir and hard-link destination
creations are rejected, with unchanged probe bytes. This establishes the tested
cross-client namespace behavior; it is not a simultaneous worker-to-worker race
benchmark. Proof: `sharded/cross-node-verified-r4b.json`.

Current jobs: r4b-s0/s1/s2, identities in local `evidence/jobs-r4b.json` and remote
`sharded/job-r4b-s*.json`. Current worker states: `sharded/evidence-v2`, batch and
plan paths unchanged. `collect.py` now reads r4b jobs and all24 worker states and
updates local `evidence/latest.json` and `PROGRESS.md`. Old r4/evidence status is
archived/historical. Exactly41,606 r3 samples were inherited; additional successful
r4 samples are also retained. No inference arithmetic, inputs or batch shapes
were changed by r4b. Persistent per-worker Triton caches are reused.


## Authorized s0 recovery at 2026-11-07 19:21 CST

Old r4b-s0 `job-bdaafde6397e-20261107143915` was Aborted with zero instances; reason remains unknown (no platform events/exit reason or application failure receipt). New `komp340m-downstream-eval-20261107-r4c-s0`, identity `job-de184fdde2fc-20261107192117`, replaces its8-card allocation. Only model indices1–7 launch; KDA-s0 remains completed and is not rerun. Seven persistent s0 claims explicitly migrated after identity/terminal checks, old claims/evidence retained in `recovery-r4c-s0/claim-evidence-archive`. Frozen runner, task plan and4765 prior s0 batch files are unchanged. Resume publishes only absent original batches.

Recovery receipts and before/after audit are under `recovery-r4c-s0/`. Current collector `jobs` contains r4c-s0 plus r4b-s1/s2; `historical_jobs` retains old aborted job state. No s1/s2 or training changes. Evaluation allocation ceiling remains24 GPUs.


## Authorized s1/s2 recovery at 2026-11-08 01:33 CST

The user authorized continuing the interrupted evaluation. Old r4b-s1 `job-ed375f508254-20261107143937` and r4b-s2 `job-01a5e891f391-20261107143959` were verified Aborted, with only terminal Failed instances; the platform did not provide a cause. Replacements are `komp340m-downstream-eval-20261108-r4d-s1` (`job-b975c4a6b8da-20261108013315`) and `komp340m-downstream-eval-20261108-r4d-s2` (`job-ab970a5f32c2-20261108013328`). Each reserves 8 GPUs in WeaverRSI/proj-32; active evaluation remains r4c-s0 plus r4d-s1/s2, ceiling 24 GPUs.

Only indices 1,2,3,4,5,7 resume on each replaced shard. KDA and old Value are complete and skipped. Frozen runner, plan, checkpoint, original batch shapes and 12,874 existing complete batch files are preserved. Explicit claim migration follows terminal-owner verification; old claims and evidence are archived. Read `recovery-r4d-s12/` for receipts, before hashes, migration and recovery audit. Loading acceptance and new batch publication determine successful recovery. Original s0 and training are unchanged. The collector and existing hourly heartbeat now target these replacements; old r4b jobs are historical.
