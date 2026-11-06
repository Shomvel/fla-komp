# Block-KOMP learned/decay variants

User requested three variants, reusing prior learned eviction and data-dependent decay work. Original 100M KOMP already uses input-dependent GSA per-slot decay. This experiment tests:

- `komp_learned`: original GSA per-slot decay + learned hard top-M eviction + `beta *= 2 sigmoid(score)` (score learns through writing, selection is detached).
- `komp_gdn_slot`: FIFO + GDN per-slot `log alpha = -exp(A_log) softplus(Wx + dt_bias)`.
- `komp_learned_gdn_slot`: both additions.

Initial alpha=.98 and beta=.2; score initially zero, initial atom scores fixed zero buffers. GDN A_log/dt_bias excluded from WD. Original baseline has no timescale parameters; this optimizer difference is part of the new parameterization. Shared backbone/conv/dictionary initialization and FFN width exactly match original KOMP; additional parameters do not trigger FFN resizing. M32,C128, shortconv4, width512,15layers, batch128 unchanged.

Read-only references: `~/Downloads/komp-slot-attention-prototype/physics_lm/block_experiment.py`, `physics_lm/decay_experiment.py`, frozen `physics100m-20261101/snapshot/komp_slot/decay.py`. No kernel modified. Log decay computed directly in FP32 to avoid exp/log underflow; GPU check compares formula and gradients to existing Triton decay helper. C128 operator fwd/state/all-input gradients are compared with frozen flex; causality tested across128/256 boundaries with nonzero learned scores, and added parameters must receive finite nonzero gradients. Full-model loss descent and causal preflight runs before tuning, with source-hash tickets.

Remote root `/gpfs/users/linlizhi/physics100m-komp-variants-20261102`; original data and pinned KOMP/FLA snapshot are shared read-only. New local sources are under `suite100m/`. Do not change original frozen suite or kernels.

Five tasks separately; LR{.00025,.0005,.001} x WD{0,.01,.1} on10M tokens,seed731 (135 tuning runs); pick each task/variant's lowest final validation CE; then100M tokens at seeds731/732 (30 final runs) plus testCE and256 generation probes. Compare against original KOMP final results, never tuning versus final. No new data generation.

GPU aliases: `physics100m-komp-variants-20261102-r1` onward. Transfer idle original GPU slots rather than exceed16total. Original r16 confirmed idle, deleted and queried absent; receipts under evidence/old-r16-*.json. New r1 runs the three preflights then runner. Other workers launch only after preflights pass. Consult receipts for actual launch state.

Original60runs continue independently. Its remaining training and evaluation must not be stopped for this experiment. Monitoring must track both roots; all original60+new30 formal runs and evaluations need completion before final cleanup. Store exact job state/deletion/absence receipts. No checkpoint copied locally.

## GPU acceptance recovery, Nov2 06:59 China

First preflight job `job-354cda09a28d-20261102064815` passed learned acceptance and then stopped on FIFO final-state order comparison. Static audit confirmed fast FIFO returns chronological atoms while flex returns physical slots. The acceptance script now rotates both reference state tensors within autograd and checks the returned keys against the latest32input keys, so state and gradient comparisons address the same atoms. No kernel changed. All old tickets/logs were archived remotely; local source manifest and remote hashes match. Failed job terminal, exact delete and absent-query receipts are `evidence/preflight-r1-{terminal,delete,absent}.json`.

Replacement `physics100m-komp-variants-20261102-r1`, identity `job-dc890341291f-20261102065925`, is Running and repeating all three source-pinned preflights before allowing training. Submission: `evidence/new-r1-resubmit-state-order-fix.json`. No training result yet at this update.

## Acceptance passed, Nov2 07:03 China

All3 preflights passed after state-order correction. Source-pinned tickets and logs are in `evidence/preflight-passed.json`; compact metrics in `evidence/acceptance-summary.json`. All3 shared-parameter SHA256 fingerprints match exactly. Max operator output absolute error 0.0001455; elementwise forward/state and all-input-gradient tolerance checks passed. Initial decay .98, causality across C128 boundaries, finite/nonzero score and decay gradients, GDN formula/gradient equivalence, and full-model loss descent all passed. First10M tuning run claimed on newr1. Worker transfer helper is running; only receipts establish which old/new aliases have been changed.

## Worker transfer completed, Nov2 07:08 China

Original r1,r3,r4,r6,r7,r8,r9,r10,r12,r14,r15,r16 were deliberately retired after idle checks. Terminal/delete/absence evidence is in the variants experiment evidence directory; r15 absence receipt uses `old-r15-absent-verified.json` because the first read-only helper response was empty and was not accepted as evidence. Do not recreate these original aliases. Original r2,r5,r11,r13 remain active for Lano KOMP evaluation and Mano GSA2 training. New variants r1..r12 submitted; allocation and platform identities recorded in `evidence/allocation.json`. Total16 GPUs, no other jobs touched. All three variant preflights passed; tuning is now claimed and kernels are compiling for actual batch/length shapes. No new formal score yet.

Latest launch verification: all12 new workers have claimed 10M tuning specs; no FAILED.json, no completed tuning or formal result yet, and no first25-step metrics at this observation. Actual training-shape Triton compilation is writing new backward cache artifacts. Do not report these jobs as completed training or quality results. Latest snapshot `evidence/launch-progress-6.json`; monitoring state `evidence/monitor-state.json`. The existing10-minute heartbeat remains ACTIVE and includes both experiments.

## Original baseline completed and cleaned, Nov2 07:31–07:37 China

Original60/60 formal training and60/60 evaluations completed. Report `/Users/shom/Downloads/physics100m-20261101/FINAL_RESULTS.md`; per-run tables/JSON under its `results/`. Independent generation scoring audit checks15360records with no mismatch; Capo uses saved target tokens and CE was not recomputed from checkpoints. Logs,330run summaries,30selection files and final generations are archived locally without*.pt. `evidence/BASELINE_COMPLETE.json` verifies completion, source fingerprints and cleanup. Last originalr2/r5/r11/r13 observed Completed, deleted by exactalias, queried absent; receipts in original `evidence/cleanup-original-20261101T2331Z/`. All16 original aliases are now intentionally absent. Never recreate them.

The only remaining jobs are newvariant r1..r12. Existing10-minute automation remains ACTIVE for these135tuning→30formal runs and evaluations, and was updated with baseline completion. Original frozen sources remain unchanged. Latest newvariant probe still has no25step training metrics, no completed tune and nofailure; cache activity confirms ongoing compilation but does not establish optimizer-step progress.

## First training metrics confirmed, Nov2 07:51 China

All12 active Brevo2 tuning runs have now written optimizer step25/271, each with832685 consumed tokens and finite loss/gradient norms. This confirms actual training progress after compilation, not just Running job state. Nine GDN-slot configurations and three learned configurations are active; the combined variant remains queued.0/135tunes completed,0/30formal results,123pending tasks. NoFAILED or controller error. Snapshot `evidence/monitor-20261101T2351Z.json`; source/kernel hashes unchanged. Compilation remains active for further shapes.

## First variant deadline recovery, Nov2 11:00–11:12 China

All135 tuning runs completed and all15 selections validated. The initial12 variant workers reached their four-hour deadlines and were each observed Aborted0/1. Each exact alias r1..r12 has identity-pinned terminal, delete, confirmed-absent, unchanged-spec requeue and replacement-submit JSON under `evidence/recovery-20261102T0257Z/`. All12 replacement jobs are Running;12 interrupted formal training commands resume their remote checkpoints. r1 metrics have advanced beyond its pre-deadline position. No FAILED marker; no formal summary yet at this check. Original frozen sources and variant training/kernel hashes remain unchanged. r1 replacement skips already-passed preflights and runs the worker directly, preserving prior preflight logs/tickets. Do not run the recovery helper again for these new identities; its expected identities are the old generation. Next deadlines must be calculated from replacement submission identities in `jobs-after-all.json`, around07:05–07:12Z, not the original02:59–03:08Z deadlines. Monitoring state stores the current identities and notified recovery.


## Evaluation source dependency repaired, Nov2 14:19–14:23 China

26/30 formal training summaries now complete. Mano GDN-slot validation CE at seeds731/732 is1.442147/1.449136 versus original KOMP1.443853/1.492291; no complete variant heldout/generation results yet. Four Mano runs continue.

Capo evaluation first failed because the variant suite lacked its shared `source` directory. Static audit identified the same dependency for Lano; restored `suite100m/source` symlink to the original frozen source tree. All five factories and grading inputs verified; all256Capo test probes independently regenerated exactly. No training/kernel Python files changed. Failed log/claim/done/FAILED marker archived remotely under `evidence/repair-source-20261102T0619Z/`; local repair script and receipt stored in the matching directory. The unchanged failed evaluation was requeued, FAILED barrier released, and existing workers2/4 claimed Capo evaluation. No jobs recreated. Snapshot `evidence/monitor-20261102T0619Z-after-repair.json`.

Monitoring interval is30minutes. Current12job identities remain Running and match monitor-state. Next deadlines07:05–07:12Z. Six slow Brevo evaluations reached48/256probes; partial accuracy is not a formal score.


## All variant training complete, Nov2 14:49 China

30/30 formal100Mtraining runs completed; sixCapo heldout+generation evaluations complete. All1536Capo saved predictions independently compared against target tokens and matched saved scores. Mano learned validationCE731/732=1.3626758/1.3857173; combined=1.4921087/1.5245024. Complete per-seed results are in `results/latest-all/TABLE.md` (90formal runs,66complete evaluations including baseline).

Twelve existing workers remainRunning with identities matching monitor-state. Active: sixBrevo evaluations at48–64/256probes, sixDepo at24/256;12evaluations queued. NoFAILED/controllererror. Next deadlines07:05–07:12Z; only recover after exact terminal verification. Current evidence: `evidence/monitor-20261102T0649Z.json`, `evidence/jobs-20261102T0649Z.json`, `evidence/comparison-20261102T0649Z.json`, `evidence/capo-score-check-20261102T0649Z.json`.


## Second variant deadline recovery and evaluation resume, Nov2 15:19–15:26 China

All12generation2 workers reached4hdeadline and were observed Aborted0/1. Exact current identities pinned from `evidence/jobs-20261102T0719Z.json`; terminal/delete/confirmed-absent/unchanged-requeue/submission receipts are under `evidence/recovery-20261102T0719Z/`. All12generation3 workers are Running; current identities saved in monitor-state and `jobs-after-all.json`. All30formal training and sixcompleteCapo evaluations preserved. Next deadlines are approximately11:23–11:26Z (19:23–19:26China). Do not rerun either recovery helper unchanged: each pins its old generation.

Static audit found standalone evaluator saved predictions only after all256probes, so deadline interruption discarded unfinished generation records. Only variant-root `evaluate.py` and new `eval_progress.py` changed; frozen training/kernel source manifests verified unchanged before deployment. Evaluator now atomically saves `eval-progress.json` after testCE and every8generation probes, validates run/summary/probes/evaluator/checkpoint identity and prefix+grades, and resumes from completed batches. Sampling, batch size, stopping rules and scoring unchanged. Local `test_eval_progress.py` exercises interruption→restart equivalence through the actual generation loop with a fake model, truncated temporary write, wrong identity, reordered probes and corrupted score rejection. Successful test log and pre-edit evaluator backup retained in recovery evidence. Existing completed evaluation files unchanged.

Old aggregate logs lack predicted tokens, so earlier incomplete predictions cannot be reconstructed; this first evaluation restart uses the persistent compilation caches. Subsequent interruptions can resume persisted batches. Ten replacement evaluators have already written durable progress (up to144/256); noFAILED. Probe script now collects `evaluation_progress` counts. Partial progress is never counted as a completed heldout result. Evaluation seconds include saved attempts since this change, not discarded work from older attempts. Monitor remains every30minutes.


## Variant evaluations 22/30 complete, Nov2 16:19 China

SixMano, sixDepo2 and fourBrevo2 evaluations have now completed in addition to sixCapo. Original independent `audit_final.py` rescored all5632records across22runs with zero mismatches; evidence `evidence/generation-audit-20261102T0819Z.json`. The audit uses a separate grader implementation; Capo compares saved targets and testCE is not recomputed. Full90run report (82completeevals including baseline) updated in `results/latest-all/`. Per-seed paired comparisons saved in `evidence/comparison-20261102T0819Z.json`.

Mano learned mean testCE1.389448 / generation20.8984%, and Brevo2 learned0.552269 /20.7031%, lead currently completed models by two-seed mean. Depo2 learned0.588861 /6.25%; GDN baseline remains6.4453% generation. Combined Mano generation8.9844% trails originalKOMP10.9375%. These are descriptive two-seed results, not significance claims.

Remaining8evaluations: twoBrevo2GDN-slot at144/256probes and sixLano at0/256(first generation shapes). No pending queue items, FAILED marker or controller error. All12existingjobsRunning with matching identities, including4idleworkers awaiting completion. No restart or job mutation. Next deadlines11:23–11:26Z.


## Variant evaluations 24/30 complete, Nov2 17:19 China

TwoBrevo2GDN-slot evaluations completed: seeds731/732 testCE0.7317706653/0.7864092440, generation12.5%/8.59375%. Mean generation10.546875% trails originalKOMP12.5%; learned remains20.703125%, combined17.96875%. Independent grader checked6144records across24complete runs without mismatch (`evidence/generation-audit-20261102T0919Z.json`). Per-seed report and paired comparisons updated. Only sixLano evaluations remain, still first generation batch0/256; noFAILED/controllererror.12jobsRunning with matching identities,6idle. No restart or cleanup yet. Nextdeadline11:23–11:26Z.


## COMPLETE, Nov2 18:49 China monitoring cycle

All30/30variant formal100Mtraining runs and all30heldout+256generation evaluations completed. Original60+new30=90complete runs. Script-generated `FINAL_RESULTS.md` and `results/latest-all/` contain both-seed validation/testCE, generation scores, hyperparameters, paired comparisons, means and ranks. Independent scorer verified7680variant records without mismatch (23040including baseline). No significance claim from two seeds; CE not independently recomputed.

FinalLano mean generation: learned10.546875%,GDN-slot7.03125%,combined7.6171875%,originalKOMP11.328125%. GDN-slot seed731/732 differs substantially(13.671875%/0.390625%), so mean does not imply consistency. Overall per-task mean generation winners: learnedMano20.8984375%,originalKOMPLano11.328125%,GDNDepo2 6.4453125%,learnedBrevo2 20.703125%,GDN-slotCapo80.859375%.

`final_archive/` preserves1285JSON/log/source files with per-fileSHA256,165training summaries,15selections,30heldoutandgenerations; no*.ptcopied. Source manifest verified unchanged. All12variant jobs observedCompleted0/1,exactdeletedandconfirmedabsent; receipts `evidence/cleanup-final-20261102T1049Z/`. All16baseline aliases were alreadydeleted; do not recreate any. Final completionreceipt `evidence/VARIANTS_COMPLETE.json`. Monitoring is no longer needed and is being deleted via the app tool.

Automation `physicslm-100m` deletion confirmed by app tool; receipt `evidence/automation-deleted.json`. No further heartbeat required.
