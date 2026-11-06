# DDP startup stall diagnosis

Initial exact8GPU jobs passed Exact, GSA, Full, then stalled in GDN/GSA2 without emitting any optimizer-step log or ticket. First diagnostic signal: shared Triton cache had zero changes for120seconds; parent observed87.5% GPU utilization but~0.00663 SM activity and~900W across8H100s, compatible with7ranks polling NCCL. This does not prove the exact wait site. Default kubectl was Unauthorized and platform exposed no verified exec endpoint; node-name SSH DNS failed. No guessed endpoint or credential changes were used.

Read-only static findings:
- Native GDN calls chunk_gated_delta_rule without cp_context; native GSA2 dense/packed branch only changes tensor layout. No automatically activated CP collective was found.
- Global update microstep count is computed from all128rows, and no_sync encloses forward+backward. Uneven tail ranks execute graph-connected zero losses. This passed Exact/GSA/Full.
- A rank0-only reference or comparison exception could previously enter finally destroy_process_group before printing its traceback, hiding the error while peers wait.
- Native Triton autotuning may synchronize CUDA inside backward. If DDP buckets overlap with new rank-local shapes, a rank could wait for a collective before another rank gets through autotuning. This remains a hypothesis pending stack dumps. An explicit512MB DDP bucket would prevent overlap for this100M model, but has NOT been applied without evidence.

Two original owned jobs were deleted and absent verified. Diagnostic source adds rank-stage prints,90second faulthandler stacks,180second preflight process-group timeout, a Gloo control group for rank0 reference/comparison error broadcast, and exception traceback+flush+os._exit to ensure torchrun tears peers down. All7tickets must match the new trainer hash before formal training. New jobs are Pending; no diagnostic rerun has executed yet.

Next acceptance:
1. Wait for the two existing jobs; do not submit duplicates.
2. Read logs/ddp-preflight-gdn.log and gsa2.log for STAGE and timed stacks. Archive first failure; investigate all instances statically before another rerun.
3. Resolve actual root cause without changing accepted kernel semantics, frozen data, selections or100M-token budget.
4. Refreeze changed trainer/runner hashes; all source-dependent tickets must pass.
5. Require both task groups to log actual world_size8 optimizer-step progress, preferably step1and25. Running and prior tickets are insufficient.
6. Sync evidence and update STATUS.md. Never mix runs/ superseded singleGPU state into runs-ddp/.
