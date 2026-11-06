# GSA2 real-data numerical diagnosis, 2026-11-05

The original two formal GSA2 runs await a fresh distributed preflight. The parent has authorized a same-rank-local-shape distributed acceptance boundary with explicit numerical limitations. None of the isolated candidates below has been deployed into the frozen model/kernel snapshot, and the strict whole-batch/partitioned Adam witness remains failed.

## Established forward causes and isolated corrections

The witness consists of the same 17 real Depo2 documents padded to 960 positions, seed 731, 15 layers and BF16 network arithmetic. Reordering the documents, changing other documents, and repeating the identical shape give exactly identical outputs. Local/global tail padding does not explain the discrepancy.

For the worst first-layer document (index 10), packed whole-batch versus partitioned inputs produce identical projection, convolution, Oja and GDN2 outputs. The first difference is the sigmoid-gated RMSNorm output (relative RMS 4.3078966e-5). Its Triton autotuner selects BT/warp configurations by a row-count bucket. Fixing BT=32 and num_warps=4 makes all 15 layer outputs and final normalization exactly identical between the whole batch and actual rank-local microbatches, including different padding lengths. This reduces the original final hidden RMS difference from 18.337% to zero.

Dense evaluation versus packed training has a second source: with identical captured operator inputs, Oja output differs slightly across dense/packed layouts. Making cache-free long-sequence evaluation use the same packed layout as training reduces the original 33.177% final hidden RMS difference to zero on the same witness. The adapter preserves the native short-sequence recurrent path, cache behavior and original mathematical recurrence.

These are demonstrated witness results, not a claim covering every sequence length or trained checkpoint.

## Remaining gradient/update sensitivity

| Candidate | Mano embedding-gradient RMS difference | Depo embedding-gradient RMS difference | Mano post-Adam hidden RMS difference | Depo post-Adam hidden RMS difference |
|---|---:|---:|---:|---:|
| Fixed gated RMSNorm, original BF16 head | 2.105% | 1.699% | 67.07% | 67.77% |
| Fixed gated RMSNorm + packed evaluation, whole FP32 network | 46.85% | 71.71% | 106.64% | 120.16% |
| Fixed gated RMSNorm + L2Norm + packed evaluation, FP32 loss head only | 1.769% | 1.935% | 59.33% | 67.90% |

All three retain the failed strict verdict; a small aggregate gradient difference alone does not establish functional equivalence after Adam. The FP32-head variant reduces head-weight gradient differences to approximately 1e-7 relative RMS but does not resolve the model-level update sensitivity. Neither precision change is recommended for formal deployment based on this evidence.

A separate zero-optimizer diagnostic fixes both gated RMSNorm and L2Norm configuration and uses exactly the same upstream cotangent (seed 831) for the whole batch and each partition. All 15 activation-gradient tensors then agree exactly. Embedding parameter-gradient RMS difference is 2.80e-7; first/last output-projection weight-gradient differences are 0.2377% and 0.2348%. This distinguishes the deterministic per-token backward computation from BF16 parameter-gradient reductions and the sensitivity to real head/CE upstream perturbations. It is not a replacement for the failed Adam witness.

## Artifacts and release boundary

- Original failure: `gsa2-bf16-adam-witness.json`.
- First-divergence localization: `gsa2-worst-layer.json`.
- Fixed norm forward/Adam: `gsa2-fixed-norm-layout.json`, `gsa2-fixed-norm-adam.json`.
- Train/eval canonical layout: `gsa2-packed-eval-layout.json`.
- FP32 network and FP32 head: `gsa2-fp32-candidate-adam.json`, `gsa2-fp32-head-adam.json`.
- Shared-cotangent backward: `gsa2-backward-stages.json`.
- Prepared runtime, submitted for preflight only: `../ddp_train_gsa2_candidate.py`; binding `gsa2-candidate-source.json`; the earlier `gsa2-candidate-draft.json` is historical.

The prepared runtime retains BF16 model/head, original hyperparameters and budgets, pins both normalization launch configurations, and unifies cache-free long evaluation layout. It writes separate `runs-gsa2-consistent` results and binds candidate code in each source identity. Old preflight tickets cannot satisfy its source check. The parent has approved the documented same-rank-local-shape acceptance boundary; the new 8GPU differential preflight must still pass before training. This preserves the original thresholds and includes real BF16 reference-gradient/update comparisons. No formal release marker has been written. Every one-GPU diagnostic allocation above has completed and been deleted, with exact terminal/delete/absence receipts preserved.
