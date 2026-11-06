# Seven-model8GPU DDP acceptance

Every model passed the same current-source checks. This verifies distributed execution and checkpoint recovery; it does not establish training quality or make the FP32 native batch-shape discrepancy disappear.

| Model | Parameters | Gradient RMS error | Updated parameter max abs |
|---|---:|---:|---:|
| gdn | 99,846,832 | 7.68e-08 | 8.01e-08 |
| full | 99,723,712 | 7.62e-08 | 1.45e-07 |
| swa | 99,723,712 | 7.62e-08 | 1.45e-07 |
| gsa | 100,713,472 | 6.55e-08 | 6.05e-08 |
| gsa2 | 99,286,192 | 8.04e-08 | 7.08e-08 |
| komp | 100,526,392 | 7.65e-08 | 1.26e-08 |
| exact | 100,526,392 | 5.83e-07 | 2.46e-06 |

The numerical reference computes the exact same rank-local microbatches sequentially on one GPU. DDP scales each local supervised-loss sum by8/global_supervised_count before its average. Thus both sides implement the same global objective with controlled kernel shapes. Checks include an uneven17-document batch, a3-document tail with5dummy ranks, replica equality after updates, all-rank RNG/model/AdamW checkpoint restore, whole-model causality, and a real long Depo2 BF16 optimizer step.

A separate original whole17-document reference was retained. GDN CE differs by3.8e-6 and selected-gradient relative RMS up to0.00470; its Adam maximum difference was nearly2×LR because of sign changes near zero. GSA2 FP32 whole-batch CE differs by0.006335 and selected gradients by up to0.686. Its layers amplify a small batch-shape difference from4e-5relative RMS at layer0 to0.0345 at layer14. This is not a DDP normalization discrepancy.

The independent GSA2 forward diagnostic found exactly zero difference at every layer for document permutation and intervention on other documents, under bothFP32 andBF16. Under the actualBF16 training path, all tested batch partitions and extra padding lengths also gave exactly identical forward outputs. Formal comparison remains gated on whole-vs-partitionedBF16 gradients and Adam updates on bothMano and realDepo2 batches.

Trainer fingerprint: `f19afcdfb6aa27bb185b05a6e615e707666df652b268020cedd2776ebfe752e2`. Snapshot manifest fingerprint: `ae087c67e60bd720377304db7e95a193d318b8b34ee1839a9ce1fa98170f7edd`. Source-matched individual tickets are stored in `ddp-preflight/`.
