# Block-KOMP acceptance complete

User requested testing the Opus implementation referenced by ../job. The job scripts point to code in /Users/shom/Downloads/komp-slot-attention-prototype; the tested implementation is komp_slot/block_fast.py. Its dependencies and test scripts were frozen under work/ before GPU execution. Original kernels were not modified.

RESULTS.md contains numerical and performance results. verification.json verifies all 457 checks, 16 benchmark rows with three measurements each, and source fingerprints. Exact tolerances and per-tensor errors are in work/evidence/. Precision tests include a relaxed comparison-to-v0 criterion for near-duplicate dictionaries; this is disclosed in the report.

Environment: dedicated H100 80GB, Torch 2.10.0a0, Triton 3.7.1, pinned container image in job.yaml. Source hashes in source-sha256.json, remote source/FLA fingerprint in provenance.json. CUDA Graph timings exclude compilation. No Depo/Brevo training was started in this test.

Remote root: /gpfs/users/linlizhi/komp-block-acceptance-20261101-r1. Job job-540a773e46f0-20261101052418 completed with exit code 0, evidence collected, exact alias deleted and absence verified. See terminal-status.json, cleanup.json, after-cleanup.json. GPFS files remain. No other jobs were changed.
