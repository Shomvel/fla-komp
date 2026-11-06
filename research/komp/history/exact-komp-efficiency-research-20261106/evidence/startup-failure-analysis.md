# First startup failure

GPU allocation identified H100 80GB and the expected torch image. Fixture preparation failed before operator or training tests because `baseline/configs/data-manifest.json` was not in the frozen source manifest. All 658 listed source files had copied correctly; source provenance was insufficient to guarantee runtime-data dependency closure.

Static audit covered the explicit stage scripts, model import paths, fixture/data loader dependencies, optional memory-policy receipt, comparison paths and profiler outputs. Data manifest and audit are now explicit research inputs; a stdlib preflight verifies all checkpoint files, 253 training shards, tokenizer file hashes and exclusion hash before output creation. The CPU regression checks aggregated missing dependency reporting without side effects. The preflight passed on BP Dev before submitting r2.

The partial fixture and failed job logs are preserved. First job runtime reported 33 seconds; conservatively charged 60 seconds against the 2700-second total budget. r2 is capped at2640 seconds. The first exact owned job was deleted and absence verified before r2 submission. No numeric result was produced by r1.
