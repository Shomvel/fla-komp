# Exact beta × M formal comparison

Two tasks, four cells each: beta0 {0.1, 0.2} × M {32, 64}. Reuse the completed best-LR beta .2/M32 checkpoints; run six new 100M-position candidates. Alpha .995, FP32 projection/state, BF16 network/head, seed 731, WD .1, Depo2 LR .001 and Brevo2 LR .002 remain fixed. Rank solely by full validation CE.

Width 512, 15 layers, 8 heads, head dimension 64 and FF1984 remain fixed. M32 has 100526392 parameters; M64 has 102742072. Initial dictionary keys and coefficient state double from 32 to 64 slots; M×M maps grow fourfold and Gram factorization arithmetic grows roughly eightfold. Actual warm benchmark and training peak memory/throughput are recorded separately.

The adapter constructs the original M32 model, preserving shared initialization, then extends only M-dependent decay projection and dictionary keys. M64 preserves the first32 original atoms and appends an orthogonal complement. An executable initialization audit checks every tensor, beta bias, alpha bias, and dictionary orthogonality. The frozen source snapshot is never edited.

Each candidate requires representative CUDA differential checks and a fresh eight-GPU preflight, including uneven/tail batches, model causality, rank-shaped single-GPU accumulation versus DDP, checkpoint/RNG restore, and real-data BF16 updates. The original near-collinear FP32 stress failure remains a known limitation, not an all-tests-passed claim. Each task runs beta .1/M32, beta .2/M64, then beta .1/M64 sequentially. Failure blocks new stages and retains evidence.
