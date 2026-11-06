# Alpha .995 single-factor formal comparison

Two newly authorized 100M-parameter Exact-KOMP runs, one each for Depo2 and Brevo2. Each consumes the same original whole-document 100M next-token-position schedule and data order, seed731, global128-document batch, 8GPU DDP. Only initial alpha changes from .98 to .995: all15 `mix.decay.bias` tensors change from logit(.98^8) to logit(.995^8), and all other initial tensors are bitwise identical. Fresh8GPU preflights and actual finite optimizer steps are recorded in `evidence/training-startup.json`.

Original FP32 projection/state and BF16 network/head remain. Depo LR=.0005, Brevo LR=.001, WD=.1. Brevo's separate short-run LR=.002 winner is not used. The earlier alpha.995 short-run screen used FP64 projection; this experiment deliberately retains the original formal FP32 projection for a single-factor comparison. Original near-collinear FP32 stress failure remains a known limitation; no all-stress-tests-passed claim is made.

Source manifests and result/checkpoint paths are isolated here. The snapshot symlink refers to the original frozen snapshot without modification. Source-bound task-specific preflight directories avoid checkpoint/ticket races. Training runs end and release8GPU allocations before independent1GPU evaluation using the original fixed scorer and256probes. Original14-run results and tuning source are not overwritten, merged into this experiment, or rerun.

Remote root: `/gpfs/users/linlizhi/physics100m-exact-20261105/exact_alpha995_formal_20261105`. Read progress with `ssh linlizhi-bp-dev python3 - < probe.py`. Job aliases: `physics100m-alpha995-depo2-20261105` and `physics100m-alpha995-brevo2-20261105`.
