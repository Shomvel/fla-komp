# Short convolution experiment receipts

Experiment: three existing mixers (KOMP GSA-slot, GSA, GDN), with/without the same FLA width-4 q/k/v causal depthwise convolution + SiLU, seeds 731 and 732.

Base source is the immutable decay-training archive at `../decay-20261031/training-source.tgz`; overlays contain the new experiment only. Existing kernels and historical training evidence are not edited.

First submission `komp-shortconv-20261031-r1` / `job-76cb9e4e7335-20261031223455` failed before GPU numerical checks or training because the frozen archive omitted `validate_data.py`. Static inspection also identified its dependency `physics_port.py`; both were included in r2. The package member check, Python syntax checks and verifier import all passed before resubmission. The failed log and status are retained. No model result from r1 is counted.

Second submission: `komp-shortconv-20261031-r2` / `job-caf5cfaf6787-20261031223724`, project WeaverRSI, three GPUs. Remote root `/gpfs/users/linlizhi/komp-shortconv-20261031-r2`. Checkpoints remain under `physics_lm/shortconv_runs/results/<arm>-seed<seed>/model.pt` there.

FLA source is pinned to `97874af2400916e2885c33ea30559a70ff686b66`. The remote and local `fla/modules/causal_conv1d/ops.py` SHA256 both equal `9b59b5004b843bdface81164c0850c0b0e791df688f046fcee1c1810e06f7b16`.

See `../../physics_lm/shortconv_runs/PLAN.md` for prespecified conditions. Final status, raw logs, source/checkpoint hashes and cleanup receipts are collected here; model manifests, metrics and individual generations are under `../../physics_lm/shortconv_runs/results/`.

All 12 planned runs completed. The six initial models and 12 trained checkpoints passed 234 prefix-intervention checks in total (13 cuts each). All 12 checkpoint reload CE errors were exactly zero. Convolution FP32/BF16 output/input-gradient/weight-gradient reference checks and streaming-state checks passed. Both r1 and r2 jobs were deleted by exact alias and independently queried absent; only this experiment's jobs were cleaned.
