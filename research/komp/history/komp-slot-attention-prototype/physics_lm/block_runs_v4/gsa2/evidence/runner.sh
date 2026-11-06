#!/bin/bash
set -euo pipefail
task_dir=/gpfs/users/linlizhi/komp-gsa2-trials-20261101
cd "$task_dir"
exec > >(tee "$task_dir/runner.log") 2>&1
trap 'code=$?; echo "$code" > "$task_dir/exit-code.txt"' EXIT
nvidia-smi > "$task_dir/nvidia-smi.txt"
export GSA2_TRIALS="$task_dir/results"
export BLOCK_RUNS="$task_dir/results"
export TRIAL_BASELINES="$task_dir/results/baselines"
export KAPA_RUNS="$task_dir/results/kapa"
export PHYSICS_DATA=/gpfs/users/linlizhi/softmax-10m-20261023/data
mkdir -p "$task_dir/results/logs" "$task_dir/results/baselines" "$task_dir/results/kapa"
python "$task_dir/run_trials.py"
