#!/bin/bash
set -euo pipefail
task_dir=/gpfs/users/linlizhi/komp-proj-swa-20261101
worker_dir="$task_dir/workers/r$((PROJ_WORKER+1))"
exec > >(tee "$worker_dir/runner.log") 2>&1
trap 'code=$?; echo "$code" > "$worker_dir/exit-code.txt"' EXIT
nvidia-smi > "$worker_dir/nvidia-smi.txt"
export BLOCK_RUNS="$task_dir/results"
export PHYSICS_DATA=/gpfs/users/linlizhi/softmax-10m-20261023/data
export PYTHONDONTWRITEBYTECODE=1
python "$task_dir/run_worker.py"
