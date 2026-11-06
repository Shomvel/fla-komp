#!/bin/bash
set -euo pipefail
task_dir=/gpfs/users/linlizhi/komp-exact-physics-20261105
cd "$task_dir"
exec python run_trials.py > runner.log 2>&1
