#!/bin/bash
set -euo pipefail
task_dir=/gpfs/users/linlizhi/komp-c128-triton-20261101
cd "$task_dir/work"
exec > "$task_dir/evidence/benchmark-runner.log" 2>&1
trap 'echo "$?" > "$task_dir/evidence/benchmark-exit-code.txt"' EXIT
python environment.py
python integration_c128.py
python benchmark_c128.py
