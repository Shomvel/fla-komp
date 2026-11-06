#!/bin/bash
set -uo pipefail
task_dir=/gpfs/users/linlizhi/komp-c128-triton-20261101
cd "$task_dir"
exec > >(tee runner.log) 2>&1
trap 'echo "$?" > exit-code.txt' EXIT
nvidia-smi > evidence/nvidia-smi.txt
while true; do
  for f in queue/*.sh; do
    [ -f "$f" ] || continue
    id=$(basename "$f" .sh)
    [ -f "queue/$id.done" ] && continue
    if [ -f queue/FAILED ]; then continue; fi
    (cd work && timeout 2400 bash "../$f") > "queue/$id.out" 2>&1
    code=$?
    echo "$code" > "queue/$id.done"
    [ "$code" -eq 0 ] || echo "$id" > queue/FAILED
  done
  [ -f queue/STOP ] && exit 0
  sleep 2
done
