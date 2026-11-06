#!/bin/bash
set -u
cd /gpfs/users/linlizhi/komp-exact-opt2-20261105
mkdir -p queue evidence
nvidia-smi > evidence/nvidia-smi.txt 2>&1
python environment.py > evidence/environment.json
while true; do
  if [ ! -e queue/PAUSE ]; then
    for f in queue/*.sh; do
      [ -e "$f" ] || continue
      id=$(basename "$f" .sh)
      [ -e "queue/$id.done" ] && continue
      timeout 2400 bash "$f" > "queue/$id.out" 2>&1
      code=$?
      echo "$code" > "queue/$id.done"
      if [ "$code" -ne 0 ]; then
        touch queue/PAUSE
        break
      fi
    done
  fi
  [ -e queue/STOP ] && exit 0
  sleep 2
done
