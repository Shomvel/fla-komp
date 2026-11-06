#!/bin/bash
# usage: train_c128.sh <runs-dir> <kinds comma> <seeds space>   (run inside $task_dir/snap/physics_lm)
set -e
RUNS=$1; KINDS=$2; SEEDS="$3"
export BLOCK_RUNS=$RUNS
python block_c128.py --preflight --kinds $KINDS
for seed in $SEEDS; do
  for kind in ${KINDS//,/ }; do
    python block_c128.py --kind $kind --seed $seed > $RUNS/$kind-seed$seed.log 2>&1 || { echo FAILED $kind $seed; tail -5 $RUNS/$kind-seed$seed.log; exit 1; }
    grep COMPLETE $RUNS/$kind-seed$seed.log | cut -c1-160
  done
done
