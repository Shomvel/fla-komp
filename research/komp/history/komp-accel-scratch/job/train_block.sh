#!/bin/bash
# usage: train_block.sh <snapshot-name> <runs-dir> <kinds comma> <seeds space>
set -e
R=/gpfs/users/linlizhi/komp-accel-20261101-dev
SNAP=$R/$1; RUNS=$2; KINDS=$3; SEEDS="$4"
rm -rf $SNAP && mkdir -p $SNAP
cp -a $R/work/komp_slot $R/work/komp_bench $R/work/benchmark_fla.py $R/work/physics_lm $SNAP/
cd $SNAP/physics_lm
export BLOCK_RUNS=$RUNS
python block_experiment.py --preflight --kinds $KINDS
for kind in ${KINDS//,/ }; do
  for seed in $SEEDS; do
    python block_experiment.py --kind $kind --seed $seed > $RUNS/$kind-seed$seed.log 2>&1 || { echo FAILED $kind $seed; tail -5 $RUNS/$kind-seed$seed.log; exit 1; }
    grep COMPLETE $RUNS/$kind-seed$seed.log | cut -c1-200
  done
done
