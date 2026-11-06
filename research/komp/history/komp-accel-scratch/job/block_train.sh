set -e
R=/gpfs/users/linlizhi/komp-accel-20261031-dev
# Frozen copy so later kernel edits cannot change running trials.
rm -rf $R/train-block && mkdir -p $R/train-block
cp -a $R/work/komp_slot $R/work/komp_bench $R/work/benchmark_fla.py $R/work/physics_lm $R/train-block/
cd $R/train-block/physics_lm
python block_experiment.py --preflight
for kind in block_c32 block_c16 block_c64; do
  for seed in 731 732; do
    python block_experiment.py --kind $kind --seed $seed > block_runs/$kind-seed$seed.log 2>&1
    grep COMPLETE block_runs/$kind-seed$seed.log | cut -c1-400
  done
done
