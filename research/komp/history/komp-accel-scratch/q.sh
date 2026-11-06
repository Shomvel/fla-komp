#!/bin/bash
# usage: q.sh <id> <command...>  — sync code, enqueue command, wait for completion, print output
set -e
id=$1; shift
R=${KOMP_R:-/gpfs/users/linlizhi/komp-accel-20261101-dev}
cd /Users/shom/Downloads/komp-slot-attention-prototype
rsync -az --exclude __pycache__ --exclude .plot-env --exclude evidence --exclude physics_lm --exclude '*.png' --exclude '*.svg' ./ linlizhi-bp-dev:$R/work/
printf 'set -e\nmkdir -p evidence\n%s\n' "$*" | ssh linlizhi-bp-dev "cat > $R/queue/$id.sh"
for i in $(seq 1 ${WAIT:-600}); do
  if ssh linlizhi-bp-dev "test -e $R/queue/$id.done"; then break; fi
  sleep 5
done
ssh linlizhi-bp-dev "echo EXIT \$(cat $R/queue/$id.done 2>/dev/null); grep -v -e pynvml -e 'import pynvml' $R/queue/$id.out | tail -n ${TAIL:-80}"
