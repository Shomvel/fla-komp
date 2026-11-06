#!/bin/bash
set -euo pipefail
local_root=/Users/shom/Downloads/komp-slot-attention-prototype/physics_lm/block_runs_v4/gsa2
remote_root=/gpfs/users/linlizhi/komp-gsa2-trials-20261101
rsync -az --exclude='*.pt' "linlizhi-bp-dev:$remote_root/results/" "$local_root/"
rsync -az --include='/*.json' --include='/*.txt' --include='/runner.log' --exclude='*' "linlizhi-bp-dev:$remote_root/" "$local_root/evidence/"
