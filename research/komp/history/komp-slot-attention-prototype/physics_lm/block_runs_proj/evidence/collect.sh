#!/bin/bash
set -euo pipefail
local_root=/Users/shom/Downloads/komp-slot-attention-prototype/physics_lm/block_runs_proj
remote_root=/gpfs/users/linlizhi/komp-proj-swa-20261101
rsync -az --exclude='*.pt' "linlizhi-bp-dev:$remote_root/results/" "$local_root/"
rsync -az --exclude='triton-cache' --exclude='*.pt' "linlizhi-bp-dev:$remote_root/workers/" "$local_root/evidence/workers/"
rsync -az --include='/*.json' --exclude='*' "linlizhi-bp-dev:$remote_root/" "$local_root/evidence/"
