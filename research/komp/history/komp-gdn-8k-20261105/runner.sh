#!/bin/bash
set -euo pipefail
cd /gpfs/users/linlizhi/komp-gdn-8k-20261105
exec > >(tee evidence/job.log) 2>&1
export FLA_DISABLE_BACKEND_DISPATCH=1 FLA_DISABLE_TENSOR_CACHE=1
nvidia-smi > evidence/nvidia-smi.txt
python environment.py > evidence/environment.json
python compare.py
echo completed > evidence/DONE
