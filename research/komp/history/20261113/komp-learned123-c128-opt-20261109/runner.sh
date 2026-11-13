#!/bin/bash
set -uo pipefail
TASK_ROOT=/gpfs/users/linlizhi/komp-learned123-c128-opt-20261109
cd "$TASK_ROOT"
exec > >(tee -a runner.log) 2>&1
trap 'echo "$?" > evidence/job-exit-code.txt' EXIT
export TASK_ROOT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$TASK_ROOT/snapshot:/gpfs/users/linlizhi/fla-komp-value-integration-20261106-r1/deps:/gpfs/users/linlizhi/komp-gsa-gdn-20261031-env/triton371"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python FLA_CI_ENV=0
export FLA_DISABLE_BACKEND_DISPATCH=1 FLA_DISABLE_TENSOR_CACHE=1
export TRITON_CACHE_DIR=/tmp/komp-learned123-c128-opt-20261109-triton
export CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi > evidence/environment-nvidia-smi.txt
python3 - <<'PY'
import os,json,sys,socket,hashlib
from pathlib import Path
import torch,triton,google.protobuf
from google.protobuf.internal.api_implementation import Type
r=Path(os.environ['TASK_ROOT'])
assert torch.cuda.device_count()==1
assert 'H100' in torch.cuda.get_device_name(0)
files=['snapshot/fla/ops/komp/block_fast.py','snapshot/fla/ops/komp/scan.py','models.py']
x=dict(hostname=socket.gethostname(),python=sys.version,torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,protobuf=google.protobuf.__version__,protobuf_backend=Type(),gpus=[torch.cuda.get_device_name(0)],source_sha256={p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in files},pythonpath=os.environ['PYTHONPATH'])
(r/'evidence/environment.json').write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps(x),flush=True)
PY
[ "$?" -eq 0 ] || exit 1
while true; do
  for f in queue/*.sh; do
    [ -f "$f" ] || continue
    id=$(basename "$f" .sh)
    [ -f "queue/$id.done" ] && continue
    [ -f queue/FAILED ] && continue
    date -u +%FT%TZ > "queue/$id.started"
    timeout 3600 bash "$f" > "queue/$id.out" 2>&1
    code=$?
    echo "$code" > "queue/$id.done"
    date -u +%FT%TZ > "queue/$id.finished"
    [ "$code" -eq 0 ] || echo "$id" > queue/FAILED
  done
  [ -f queue/STOP ] && exit 0
  sleep 2
done
