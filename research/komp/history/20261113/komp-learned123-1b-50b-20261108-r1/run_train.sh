#!/bin/bash
set -euo pipefail
export TASK_ROOT=/gpfs/users/linlizhi/komp-learned123-1b-50b-20261108-r1
: "${VC_TASK_INDEX:?Volcano env plugin required}"
: "${VC_TRAINER_NUM:?Volcano svc plugin required}"
: "${VC_TRAINER_HOSTS:?Volcano svc plugin required}"
test "$VC_TRAINER_NUM" = 4
export NODE_ROOT="$TASK_ROOT/nodes/$VC_TASK_INDEX"
mkdir -p "$NODE_ROOT" "$TASK_ROOT/logs"
cd "$TASK_ROOT"
exec > >(tee -a "$NODE_ROOT/worker.log") 2>&1
trap 'code=$?; echo "$code" > "$NODE_ROOT/exit.txt"' EXIT
compgen -G '/dev/infiniband/uverbs*' >/dev/null
test -c /dev/infiniband/rdma_cm
test "${NCCL_NET:-}" = IB
test "${NCCL_IB_DISABLE:-}" = 0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$TASK_ROOT/snapshot:/gpfs/users/linlizhi/fla-komp-value-integration-20261106-r1/deps:/gpfs/users/linlizhi/komp-gsa-gdn-20261031-env/triton371"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python FLA_CI_ENV=0
export FLA_DISABLE_BACKEND_DISPATCH=1 FLA_DISABLE_TENSOR_CACHE=1
export TRITON_CACHE_DIR="/tmp/komp-1b-50b-triton-$VC_TASK_INDEX"
export NCCL_ALGO=Ring NCCL_PROTO=Simple
export NCCL_DEBUG_FILE="$NODE_ROOT/nccl-%h-%p.log"
export TRAIN_MASTER_ADDR="${VC_TRAINER_HOSTS%%,*}"
python3 - <<'PYCODE'
import json,os,shutil,socket,subprocess,sys,hashlib
from pathlib import Path
import torch,google.protobuf
from google.protobuf.internal.api_implementation import Type
root=Path(os.environ['TASK_ROOT']);node=Path(os.environ['NODE_ROOT'])
assert shutil.disk_usage(root).free>64*2**30,'Insufficient GPFS headroom'
assert torch.cuda.device_count()==8
manifest=json.loads((root/'source-manifest.json').read_text())
for name,sha in manifest['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
receipt=dict(hostname=socket.gethostname(),node_rank=int(os.environ['VC_TASK_INDEX']),hosts=os.environ['VC_TRAINER_HOSTS'],master_addr=os.environ['TRAIN_MASTER_ADDR'],world_size=32,python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,protobuf=google.protobuf.__version__,protobuf_backend=Type(),gpus=[torch.cuda.get_device_name(i) for i in range(8)],network='IB',free_bytes=shutil.disk_usage(root).free,source_sha=hashlib.sha256((root/'source-manifest.json').read_bytes()).hexdigest())
assert all('H100' in gpu for gpu in receipt['gpus']),receipt['gpus']
(node/'runtime.json').write_text(json.dumps(receipt,indent=2)+'\n')
subprocess.run(['nvidia-smi'],check=True)
print(json.dumps(receipt),flush=True)
PYCODE
launch=(python3 -m torch.distributed.run --nnodes=4 --nproc_per_node=8 --node_rank="$VC_TASK_INDEX" --master_addr="$TRAIN_MASTER_ADDR" --max_restarts=0)
"${launch[@]}" --master_port=29941 preflight.py --kind block
python3 - <<'PYCODE'
import json,os
from pathlib import Path
root=Path(os.environ['TASK_ROOT'])
assert json.loads((root/'evidence/preflight-block.json').read_text())['status']=='passed'
for rank in range(32):assert json.loads((root/f'evidence/restore-rank{rank}.json').read_text())['status']=='passed'
PYCODE
"${launch[@]}" --master_port=29942 train.py --kind block
