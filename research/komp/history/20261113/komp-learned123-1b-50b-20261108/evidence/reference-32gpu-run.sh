#!/bin/bash
set -euo pipefail
export TASK_ROOT=/gpfs/users/linlizhi/dspark-adamw-lr1e4-32gpu-20261106-r1
: "${VC_TASK_INDEX:?Volcano env plugin required}"
: "${VC_TRAINER_NUM:?Volcano svc plugin required}"
: "${VC_TRAINER_HOSTS:?Volcano svc plugin required}"
test "$VC_TRAINER_NUM" = 4
export NODE_ROOT="$TASK_ROOT/nodes/$VC_TASK_INDEX"
mkdir -p "$NODE_ROOT/training" "$TASK_ROOT/training" "$TASK_ROOT/fixed-gpu-regression"
cd "$TASK_ROOT"
exec > >(tee "$NODE_ROOT/training/worker.log") 2>&1
trap 'code=$?; echo "$code" > "$NODE_ROOT/training/exit.txt"' EXIT
compgen -G '/dev/infiniband/uverbs*' >/dev/null
test -c /dev/infiniband/rdma_cm
test "${NCCL_NET:-}" = IB
test "${NCCL_IB_DISABLE:-}" = 0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NCCL_ALGO=Ring NCCL_PROTO=Simple
export NCCL_DEBUG_FILE="$NODE_ROOT/training/nccl-%h-%p.log"
export TRAIN_MASTER_ADDR="${VC_TRAINER_HOSTS%%,*}"
mkdir -p "$NODE_ROOT/cache"
for cache_dir in train-cuda train-triton train-tilelang train-deepgemm train-xdg; do
 cp -a "/gpfs/users/linlizhi/dspark-perf-20261105-r5/cache/$cache_dir" "$NODE_ROOT/cache/"
done
source "$TASK_ROOT/train_env.sh"
python3 - <<'PY'
import hashlib,json,os,shutil,socket,yaml
from pathlib import Path
root=Path(os.environ['TASK_ROOT']);node=Path(os.environ['NODE_ROOT'])
p=json.loads((root/'source-provenance.json').read_text())
for name,sha in p['files'].items():
 assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha,name
assert json.loads((root/'data-audit.json').read_text())['passed']
choice=json.loads((root/'performance-selection.json').read_text());assert choice['passed']
assert shutil.disk_usage(root).free>850_000_000_000
raw=yaml.safe_load((root/'train.yaml').read_text());raw['deployment']['trainer'].update(master_addr=os.environ['TRAIN_MASTER_ADDR'],node_rank=int(os.environ['VC_TASK_INDEX']))
(node/'train.yaml').write_text(yaml.safe_dump(raw,sort_keys=False))
from specforge.config import load_config
c=load_config(str(node/'train.yaml'))
assert c.training.learning_rate==1e-4
assert c.training.anchor_sampling=='all' and c.training.total_steps==118
assert c.training.accumulation_steps==4 and c.deployment.trainer.nnodes==4
assert c.training.anchor_forward_chunk_blocks==choice['selected_chunk']==64
assert json.loads(Path(c.model.draft_model_config).read_text())['gradient_checkpointing']==choice['selected_gradient_checkpointing']
(node/'resolved-config.json').write_text(c.model_dump_json(indent=2)+'\n')
(node/'launcher.json').write_text(json.dumps({'hostname':socket.gethostname(),'node_rank':int(os.environ['VC_TASK_INDEX']),'hosts':os.environ['VC_TRAINER_HOSTS'],'master_addr':os.environ['TRAIN_MASTER_ADDR'],'world_size':32,'network':'IB'},indent=2)+'\n')
PY
python3 "$TASK_ROOT/preflight_train.py" --gpu
cd "$TASK_ROOT/source"
launch=(python3 -m torch.distributed.run --nnodes=4 --nproc_per_node=8 --node_rank="$VC_TASK_INDEX" --master_addr="$TRAIN_MASTER_ADDR" --max_restarts=0)
"${launch[@]}" --master_port=29852 "$TASK_ROOT/fixed_gate.py" --output "$TASK_ROOT/fixed-gpu-regression"
python3 - <<'PY'
import json,os
from pathlib import Path
root=Path(os.environ['TASK_ROOT'])
for rank in range(32):
 r=json.loads((root/f'fixed-gpu-regression/rank{rank}.json').read_text())
 assert r['passed'] and r['world_size']==32
 assert r['streamed']['optimizer_steps']==2
PY
"${launch[@]}" --master_port=29853 "$TASK_ROOT/audit_train.py" train --config "$NODE_ROOT/train.yaml"
if [ "$VC_TASK_INDEX" = 0 ]; then
python3 - <<'PY'
import json,os,time
from pathlib import Path
from specforge.training.nfc_checkpoint import read_manifest
root=Path(os.environ['TASK_ROOT']);ckpt=root/'training/denovo-v41-all-anchors-step118'
deadline=time.time()+300
while not all((root/f'training/complete-rank{rank}.json').exists() for rank in range(32)):
 assert time.time()<deadline,'timed out waiting for all rank receipts'
 time.sleep(1)
m=read_manifest(ckpt);assert m['training']['global_step']==118 and m['training']['world_size']==32
for rank in range(32):
 r=json.loads((root/f'training/complete-rank{rank}.json').read_text())
 assert r['passed'] and r['optimizer_steps']==118 and r['training_micro']==472
(root/'training/result.json').write_text(json.dumps({'complete':True,'global_step':118,'world_size':32,'checkpoint':str(ckpt),'train_trajectories':1326,'final_checkpoint_restore_verified':False},indent=2)+'\n')
PY
fi
