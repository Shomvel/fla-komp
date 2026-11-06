#!/bin/bash
# Sequential GPU task runner: executes queue/<id>.sh once, writes <id>.out and <id>.done (exit code).
set -u
task_dir=/gpfs/users/linlizhi/komp-accel-20261101-dev2
cd "$task_dir"
mkdir -p queue work
nvidia-smi > runner-nvidia-smi.txt 2>&1 || true
python -c 'import sys,torch,triton;print(sys.version);print(torch.__version__,triton.__version__,torch.cuda.get_device_name())' > runner-env.txt 2>&1
echo "runner started $(date -u +%FT%TZ)" >> runner.log
while true; do
  for f in queue/*.sh; do
    [ -e "$f" ] || continue
    id=$(basename "$f" .sh)
    [ -e "queue/$id.done" ] && continue
    echo "start $id $(date -u +%FT%TZ)" >> runner.log
    (cd work && timeout 2400 bash "../$f") > "queue/$id.out" 2>&1
    echo $? > "queue/$id.done"
    echo "done $id $(date -u +%FT%TZ)" >> runner.log
  done
  [ -e queue/STOP ] && { echo "stop $(date -u +%FT%TZ)" >> runner.log; exit 0; }
  sleep 2
done
