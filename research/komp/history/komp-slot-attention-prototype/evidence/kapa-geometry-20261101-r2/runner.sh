#!/bin/bash
set -euo pipefail
cd /gpfs/users/linlizhi/kapa-geometry-20261101-r2/work
exec > >(tee ../job.log) 2>&1
trap 'code=$?; echo "$code" > ../exit-code.txt' EXIT
nvidia-smi > ../nvidia-smi.txt
python - <<'PYENV'
import sys,json,torch,triton,google.protobuf
from google.protobuf.internal.api_implementation import Type
from pathlib import Path
Path('../environment.json').write_text(json.dumps(dict(python=sys.version,torch=str(torch.__version__),triton=triton.__version__,gpu=torch.cuda.get_device_name(),protobuf=google.protobuf.__version__,protobuf_backend=Type()),indent=2))
PYENV
python benchmark_kapa_geometry.py > ../benchmark.log 2>&1
