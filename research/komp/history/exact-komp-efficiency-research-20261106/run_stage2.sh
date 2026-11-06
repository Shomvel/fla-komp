#!/usr/bin/env bash
# Invoke only inside a separately authorized 8 x H100 job after Stage 1 passes.
set -euo pipefail
cd "$(dirname "$0")"
: "${EXACT_RESEARCH_FIXTURE:?Set frozen fixture}"
: "${EXACT_RESEARCH_RESULTS:?Set NEW results directory}"
: "${EXACT_RESEARCH_STAGE1:?Set completed stage1 results directory}"
python - "$EXACT_RESEARCH_STAGE1" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for case in ('full','tail'):
 for suffix in ('comparison','control-comparison'):
  assert json.loads((root/f'{case}-{suffix}.json').read_text())['status']=='passed'
PY
mkdir "$EXACT_RESEARCH_RESULTS"
for case in full tail; do
  torchrun --standalone --nproc_per_node=8 benchmark.py --variant baseline --fixture "$EXACT_RESEARCH_FIXTURE" --out "$EXACT_RESEARCH_RESULTS/$case-baseline" --case "$case" --mode check
  torchrun --standalone --nproc_per_node=8 benchmark.py --variant candidate --cache --fixture "$EXACT_RESEARCH_FIXTURE" --out "$EXACT_RESEARCH_RESULTS/$case-cache" --case "$case" --mode check
  python compare.py "$EXACT_RESEARCH_RESULTS/$case-baseline" "$EXACT_RESEARCH_RESULTS/$case-cache" --out "$EXACT_RESEARCH_RESULTS/$case-comparison.json"
done
for run in a1 b1 b2 a2; do
  extra=();variant=baseline
  if [[ "$run" == b* ]]; then variant=candidate;extra=(--cache --prune-gc);fi
  torchrun --standalone --nproc_per_node=8 benchmark.py --variant "$variant" "${extra[@]}" --fixture "$EXACT_RESEARCH_FIXTURE" --out "$EXACT_RESEARCH_RESULTS/bench-$run" --mode bench
 done
