#!/usr/bin/env bash
# Invoke only inside a separately authorized 1 x H100 job.
set -euo pipefail
cd "$(dirname "$0")"
: "${EXACT_RESEARCH_FIXTURE:?Set to the frozen private fixture directory}"
: "${EXACT_RESEARCH_RESULTS:?Set to a NEW private output directory}"
mkdir "$EXACT_RESEARCH_RESULTS"
python operator_check.py --prune-gc --out "$EXACT_RESEARCH_RESULTS/operator.json"
for case in full tail; do
  python benchmark.py --variant baseline --fixture "$EXACT_RESEARCH_FIXTURE" --out "$EXACT_RESEARCH_RESULTS/$case-baseline" --case "$case" --mode check
  python benchmark.py --variant candidate --fixture "$EXACT_RESEARCH_FIXTURE" --out "$EXACT_RESEARCH_RESULTS/$case-cache-off" --case "$case" --mode check
  python compare.py "$EXACT_RESEARCH_RESULTS/$case-baseline" "$EXACT_RESEARCH_RESULTS/$case-cache-off" --out "$EXACT_RESEARCH_RESULTS/$case-control-comparison.json"
  python benchmark.py --variant candidate --cache --prune-gc --fixture "$EXACT_RESEARCH_FIXTURE" --out "$EXACT_RESEARCH_RESULTS/$case-cache-on" --case "$case" --mode check
  python compare.py "$EXACT_RESEARCH_RESULTS/$case-baseline" "$EXACT_RESEARCH_RESULTS/$case-cache-on" --out "$EXACT_RESEARCH_RESULTS/$case-comparison.json"
done
python benchmark_layer.py --variant baseline --fixture "$EXACT_RESEARCH_FIXTURE" --out "$EXACT_RESEARCH_RESULTS/layer-baseline"
python benchmark_layer.py --variant candidate --cache --prune-gc --fixture "$EXACT_RESEARCH_FIXTURE" --out "$EXACT_RESEARCH_RESULTS/layer-cache"
# Same device/process topology, ABBA order, separate fresh processes and source variants.
for run in a1 b1 b2 a2; do
  extra=();variant=baseline
  if [[ "$run" == b* ]]; then variant=candidate;extra=(--cache --prune-gc);fi
  python benchmark.py --variant "$variant" "${extra[@]}" --fixture "$EXACT_RESEARCH_FIXTURE" --out "$EXACT_RESEARCH_RESULTS/bench-$run" --mode bench
 done
python benchmark.py --variant baseline --fixture "$EXACT_RESEARCH_FIXTURE" --out "$EXACT_RESEARCH_RESULTS/profile-baseline" --mode profile
python benchmark.py --variant candidate --cache --prune-gc --fixture "$EXACT_RESEARCH_FIXTURE" --out "$EXACT_RESEARCH_RESULTS/profile-cache" --mode profile

python benchmark.py --variant candidate --cache --fixture "$EXACT_RESEARCH_FIXTURE" --out "$EXACT_RESEARCH_RESULTS/bench-cache-only" --mode bench
python benchmark.py --variant candidate --prune-gc --fixture "$EXACT_RESEARCH_FIXTURE" --out "$EXACT_RESEARCH_RESULTS/bench-gc-only" --mode bench
python summarize_bench.py "$EXACT_RESEARCH_RESULTS" --out "$EXACT_RESEARCH_RESULTS/speed-summary.json"
