set -euo pipefail
python3 probe.py --mode operator --snapshot snapshot-baseline --tag baseline-operator --profile --warmup 4 --reps 10
