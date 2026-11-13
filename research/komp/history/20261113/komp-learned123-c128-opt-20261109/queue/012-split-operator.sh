set -euo pipefail
python3 probe.py --mode operator --snapshot snapshot-split --tag split-operator --profile --checkpoint fixture/checkpoint-010240
