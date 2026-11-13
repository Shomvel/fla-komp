set -euo pipefail
python3 probe.py --mode operator --snapshot snapshot-slot --tag slot-operator --profile --checkpoint fixture/checkpoint-010240
